"""CosyVoice2 with the LM and decoder in separate processes. Work in progress.

modal_cosyvoice2.py stays as the baseline: the benchmark number has to be what
CosyVoice2 ships. This file replaces its inference loop.

Why. The LM emits 258 tokens in 4003 ms alone and 7754 ms with the decoder
running alongside it, same GPU (cosyvoice2-profile.md). Both live in one process
and Python threads share one interpreter lock, so the LM only gets a turn when
the decoder is between ops. Separate processes have separate locks.

Shape:

  A (this process)   frontend, LM, dispatches chunks, never waits for audio
  B (worker)         flow + hift, decodes chunks, sends audio back

A's generation loop does not block on B. It pushes a chunk onto a queue and
keeps generating. A reader thread collects the audio, and it spends its life
blocked on get(), which holds no lock.

This depends on CosyVoice internals, hence the pinned commit.
"""

import modal

MODEL = "iic/CosyVoice2-0.5B"
REPO = "/opt/CosyVoice"
# Pinned. This file drives CosyVoice internals, so a rebuild must not be able to
# change llm.inference or token2wav underneath it. Update deliberately.
#   git ls-remote https://github.com/FunAudioLLM/CosyVoice main
COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"

# One utterance per run, so the id the decoder keys its hift cache on is fixed.
UTTERANCE = "split-spike"

image = (
    modal.Image.micromamba(python_version="3.10")
    # pynini via conda-forge. pip builds of it fail on glibc and cost an hour.
    .micromamba_install("pynini=2.1.5", channels=["conda-forge"])
    .apt_install("git", "ffmpeg", "sox", "libsox-dev", "build-essential")
    .run_commands(
        f"git clone --recursive https://github.com/FunAudioLLM/CosyVoice {REPO}",
        f"cd {REPO} && git checkout {COMMIT} && git submodule update --init --recursive",
        # setuptools 81 dropped pkg_resources and one of CosyVoice's requirements
        # still imports it from its setup.py. PIP_CONSTRAINT is the pin that
        # reaches inside pip's isolated build environment; installing an older
        # setuptools alongside does not, because the build env builds its own.
        "echo 'setuptools<81' > /tmp/constraint.txt",
        "pip install 'setuptools<81' wheel",
        f"PIP_CONSTRAINT=/tmp/constraint.txt pip install -r {REPO}/requirements.txt",
        # DeepSpeed is a training dependency. transformers imports it whenever it
        # is installed, and importing it compiles CUDA ops, which needs nvcc that
        # this image does not carry. Inference never touches it.
        "pip uninstall -y deepspeed",
    )
    .pip_install("modelscope")
    .env({"PYTHONPATH": f"{REPO}:{REPO}/third_party/Matcha-TTS"})
)

app = modal.App("cosyvoice2-split-spike", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)
out = modal.Volume.from_name("s2s-results", create_if_missing=True)

# What the decoder needs from the frontend. Sent once, before any chunk.
PROMPT_KEYS = ("flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding")


def load_model():
    """Builds CosyVoice2. Both processes call this and each uses half of it."""
    import logging

    from modelscope import snapshot_download

    from cosyvoice.cli.cosyvoice import CosyVoice2

    # Importing cosyvoice calls logging.basicConfig(level=DEBUG), which also
    # switches on Modal's HTTP/2 client logger and buries everything under gRPC
    # header dumps. Undo it here, where both processes pass through.
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("h2", "hpack", "grpclib", "modal", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    model_dir = snapshot_download(MODEL, local_dir=f"/cache/{MODEL}")
    return CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)


def move(obj, device):
    """Walks a structure and moves any tensor in it."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move(v, device) for v in obj)
    return obj


# --------------------------------------------------------------------------
# B: the decoder process
# --------------------------------------------------------------------------

def decoder_process(requests, replies, gpu: int | None) -> None:
    """Speech tokens in, audio out. The LM in this process is idle weight.

    The hift cache lives here, so successive chunks of one utterance decode
    continuously rather than restarting.

    `gpu` pins this process to one card. Setting CUDA_VISIBLE_DEVICES before
    torch is imported means the process sees a single device and calls it
    cuda:0, so nothing downstream has to know which card it got. It has to
    happen here rather than in the parent, because spawn re-imports the module
    and torch must not already be initialised.

    Unlike the threaded attempt at a second GPU, this adds no cross-device
    copies: tensors already round-trip through CPU to be pickled across the
    process boundary.
    """
    import os

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    import torch

    print(f"decoder: loading (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    cosyvoice = load_model()
    model = cosyvoice.model
    prompt = None
    print(f"decoder: ready on {model.device}", flush=True)
    replies.put(("ready", None))

    while True:
        message = requests.get()
        if message is None:
            return
        kind, payload = message

        if kind == "prompt":
            prompt = move(payload, model.device)
            # tts() seeds these per utterance before its first decode. We drive
            # token2wav directly, so nobody has done it for us. hift_cache_dict
            # is the one it reads on entry; the rest are seeded to match.
            for name, initial in (
                ("hift_cache_dict", None),
                ("mel_overlap_dict", None),
                ("flow_cache_dict", None),
                ("tts_speech_token_dict", []),
                ("llm_end_dict", False),
            ):
                slot = getattr(model, name, None)
                if isinstance(slot, dict):
                    slot[UTTERANCE] = initial
            continue

        tokens, token_offset, finalize = payload
        audio = model.token2wav(
            token=torch.tensor(tokens).unsqueeze(0).to(model.device),
            prompt_token=prompt["flow_prompt_speech_token"],
            prompt_feat=prompt["prompt_speech_feat"],
            embedding=prompt["flow_embedding"],
            token_offset=token_offset,
            uuid=UTTERANCE,
            finalize=finalize,
        )
        replies.put(("audio", audio.cpu()))


# --------------------------------------------------------------------------
# A: frontend, LM, dispatch
# --------------------------------------------------------------------------

def collect(replies, sink, events) -> None:
    """Reads audio as it arrives. Lives blocked on get(), which holds no lock."""
    import time

    while True:
        kind, audio = replies.get()
        if kind == "stop":
            return
        events.append((f"chunk {len(sink)} out", time.monotonic()))
        sink.append(audio)


def run_split(cosyvoice, text, prompt_wav, requests, replies, events):
    """One piece at a time, the way inference_cross_lingual does it.

    The library splits English text into 60 to 80 token pieces and runs each
    through the LM and the decoder in turn. Feeding the whole paragraph as one
    sequence would be a different workload, and a faster one, since it never
    pays the gap between pieces.
    """
    audio: list = []
    # Carried across pieces deliberately. CosyVoice2Model.tts grows
    # self.token_hop_len and never puts it back, so piece two starts at
    # whatever piece one left behind, with a first chunk four times larger.
    # Replicated here so the arms stay comparable. Resetting it is a fix to
    # send upstream, not one to quietly take credit for.
    state = {"hop": cosyvoice.model.token_hop_len}
    for piece in cosyvoice.frontend.text_normalize(text, split=True):
        run_piece(cosyvoice, piece, prompt_wav, requests, replies, events,
                  audio, state)
    return audio


def run_piece(cosyvoice, text, prompt_wav, requests, replies, events,
              audio, state):
    """Our own streaming loop, in place of CosyVoiceModel.tts().

    The library's version runs the LM on a thread and polls a shared list every
    100ms. Here the LM runs on this thread and dispatches directly, so there is
    no polling interval to wait out and nothing to poll.
    """
    import threading
    import time

    import torch

    model = cosyvoice.model
    # The empty id means "clone from the wav" rather than using a speaker saved
    # earlier under a name.
    inputs = cosyvoice.frontend.frontend_cross_lingual(
        text, prompt_wav, cosyvoice.sample_rate, ""
    )
    print("frontend keys:", sorted(inputs), flush=True)

    requests.put(("prompt", move({k: inputs[k] for k in PROMPT_KEYS}, "cpu")))

    start = len(audio)
    reader = threading.Thread(target=collect, args=(replies, audio, events))
    reader.start()

    # The chunk schedule, copied from CosyVoice2Model.tts rather than
    # simplified. Each decode re-runs the whole sequence and trims by
    # token_offset, so a different schedule means a different amount of decoder
    # work and the comparison stops being like for like.
    #
    # The first chunk is longer than the rest. The flow decoder blocks the
    # sequence in token_hop_len-sized pieces counted from the first *prompt*
    # token, and a prompt rarely ends on a block boundary, so the first
    # generated chunk absorbs the remainder and everything after it lands
    # cleanly. Omitting this pad makes first audio look faster than the
    # library's for no reason other than a smaller first chunk.
    lookahead = model.flow.pre_lookahead_len
    base = state["hop"]                           # 25, then whatever the
                                                  # previous piece left
    scale = model.stream_scale_factor             # 2
    max_hop = model.token_max_hop_len             # 4 * 25
    prompt_len = inputs["flow_prompt_speech_token"].shape[1]
    pad = (-prompt_len) % base
    hop = base + pad
    print("[run_split] hop=%s (base %s + pad %s) lookahead=%s max_hop=%s"
          % (hop, base, pad, lookahead, max_hop), flush=True)

    tokens: list = []
    token_offset = 0
    sent = 0
    device = model.device

    # The frontend only returns what applies. Cross-lingual has no prompt
    # transcript, so `prompt_text` is simply absent and tts() would have filled
    # in an empty tensor. The _len arguments are derived from shapes, not
    # returned by the frontend.
    def arg(key, rows=1, cols=0, dtype=torch.int32):

        default = torch.zeros(rows, cols, dtype=dtype)
        return inputs.get(key, default).to(device)

    text_in = arg("text")
    prompt_text = arg("prompt_text")
    prompt_token = arg("llm_prompt_speech_token")

    def length(t):
        return torch.tensor([t.shape[1]], dtype=torch.int32).to(device)

    for token in model.llm.inference(
        text=text_in,
        text_len=length(text_in),
        prompt_text=prompt_text,
        prompt_text_len=length(prompt_text),
        prompt_speech_token=prompt_token,
        prompt_speech_token_len=length(prompt_token),
        embedding=arg("llm_embedding", rows=0, cols=192, dtype=torch.float32),
    ):
        if not tokens:
            events.append(("llm first token", time.monotonic()))
        tokens.append(token)

        # Dispatch and carry on. Not waiting for the audio is the whole point:
        # the decode happens in B while this loop keeps generating.
        if len(tokens) - token_offset >= hop + lookahead:
            requests.put(
                ("chunk", (tokens[: token_offset + hop + lookahead], token_offset, False))
            )
            token_offset += hop
            # The library grows its own token_hop_len and uses the grown value
            # from the next chunk on, so the pad applies once and never again.
            base = min(base * scale, max_hop)
            hop = base
            sent += 1

    events.append((f"llm last token ({len(tokens)})", time.monotonic()))
    requests.put(("chunk", (tokens, token_offset, True)))
    sent += 1

    state["hop"] = base

    # Everything is dispatched; wait for this piece's audio to come back.
    deadline = time.monotonic() + 300
    while len(audio) - start < sent:
        if time.monotonic() > deadline:
            raise RuntimeError(f"got {len(audio) - start} of {sent} chunks back")
        time.sleep(0.005)
    replies.put(("stop", None))
    reader.join(timeout=5)


# Two cards requested so the decoder can be pinned to the second one. With
# --decoder-gpu unset the second idles, which keeps both arms on one image.
@app.function(gpu="A10G:2", volumes={"/cache": cache, "/out": out}, timeout=3600)
def synth(text: str, prompt_wav: str, warmup: bool, decoder_gpu: int | None) -> list[str]:
    import multiprocessing as mp
    import time

    import torchaudio

    cosyvoice = load_model()

    # spawn, not fork. A forked process inherits a CUDA context it cannot use.
    ctx = mp.get_context("spawn")
    requests, replies = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(
        target=decoder_process, args=(requests, replies, decoder_gpu), daemon=True
    )
    worker.start()
    try:
        kind, _ = replies.get(timeout=600)
    except Exception:
        alive = worker.is_alive()
        raise RuntimeError(
            f"decoder never reported ready (alive={alive}, exit={worker.exitcode}). "
            "Its traceback is in the container log above this line."
        )
    assert kind == "ready", kind
    print("decoder process up", flush=True)

    if warmup:
        # Warms both processes. B's first decode pays for kernel selection over
        # there just as the LM's first token does here.
        run_split(cosyvoice, "Warming up.", prompt_wav, requests, replies, [])
        print("warmed", flush=True)
        print("[after warmup] token_hop_len=%s" % cosyvoice.model.token_hop_len,
              flush=True)

    events: list = []
    t0 = time.monotonic()
    chunks = run_split(cosyvoice, text, prompt_wav, requests, replies, events)
    total = (time.monotonic() - t0) * 1000

    paths = []
    for i, audio in enumerate(chunks):
        path = f"/out/cosyvoice2_split_{i:03d}.wav"
        torchaudio.save(path, audio, cosyvoice.sample_rate)
        paths.append(path)

    print("\n--- timeline, ms from synthesis start ---", flush=True)
    previous = t0
    for label, when in sorted(events, key=lambda e: e[1]):
        print(f"{(when - t0) * 1000:8.0f}  (+{(when - previous) * 1000:6.0f})  {label}")
        previous = when
    print(f"total {total:.0f} ms, {len(paths)} chunks", flush=True)

    requests.put(None)
    worker.join(timeout=30)
    out.commit()
    return paths


@app.local_entrypoint()
def main(
    text: str = (
            "The capital of France is Paris. "
            "It has been the seat of government since the tenth century. "
            "Today it is home to just over two million people."
        ),
    # text: str = (
    #          "Photosynthesis converts light energy into chemical energy. "
    #          "Plants absorb sunlight through chlorophyll in their leaves, "
    #          "then use that energy to combine carbon dioxide from the air with water drawn up from the roots. "
    #          "The result is glucose, which the plant uses for growth, and oxygen, "
    #          "which is released back into the atmosphere. "
    #          "Almost every food chain on the planet starts with this reaction, "
    #          "which is why a change in plant cover affects far more than the plants themselves."
    # ),
    prompt_wav: str = f"{REPO}/asset/zero_shot_prompt.wav",
    warmup: bool = False,
    # Which card the decoder process gets. Leave it and the decoder shares
    # cuda:0 with the LM; set it to 1 and the two stop sharing a CUDA context,
    # which is the remaining suspect for the 4.2ms per token that separate
    # processes alone did not recover.
    decoder_gpu: int = -1,
):
    print("\n".join(
        synth.remote(text, prompt_wav, warmup, None if decoder_gpu < 0 else decoder_gpu)
    ))
