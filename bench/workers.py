"""One process per stage, one GPU per process.

Stages could be plain objects in the harness process, and for `path: batch`
nothing would contend, since the stages run one after another. The streaming
paths are different: the LLM and the TTS overlap, and Python threads share one
interpreter lock, so they would take turns rather than run together. Profiling
CosyVoice2 measured that directly, where its own LM went from 15.6 to 23.6 ms
per token with a decoder thread alive beside it.

So every stage gets a process, in every config, including the ones where it
buys nothing. Homogeneity is the point: if `batch` ran in-process and
`stream_gen` did not, the difference between them would include the process
boundary and not just the overlap.

Each worker pins itself to one card with CUDA_VISIBLE_DEVICES before torch is
imported, so it sees a single device and calls it cuda:0. That has to happen
inside the child, because spawn re-imports the module and torch must not be
initialised yet.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from typing import Any, AsyncIterator


class WorkerError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# The child process
# --------------------------------------------------------------------------

def _serve(kind: str, cfg: dict, gpu: int | None, requests, replies, events) -> None:
    """Loads one stage and answers requests until told to stop."""
    import os
    import sys

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # When spawned with a separate venv executable, multiprocessing spawn
    # overwrites sys.path with the parent process's sys.path, stripping the
    # child venv's own site-packages. Re-add the venv's site-packages and place
    # them ahead of the parent's site-packages so the venv's dependencies take
    # precedence.
    venv_python = cfg.get("venv_python", "")
    prefix = sys.prefix if sys.prefix != sys.base_prefix else (
        os.path.dirname(os.path.dirname(os.path.abspath(venv_python))) if venv_python else None
    )
    if prefix:
        import site
        sp_dirs = []
        if hasattr(site, "getsitepackages"):
            sp_dirs.extend(site.getsitepackages([prefix]))
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        sp_dirs.append(os.path.join(prefix, "lib", py_ver, "site-packages"))
        sp_dirs.append(os.path.join(prefix, "Lib", "site-packages"))
        for sp in sp_dirs:
            if os.path.isdir(sp):
                site.addsitedir(sp)
        venv_paths = [p for p in sys.path if p.startswith(prefix)]
        # Filter out ANY site-packages that do not belong to this venv
        other_paths = [
            p for p in sys.path 
            if not p.startswith(prefix) 
            and "site-packages" not in p
        ]
        sys.path[:] = venv_paths + other_paths
        
        # DEBUG/FIX: If any modules leaked during the `spawn` unpickling phase
        # before we scrubbed sys.path, they must be evicted so they can be
        # correctly re-imported from the scrubbed venv path!
        import sys
        for mod in list(sys.modules.keys()):
            if mod.startswith("numpy") or mod.startswith("torch"):
                del sys.modules[mod]

    print(f"[{kind}] worker started, executable={sys.executable}", flush=True)

    asyncio.run(_serve_async(kind, cfg, requests, replies, events))


async def _serve_async(kind: str, cfg: dict, requests, replies, events) -> None:
    from . import stages

    stage = stages.build(kind, cfg, {})
    await stage.load()
    session = None
    replies.put(("ready", None))

    while True:
        message = await asyncio.to_thread(requests.get)
        if message is None:
            return
        op, payload = message

        try:
            if op == "warmup":
                await stage.warmup()
                replies.put(("done", None))

            elif op == "asr_open":
                session = stage.new_session()
                replies.put(("done", None))

            elif op == "asr_frame":
                # No reply. Frames arrive at 50/s and a round trip each would
                # cost more than the work. A partial goes on the events queue
                # with the instant it appeared, stamped here rather than when
                # the harness gets round to reading it. Both processes share a
                # host, so the clocks are comparable.
                if await session.accept(payload):
                    events.put(("partial", time.monotonic()))

            elif op == "asr_final":
                replies.put(("done", await session.final()))

            elif op == "llm":
                n = 0
                async for token in stage.generate(**payload):
                    n += 1
                    replies.put(("item", token))
                replies.put(("done", n))

            elif op == "tts":
                n = 0
                async for audio in stage.synth(**payload):
                    n += 1
                    replies.put(("item", audio))
                replies.put(("done", n))

            else:
                replies.put(("error", f"unknown op {op!r}"))

        except Exception as exc:  # noqa: BLE001
            import traceback

            replies.put(("error", f"{kind}/{op}: {exc}\n{traceback.format_exc()}"))


# --------------------------------------------------------------------------
# The harness side
# --------------------------------------------------------------------------

class StageWorker:
    """Handle on one stage process."""

    def __init__(self, kind: str, cfg: dict, gpu: int | None):
        self.kind = kind
        self.cfg = cfg
        self.gpu = gpu
        self.proc: mp.process.BaseProcess | None = None
        self.requests = None
        self.replies = None
        self.events = None

    async def start(self, timeout: float = 1800.0) -> None:
        # spawn, not fork. A forked process inherits a CUDA context it cannot use.
        ctx = mp.get_context("spawn")
        # If the stage config names a separate venv, run the child under that
        # Python. The Queue IPC is unaffected: multiprocessing serialises handles
        # through its own bootstrap, not through the Python executable.
        venv_python = self.cfg.get("venv_python", "")
        if venv_python:
            import os
            if not os.path.isfile(venv_python):
                raise RuntimeError(
                    f"{self.kind}: venv_python={venv_python!r} does not exist. "
                    "Rebuild the Modal image so /opt/tts-venv is created."
                )
            ctx.set_executable(venv_python)
            print(f"  {self.kind} worker will use venv: {venv_python}", flush=True)
        self.requests, self.replies, self.events = ctx.Queue(), ctx.Queue(), ctx.Queue()
        # Not daemonic. vLLM's AsyncLLM spawns its own engine core child, and
        # a daemonic process is not allowed children. Cleanup is explicit
        # instead, in the runner's finally.
        self.proc = ctx.Process(
            target=_serve,
            args=(self.kind, self.cfg, self.gpu, self.requests, self.replies, self.events),
        )
        self.proc.start()
        kind, payload = await self._recv(timeout)
        if kind != "ready":
            raise WorkerError(f"{self.kind} failed to start: {payload}")

    async def stop(self) -> None:
        if self.proc is None:
            return
        self.requests.put(None)
        await asyncio.to_thread(self.proc.join, 30)
        if self.proc.is_alive():
            self.proc.terminate()

    # ---- messaging -------------------------------------------------------

    async def _recv(self, timeout: float = 600.0):
        def get():
            try:
                return self.replies.get(timeout=timeout)
            except Exception:
                alive = self.proc.is_alive() if self.proc else False
                return ("error", f"{self.kind} silent for {timeout}s (alive={alive})")

        return await asyncio.to_thread(get)

    def send(self, op: str, payload: Any = None) -> None:
        """Fire and forget. Used for audio frames, which expect no reply."""
        self.requests.put((op, payload))

    async def call(self, op: str, payload: Any = None) -> Any:
        """One request, one result."""
        self.requests.put((op, payload))
        kind, result = await self._recv()
        if kind == "error":
            raise WorkerError(result)
        return result

    async def stream(self, op: str, payload: Any = None) -> AsyncIterator[Any]:
        """One request, many results, yielded as they arrive."""
        self.requests.put((op, payload))
        while True:
            kind, result = await self._recv()
            if kind == "item":
                yield result
            elif kind == "done":
                return
            else:
                raise WorkerError(result)


    def drain_events(self) -> list[tuple[str, float]]:
        """Everything the stage reported out of band since the last drain."""
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except Exception:
                return out


async def start_all(workers: list[StageWorker]) -> None:
    """Brings every stage up and warms it.

    Warmup happens here, before any clip is fed, because the first inference
    pays for kernel selection and lazy allocation. On CosyVoice2 that was worth
    2.4 seconds on the first chunk, which would otherwise land in trial one.
    """
    for worker in workers:
        await worker.start()
        print(f"  {worker.kind} up on gpu {worker.gpu}", flush=True)
    for worker in workers:
        await worker.call("warmup")
        print(f"  {worker.kind} warm", flush=True)
