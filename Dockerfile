# One image for the whole benchmark. Every stage runs in one process in one
# container with several GPUs attached, so there is nothing to compose.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# whisper-streaming is distributed as a repo rather than a package. It loads the
# same Whisper weights as the batch backend; only the policy differs.
RUN git clone --depth 1 https://github.com/ufal/whisper_streaming /opt/whisper_streaming

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/root:/opt/whisper_streaming
ENTRYPOINT ["python3", "-m", "bench.runner"]
