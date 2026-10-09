# Comet pi0.5 VLA, 2x speed: evaluation instructions

Policy server for the BEHAVIOR challenge. It runs Comet's public pi0.5 b1k-pt50 checkpoint unchanged and executes
every second action of each predicted chunk (2x speed). OmniGibson runs outside the container and connects to the
server over a websocket.

## Setup

The image is ghcr.io/kantas12/b1k-serve:pt50-baseline, digest
sha256:f5ad61aca9d13fe9343b6b44f82206bc62d7f7cdc37748529392ce5d2af57eaf. The checkpoint is baked in, so nothing is
downloaded at start-up. Pull it on the machine with the GPU:

    docker pull ghcr.io/kantas12/b1k-serve:pt50-baseline

The host needs Docker with the NVIDIA Container Toolkit and an NVIDIA driver that supports CUDA 12.8 (the image is
built on nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04). The evaluator side is the standard BEHAVIOR-1K v3.9.3 install.

## Authentication

None. The image is public on GHCR and pulls anonymously, and the server needs no tokens, keys or network access
beyond the websocket port.

## Capacity

One GPU with 16 GB of memory is enough for the server. All our results were produced with the server alone on an
NVIDIA RTX A4000 (16 GB) and the simulator on a second GPU. The compressed image is about 23 GB. The server is ready
about a minute after the container starts once the image is local.

One server can serve several simulators at once. Each websocket connection gets its own rollout state and all
connections share one model, and batched requests from the v3.9.3 evaluator (--num-envs N) run as one model call.
We ran 4 environments per server.

## Run Instructions

Start the server (port 8000 by default):

    docker run --rm --gpus all -p 8000:8000 ghcr.io/kantas12/b1k-serve:pt50-baseline

Wait for the line "policy server ready", then run the standard evaluator against it:

    python -m omnigibson.eval.eval --task-name <task> --host <server host> --port 8000 \
      --instance-indices 0 1 2 3 4 5 6 7 8 9 \
      --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --write-video

Leave every environment variable at its default. The defaults are the submitted configuration.

| variable | default | meaning |
|---|---|---|
| PORT | 8000 | websocket port |
| TIME_WARP | 2 | execute every 2nd action of each chunk (2x speed) |
| BATCH_INFER | 1 | one model call per batched evaluator request |
| POLICY_CONFIG | pi05_b1k-base | openpi config of the checkpoint |
