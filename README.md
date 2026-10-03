# Agens SGLang

SGLang with support for **Agens Volundr 32B**, Blockway's open-weights model
([`Blockway/Agens-Volundr-32B-Preview`](https://huggingface.co/Blockway/Agens-Volundr-32B-Preview)).

This repository is a small patch on top of upstream [SGLang](https://github.com/sgl-project/sglang)
plus the Dockerfile that builds the serving images. Prebuilt images:

| image | GPUs |
|---|---|
| `ghcr.io/blockwayz/agens-sglang:preview-sm89` | 48 GB Ada GPUs (sm89) |
| `ghcr.io/blockwayz/agens-sglang:preview-sm90` | H100 / H200 (sm90) |

## What it adds

* **Volundr model class** (`VolundrForConditionalGeneration`, `VolundrForCausalLM`) for SGLang's
  hybrid-attention runtime:
  * **KDA** (Kimi Delta Attention, linear attention with a per-channel forget gate) on SGLang's KDA
    backend — a fixed-size recurrent state per request instead of a growing KV cache;
  * **BCSA** (Blockway Compressed-Sparse Attention): a 4,096-token sliding window plus a 4:1
    mean-pooled far field selected by a learned indexer, merged under one softmax. The window runs
    through FlashAttention on sm90 and through FlashInfer on sm89; decode runs in CUDA graphs on both;
  * **Engram** (hashed n-gram conditional memory) and **mHC** (manifold hyper-connections, four
    residual streams);
  * tensor parallelism for bf16, compressed-tensors INT4 at tensor-parallel size 1, and the vision
    tower (`--enable-multimodal`).
* **`agens` reasoning and tool-call parsers** (`--reasoning-parser agens --tool-call-parser agens`):
  thinking comes back as `reasoning_content`, tool calls as OpenAI-style `tool_calls`.
* **Per-request reasoning budget**: `"reasoning_budget": N` in a chat request closes the thinking block
  after N generated tokens, so the model always gets to its answer.
* **Speculative decoding** for Volundr (chain verify across KDA, BCSA and Engram state) and the
  **DFlash2** drafter (`DFlash2DraftModel`, `--speculative-algorithm DFLASH`).
* The published checkpoint is served **as shipped**: its `config.json` carries an `auto_map` for the
  transformers reference implementation, but SGLang uses its built-in Volundr classes and does not
  execute the checkpoint's Python code (`--trust-remote-code` is accepted and makes no difference).

## Quick start

All commands mount the Hugging Face cache so the weights are downloaded once. The required flags are
explained [below](#required-flags); the server checks the Volundr-specific ones at startup.

### bf16 on two 48 GB GPUs (sm89, tensor parallel 2)

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host --network host --shm-size 32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/blockwayz/agens-sglang:preview-sm89 \
  python3 -m sglang.launch_server \
    --model-path Blockway/Agens-Volundr-32B-Preview \
    --tp-size 2 --attention-backend flashinfer --page-size 1 \
    --disable-radix-cache --disable-prefill-cuda-graph --disable-custom-all-reduce \
    --mem-fraction-static 0.87 --context-length 32768 \
    --max-mamba-cache-size 16 --max-running-requests 16 --cuda-graph-max-bs 16 \
    --reasoning-parser agens --tool-call-parser agens \
    --host 127.0.0.1 --port 30000
```

Add `--enable-multimodal --mm-feature-transport cpu` to accept images (`cpu` transport is needed on
GPUs without peer-to-peer access, such as PCIe consumer cards).

### INT4 on one 48 GB GPU (sm89)

```bash
docker run --rm --gpus '"device=0"' --ipc=host --network host --shm-size 32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/blockwayz/agens-sglang:preview-sm89 \
  python3 -m sglang.launch_server \
    --model-path Blockway/Agens-Volundr-32B-Preview-INT4 \
    --tp-size 1 --attention-backend flashinfer --page-size 1 \
    --disable-radix-cache --disable-prefill-cuda-graph \
    --mem-fraction-static 0.86 --context-length 16384 \
    --max-mamba-cache-size 8 --max-running-requests 8 --cuda-graph-max-bs 8 \
    --reasoning-parser agens --tool-call-parser agens \
    --host 127.0.0.1 --port 30000
```

[`Blockway/Agens-Volundr-32B-Preview-INT4`](https://huggingface.co/Blockway/Agens-Volundr-32B-Preview-INT4)
is compressed-tensors W4A16 (group size 128; the KDA projections and the vision tower stay bf16).
Add `--enable-multimodal --mm-feature-transport cpu` for images. Tensor parallelism is bf16-only; serve
INT4 at `--tp-size 1`.

### bf16 on one H200 (sm90)

```bash
docker run --rm --gpus '"device=0"' --ipc=host --network host --shm-size 32g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/blockwayz/agens-sglang:preview-sm90 \
  python3 -m sglang.launch_server \
    --model-path Blockway/Agens-Volundr-32B-Preview \
    --tp-size 1 --attention-backend fa3 --page-size 1 \
    --disable-radix-cache --disable-prefill-cuda-graph \
    --mem-fraction-static 0.66 --context-length 16384 \
    --reasoning-parser agens --tool-call-parser agens \
    --host 127.0.0.1 --port 30000
```

The model supports up to 262,144 tokens of context; a longer `--context-length` leaves room for fewer
concurrent requests. Each BCSA layer keeps a small per-token indexer-key cache that is allocated after
SGLang sizes the KV pool, which is why `--mem-fraction-static` is lower than SGLang's default.

### Speculative decoding with the DFlash2 drafter (single user)

The DFlash2 drafter speeds up a single user's stream; it is not a throughput option. With 8 concurrent
distinct requests a drafter server is slower than plain decoding, partly because it runs only 4 request
slots. Two 48 GB GPUs, text requests only (no `--enable-multimodal`):

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host --network host --shm-size 32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/blockwayz/agens-sglang:preview-sm89 \
  python3 -m sglang.launch_server \
    --model-path Blockway/Agens-Volundr-32B-Preview \
    --tp-size 2 --attention-backend flashinfer --page-size 1 \
    --disable-radix-cache --disable-prefill-cuda-graph --disable-custom-all-reduce \
    --mem-fraction-static 0.90 --context-length 16384 --chunked-prefill-size 2048 \
    --max-mamba-cache-size 4 --max-running-requests 4 --cuda-graph-max-bs 4 \
    --speculative-algorithm DFLASH --speculative-draft-model-path Blockway/Agens-Volundr-32B-Preview-DFlash2 \
    --speculative-num-draft-tokens 8 --speculative-draft-model-quantization unquant \
    --reasoning-parser agens --tool-call-parser agens \
    --host 127.0.0.1 --port 30000
```

The drafter is [`Blockway/Agens-Volundr-32B-Preview-DFlash2`](https://huggingface.co/Blockway/Agens-Volundr-32B-Preview-DFlash2)
(3.8 GB; it shares the target's embeddings and output head). The drafter and its speculative state take memory from the KV cache: this configuration leaves about
18K tokens of KV cache on two 48 GB cards, and the smaller prefill chunks plus
`expandable_segments` keep long-prompt prefill inside the remaining headroom. Speculative decoding needs
`--attention-backend flashinfer` on every GPU (the BCSA verify step is implemented there) and is served
as a chain (top-k 1). The gain is largest on predictable output (code, JSON).

## Performance

Measured with this image on 48 GB Ada GPUs (single stream: 256 new tokens after a prompt of the given
length; batch: N concurrent 1K-token prompts, aggregate output rate including their prefill).

| | prompt 1K | 8K | 32K | 64K | 128K | batch 8 | batch 16 |
|---|---|---|---|---|---|---|---|
| bf16, 2 GPUs — decode tok/s | 25.1 | 24.1 | 24.1 | 24.0 | 23.9 | 127 | 130 |
| bf16, 2 GPUs — prefill tok/s | 2,122 | 2,180 | 1,916 | 1,679 | 1,297 | | |
| INT4, 1 GPU — decode tok/s | 31.0 | 29.3 | 29.1 | | | 117 | |
| INT4, 1 GPU — prefill tok/s | 2,236 | 2,200 | 1,764 | | | | |
| bf16 + DFlash2 drafter, 2 GPUs — decode tok/s | code 56, JSON 76, prose 32 | | | | | | |

64K / 128K use a long-context server (`--context-length 140000 --mem-fraction-static 0.90
--chunked-prefill-size 2048`, one request slot). INT4 batch 8 uses `--max-running-requests 9
--max-mamba-cache-size 9` so that 8 requests run at once (with 8 / 8 one slot stays in reserve and the
8th request waits).

## Using the API

The server speaks the OpenAI chat-completions API on `http://127.0.0.1:30000/v1`.

```bash
# thinking is on by default; the reasoning arrives in `reasoning_content`
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Blockway/Agens-Volundr-32B-Preview",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "temperature": 0.6, "top_p": 0.95
}'

# thinking off
  "chat_template_kwargs": {"enable_thinking": false}

# reasoning effort (low / medium / xhigh)
  "chat_template_kwargs": {"reasoning_effort": "low"}

# cap the thinking at 2,000 tokens; the model then writes its answer
  "reasoning_budget": 2000

# tools: standard OpenAI `tools`; calls come back in `message.tool_calls`
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]

# images (server started with --enable-multimodal)
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    {"type": "text", "text": "What is in this picture?"}]}]
```


`examples/smoke_test.py --base-url http://127.0.0.1:30000` runs all of the above against a server
(add `--image-url URL` for the image check and `--speed` for a decode-speed estimate).

Setting the environment variable `VOLUNDR_REASONING_BUDGET=N` on the server applies a default budget
to every chat request that does not set its own.

## Server environment variables

All default on; set to `0` on the server (`docker run -e NAME=0 ...`) to turn one off.

| variable | effect |
|---|---|
| `VOLUNDR_FUSED_MHC` | mHC mixing as two fused Triton kernels per layer instead of ~85 small torch ops |
| `VOLUNDR_BCSA_GRAPH_VARIANTS` | two decode CUDA graphs per batch size: "near" (every request inside the 4,096-token BCSA window, far field skipped) and "far" |
| `VOLUNDR_FUSED_BCSA` | BCSA decode block-key write and far-branch index math / gathers as fused Triton kernels |
| `VOLUNDR_MASK_UNUSED_TOKENS` | token ids at or above 248,077 (beyond the tokenizer's vocabulary) are masked out of sampling |

The first three only change speed: each is bit-exact with the plain torch path it replaces, and greedy
output is token-identical with them on or off. `VOLUNDR_REASONING_BUDGET=N` sets a default reasoning
budget (see above).

## Required flags

| flag | why |
|---|---|
| `--page-size 1` | BCSA reads pooled far-field K/V straight out of the token-indexed KV cache. |
| `--disable-radix-cache` | Engram keeps the previous token ids of each request; a prefix-cache hit would resume a sequence whose ids were never seen. |
| `--disable-prefill-cuda-graph` | the Engram and BCSA prefill paths are not graph-safe. Decode CUDA graphs stay on. |
| `--attention-backend flashinfer` (sm89) / `fa3` (sm90) | FlashAttention's windowed kernel at head dim 256 does not run on sm89. Speculative decoding needs `flashinfer` everywhere. |
| `--max-mamba-cache-size N` (sm89) | the KDA recurrent state is about 170 MB per request (85 MB per GPU at tensor parallel 2); without a cap SGLang's default sizing takes most of a 48 GB card for it. |
| `--disable-custom-all-reduce` (multi-GPU without NVLink) | use NCCL all-reduce on PCIe-connected cards. |
| `--mm-feature-transport cpu` (images, multi-GPU without peer access) | hand image features between processes through host memory instead of CUDA IPC. |
| `--reasoning-parser agens --tool-call-parser agens` | parse the Agens control tokens (`<\|think\|>`, `<\|call\|>`). |

## Building

The images are built from the official SGLang runtime image; only Python sources and SGLang's three
small Rust extension modules are added, and every CUDA kernel is the base image's own build.

```bash
docker build -f docker/Dockerfile --build-arg CUDA_ARCH=sm89 -t agens-sglang:preview-sm89 .
docker build -f docker/Dockerfile --build-arg CUDA_ARCH=sm90 -t agens-sglang:preview-sm90 .
```

Build arguments:

| arg | default | |
|---|---|---|
| `BASE_IMAGE` | `lmsysorg/sglang:v0.5.16-cu129-runtime` | official SGLang image (CUDA 12.9, PyTorch 2.11) |
| `SGLANG_COMMIT` | contents of `SGLANG_COMMIT` | upstream commit the patch applies to |
| `CUDA_ARCH` | `sm89` | recorded in the image labels; the sm89 and sm90 images share every layer (the base image carries kernels for both) |
| `PIP_INDEX_URL`, `RUSTUP_INIT_URL`, `RUSTUP_DIST_SERVER`, `RUSTUP_UPDATE_ROOT`, `CARGO_REGISTRY_MIRROR` | upstream | optional package mirrors |

Without Docker, `scripts/apply_patch.sh DIR` checks out upstream SGLang at the pinned commit and applies
the patch; install it with `pip install -e DIR/python` into an environment matching the base image's
(PyTorch 2.11, CUDA 12.9, `sglang-kernel` 0.4.5) with a Rust toolchain on `PATH`.
`scripts/refresh_patch.sh DIR` writes the patch back after editing.

## Versions

| component | version |
|---|---|
| upstream SGLang | `4051b19cc2cd7fd8903ceec1d084a69b56c6df4d` (main, 2026-07-27, after v0.5.16) |
| base image | `lmsysorg/sglang:v0.5.16-cu129-runtime` |
| CUDA / PyTorch | 12.9 / 2.11 — NVIDIA driver 575 or newer |

## Limitations

* Tensor parallelism is bf16-only; INT4 runs at tensor-parallel size 1.
* No prefix caching (`--disable-radix-cache` is required).
* Speculative decoding: chain only (top-k 1), FlashInfer backend, text requests.
* The first start of a container compiles a few kernels for the local GPU, which adds a few minutes
  (most for INT4).
* Video input is untested (the runtime image ships without FFmpeg).
* 24 GB cards are not supported.

## License

Apache-2.0 — see [LICENSE](LICENSE). This repository modifies SGLang (Apache-2.0); [NOTICE](NOTICE)
lists the added and modified files. The model weights are distributed separately under their own
licence.
