#
# Agens SGLang: upstream SGLang at a pinned commit + patches/agens-volundr.patch, installed over the
# official lmsysorg/sglang v0.5.16 (CUDA 12.9) runtime image.
#
#   docker build -f docker/Dockerfile --build-arg CUDA_ARCH=sm89 -t agens-sglang:preview-sm89 .
#   docker build -f docker/Dockerfile --build-arg CUDA_ARCH=sm90 -t agens-sglang:preview-sm90 .
#
# On top of the base image this adds the patched SGLang Python tree, SGLang's three small Rust (PyO3)
# extension modules built from source, and the official FlashInfer 0.6.15.post1 wheels that this
# SGLang commit requires. All other CUDA kernels (sglang-kernel, FlashAttention, Triton, PyTorch) are
# the base image's own build.

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.16-cu129-runtime

# ---------------------------------------------------------------------------------------------------
# sglang-src: upstream SGLang at SGLANG_COMMIT. For offline builds, replace this stage with a local
# checkout: --build-context sglang-src=DIR, where DIR/sglang is a checkout of that commit.
# ---------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS sglang-src
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_COMMIT=4051b19cc2cd7fd8903ceec1d084a69b56c6df4d
RUN set -eux; \
    git init -q /sglang; cd /sglang; \
    git remote add origin "${SGLANG_REPO}"; \
    for i in 1 2 3 4 5; do \
      git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 fetch -q --depth 1 origin "${SGLANG_COMMIT}" \
        && break; \
      sleep 15; \
    done; \
    git checkout -q FETCH_HEAD; \
    rm -rf .git

# ---------------------------------------------------------------------------------------------------
# build: apply the patch and build SGLang's Rust extension modules
# ---------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS build

ARG RUST_VERSION=1.90.0
# Optional mirrors; empty means the upstream default.
ARG PIP_INDEX_URL=
ARG RUSTUP_INIT_URL=https://sh.rustup.rs
ARG RUSTUP_DIST_SERVER=
ARG RUSTUP_UPDATE_ROOT=
ARG CARGO_REGISTRY_MIRROR=

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:${PATH}
WORKDIR /build

RUN set -eux; \
    if [ -n "${RUSTUP_DIST_SERVER}" ]; then export RUSTUP_DIST_SERVER; fi; \
    if [ -n "${RUSTUP_UPDATE_ROOT}" ]; then export RUSTUP_UPDATE_ROOT; fi; \
    curl --proto '=https' --tlsv1.2 -sSf "${RUSTUP_INIT_URL}" | \
      sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}"; \
    mkdir -p "${CARGO_HOME}"; \
    if [ -n "${CARGO_REGISTRY_MIRROR}" ]; then \
      printf '[source.crates-io]\nreplace-with = "mirror"\n\n[source.mirror]\nregistry = "%s"\n' \
        "${CARGO_REGISTRY_MIRROR}" > "${CARGO_HOME}/config.toml"; \
    fi; \
    cargo --version; rustc --version

COPY --from=sglang-src /sglang /build/sglang
# Upstream does not commit a Cargo.lock for its Rust extension crates; patches/Cargo.lock pins the
# dependency versions these images were built and tested with. The patch does not touch rust/, so
# the extensions are built before it is applied (and stay cached when only the patch changes).
COPY patches/Cargo.lock /build/sglang/rust/Cargo.lock

RUN set -eux; \
    if [ -n "${RUSTUP_DIST_SERVER}" ]; then export RUSTUP_DIST_SERVER; fi; \
    if [ -n "${RUSTUP_UPDATE_ROOT}" ]; then export RUSTUP_UPDATE_ROOT; fi; \
    if [ -n "${PIP_INDEX_URL}" ]; then export PIP_INDEX_URL; fi; \
    python3 -m pip install --no-cache-dir "setuptools-rust>=1.10" "setuptools-scm>=8.0"; \
    cargo fetch --manifest-path /build/sglang/rust/Cargo.toml; \
    RUSTFLAGS="--remap-path-prefix=${CARGO_HOME}=/cargo --remap-path-prefix=/build/sglang=/sglang --remap-path-prefix=${RUSTUP_HOME}=/rustup"; \
    for d in "${CARGO_HOME}"/registry/src/*; do \
      RUSTFLAGS="${RUSTFLAGS} --remap-path-prefix=${d}=/cargo/registry/src/crates.io"; \
    done; \
    export RUSTFLAGS; \
    cd /build/sglang/python; \
    python3 setup.py build_rust --inplace --release; \
    rm -rf build ./*.egg-info; \
    find . -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    ls -la sglang/srt/grpc/_core*.so sglang/srt/multimodal/_core*.so sglang/srt/server/_core*.so; \
    ! grep -a -l -e "${CARGO_HOME}/registry" sglang/srt/*/_core*.so

COPY patches/agens-volundr.patch /build/agens-volundr.patch
RUN set -eux; \
    cd /build/sglang; \
    git apply --whitespace=nowarn /build/agens-volundr.patch; \
    find python -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    test -f python/sglang/srt/models/volundr.py

# ---------------------------------------------------------------------------------------------------
# wheels: FlashInfer 0.6.15.post1 (the version this SGLang commit requires; the base image has 0.6.14)
# ---------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS wheels
ARG PIP_INDEX_URL=
ARG FLASHINFER_VERSION=0.6.15.post1
ARG FLASHINFER_CUBIN_INDEX_URL=https://flashinfer.ai/whl
ARG FLASHINFER_JIT_CACHE_INDEX_URL=https://flashinfer.ai/whl/cu129
RUN set -eux; \
    if [ -n "${PIP_INDEX_URL}" ]; then export PIP_INDEX_URL; fi; \
    mkdir -p /wheels; \
    python3 -m pip download --no-cache-dir --no-deps -d /wheels "flashinfer-python==${FLASHINFER_VERSION}"; \
    python3 -m pip download --no-cache-dir --no-deps -d /wheels "flashinfer-cubin==${FLASHINFER_VERSION}" \
      --index-url "${FLASHINFER_CUBIN_INDEX_URL}"; \
    python3 -m pip download --no-cache-dir --no-deps -d /wheels "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
      --index-url "${FLASHINFER_JIT_CACHE_INDEX_URL}"; \
    ls -la /wheels

# ---------------------------------------------------------------------------------------------------
# runtime: the official image with the patched SGLang tree copied over its editable install
# ---------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE}

ARG FLASHINFER_VERSION=0.6.15.post1
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    set -eux; \
    python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache; \
    python3 -m pip install --no-cache-dir --no-deps --no-index /wheels/*.whl; \
    rm -rf /root/.cache/flashinfer /root/.cache/pip; \
    python3 -c "from importlib.metadata import version as v; \
assert v('flashinfer-python') == '${FLASHINFER_VERSION}', v('flashinfer-python'); \
assert v('flashinfer-cubin') == '${FLASHINFER_VERSION}', v('flashinfer-cubin'); \
assert v('flashinfer-jit-cache').startswith('${FLASHINFER_VERSION}+'), v('flashinfer-jit-cache')"

# The base image installs SGLang editable from /sgl-workspace/sglang/python, so the patched tree
# copied over it is what `python3 -m sglang.launch_server` imports.
COPY --from=build /build/sglang/python/ /sgl-workspace/sglang/python/

RUN python3 -c "import sglang.srt.configs; \
import sglang.srt.models.volundr as v; \
from sglang.srt.models.dflash import DFlash2DraftModel; \
from sglang.srt.function_call.function_call_parser import FunctionCallParser as F; \
from sglang.srt.parser.reasoning_parser import ReasoningParser as R; \
assert 'agens' in F.ToolCallParserEnum and 'agens' in R.DetectorMap; \
print('Volundr classes:', [c.__name__ for c in v.EntryClass])"

# Declared after the layers above so the sm89 and sm90 images share them byte for byte.
ARG CUDA_ARCH=sm89
ARG SGLANG_COMMIT=4051b19cc2cd7fd8903ceec1d084a69b56c6df4d
ARG AGENS_SGLANG_VERSION=preview

ENV AGENS_SGLANG_VERSION=${AGENS_SGLANG_VERSION} \
    AGENS_SGLANG_CUDA_ARCH=${CUDA_ARCH} \
    AGENS_SGLANG_UPSTREAM_COMMIT=${SGLANG_COMMIT} \
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

LABEL org.opencontainers.image.title="agens-sglang" \
      org.opencontainers.image.description="SGLang with Agens Volundr support (model class, agens parsers, reasoning budget, DFlash2 drafter)" \
      org.opencontainers.image.source="https://github.com/BlockWayz/agens-sglang" \
      org.opencontainers.image.vendor="Blockway" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${AGENS_SGLANG_VERSION}-${CUDA_ARCH}" \
      ai.blockway.sglang.upstream-commit="${SGLANG_COMMIT}" \
      ai.blockway.sglang.cuda-arch="${CUDA_ARCH}"

