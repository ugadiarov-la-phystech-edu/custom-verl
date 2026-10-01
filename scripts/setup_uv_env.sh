#!/bin/bash
# Create the uv environment for this repo (verl release/v0.9.0 + the fork's baselines) on any machine.
#
# Usage: bash scripts/setup_uv_env.sh [ENV_DIR]
#
#   ENV_DIR         where to create the venv (1st argument or env var; default: $HOME/uv-envs/vcpo-env)
#   UV_ACTIVATE     script to source that puts `uv` on PATH (e.g. one exporting UV_CACHE_DIR/UV_PYTHON_INSTALL_DIR)
#   INSTALL_UV=1    install uv into $HOME/.local/bin when it cannot be found (set 0 to fail instead)
#   USE_MEGATRON=1  install the Megatron stack: Megatron-Core/-Bridge, mbridge, TransformerEngine, apex, modelopt
#   USE_DEEPGEMM=1  make vLLM's DeepGEMM (block-FP8 rollout GEMMs, rollout.quantization=fp8) usable: needs the
#                   CUDA toolkit below at runtime, exported as CUDA_HOME by the generated activate script
#   CUDA_ARCHS      GPU archs for the TE/apex builds, e.g. "90" or "80;90" (default: detected with nvidia-smi)
#   CUDA_TOOLKIT    CUDA 12.9 toolkit used for those builds and by DeepGEMM's JIT (default: a user-space copy
#                   next to ENV_DIR, downloaded from NVIDIA's redist archives unless $CUDA_TOOLKIT/bin/nvcc exists)
#   DG_CACHE_DIR    DeepGEMM's compiled-kernel cache (default: cache/deep_gemm next to ENV_DIR; keep it on a
#                   persistent filesystem so only the first run pays the JIT warmup)
#   MAX_JOBS        parallel compile jobs for the TE/apex builds (default: 64)
#   FORCE=1         remove an existing ENV_DIR first
#
# uv's own variables (UV_CACHE_DIR, UV_PYTHON_INSTALL_DIR, UV_LINK_MODE) are respected; put the cache on the same
# filesystem as ENV_DIR when the home directory is small.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# -----------------------------------------------------------------------------------------------------------------
# What gets installed, and why it differs from docker/Dockerfile.stable.vllm (the v0.9.0 reference image):
#   * Same versions: python 3.12, torch 2.11.0, vllm 0.24.0, trl 0.27.0, TransferQueue 0.1.8, Megatron-Core
#     core_v0.18.0, Megatron-Bridge r0.5.0, TransformerEngine 2.16.1, flash-attn 2.8.3, vllm-omni 0.24.0.
#   * CUDA 12.9 builds instead of the image's cu130. cu130 wheels need a CUDA 13 driver (R580+) and fail on
#     e.g. driver 560 / CUDA 12.6 with "NVIDIA driver on your system is too old"; a 12.9 runtime runs on any
#     CUDA 12.x or 13.x driver via CUDA minor-version compatibility.
#   * vllm comes from the official +cu129 wheel on its GitHub release (the PyPI wheel is the cu130 build).
#   * flash-attn: Dao-AILab ships no torch 2.11 + cu12 wheel; a third-party prebuilt one is used
#     (mjun0812/flash-attention-prebuild-wheels, cu12.8 build, runs on the cu12.9 torch).
#   * TransformerEngine: prebuilt cu12 core + torch extension built from source (no official torch 2.11 wheel).
#   * apex is required with TE: Megatron-Bridge then enables gradient_accumulation_fusion, and the LM head
#     (a plain ColumnParallelLinear) needs apex's fused_weight_gradient_mlp_cuda, otherwise model init fails.
#   * nvidia-modelopt 0.44.0 (hard-imported by megatron.bridge; the version paired with Megatron-Bridge r0.5.0).
#   * scipy is kept (the image uninstalls it, but modelopt hard-depends on it).
#   * cupy-cuda12x 14.0.1 (pinned as in docker/Dockerfile.stable.trtllm; Dockerfile.stable.vllm does not list it): the
#     nccl checkpoint engine (fully-async's trainer -> vLLM weight sync) imports it, and verl registers that
#     backend only if the import succeeds. Without cupy, fully-async trainer init fails with "Checkpoint engine
#     nccl not registered"; colocated runs use the naive backend and never notice.
#   * DeepGEMM (bundled in vllm as vllm.third_party.deep_gemm) JIT-compiles its kernels with nvcc and fails to
#     import without a CUDA toolkit (CUDA_HOME); vLLM then silently falls back to slower block-FP8 kernels. The
#     image has /usr/local/cuda; here the same user-space CUDA 12.9 toolkit as for TE/apex is used. Its kernels
#     run on older CUDA 12.x drivers too (verified: driver 560 / CUDA 12.6, H100).
#   * FlashInfer's block-FP8 GEMM (vLLM's choice for batches < 32 tokens when CUDA_HOME is set) additionally
#     needs cuBLAS/cuRAND dev headers the user-space toolkit lacks and crashes vLLM start-up, so the generated
#     activate script sets VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0: DeepGEMM then serves every batch size.
# -----------------------------------------------------------------------------------------------------------------

ENV_DIR=${1:-${ENV_DIR:-$HOME/uv-envs/vcpo-env}}
USE_MEGATRON=${USE_MEGATRON:-1}
USE_DEEPGEMM=${USE_DEEPGEMM:-1}
FORCE=${FORCE:-0}
INSTALL_UV=${INSTALL_UV:-1}
MAX_JOBS=${MAX_JOBS:-64}
PYTHON_VERSION=3.12   # fixed: the vllm/flash-attn/TE wheels below are cp312 builds for torch 2.11 + cu12.9
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$(dirname "$ENV_DIR")"
ENV_DIR=$(cd "$(dirname "$ENV_DIR")" && pwd)/$(basename "$ENV_DIR")
CUDA_TOOLKIT=${CUDA_TOOLKIT:-$(dirname "$ENV_DIR")/cuda-12.9}
DG_CACHE_DIR=${DG_CACHE_DIR:-$(dirname "$ENV_DIR")/cache/deep_gemm}

TORCH_IDX=https://download.pytorch.org/whl/cu129
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
FLASH_ATTN_WHEEL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu128torch2.11-cp312-cp312-linux_x86_64.whl"
TE_VERSION=2.16.1
# Git dependencies pinned to the exact commits of the working env (the Dockerfile tracks branches:
# mbridge@main, Megatron-Bridge@r0.5.0, Megatron-LM@core_v0.18.0, apex@master, which move over time).
MBRIDGE_COMMIT=a61943d7fcb34a190471cfeb0a0eb8bbda621ddf           # mbridge 0.15.1
MEGATRON_BRIDGE_COMMIT=f83545d981f32310d96c3614008a99389d7ccc71   # r0.5.0 -> megatron-bridge 0.5.2
MEGATRON_CORE_COMMIT=ba7b5ebce12af60627a80985792a1449ce45f46c     # core_v0.18.0 -> megatron-core 0.18.0
APEX_COMMIT=575968bc1f9127ccd61003a681472a83af4ff1a1              # master as of 2026-09-30
MODELOPT_VERSION=0.44.0
# CUDA 12.9 redist archives needed to compile TE/apex and DeepGEMM's kernels (nvcc must match torch's CUDA 12.9)
CUDA_REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_ARCHIVES=(
    cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz
    cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz
    cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz
    cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz
    # cuda_profiler_api.h: included by apex's Megatron softmax kernels
    cuda_profiler_api/linux-x86_64/cuda_profiler_api-linux-x86_64-12.9.79-archive.tar.xz
)

# A system LD_LIBRARY_PATH would shadow the venv's CUDA 12.9 libs (e.g. an older libcudart).
unset LD_LIBRARY_PATH VIRTUAL_ENV

UV_SOURCED=""
if [ -n "${UV_ACTIVATE:-}" ]; then
    source "$UV_ACTIVATE"
    UV_SOURCED=$UV_ACTIVATE
fi
if ! command -v uv > /dev/null; then
    if [ "$INSTALL_UV" -eq 1 ]; then
        echo "0. uv not found: installing it into $HOME/.local/bin"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
        export PATH="$HOME/.local/bin:$PATH"
        UV_SOURCED=""
    else
        echo "ERROR: uv not found. Put it on PATH, set UV_ACTIVATE=/path/to/script, or rerun with INSTALL_UV=1."
        exit 1
    fi
fi
echo "uv: $(command -v uv) ($(uv --version))"
# Network filesystems often cannot hardlink out of the uv cache; copying gives the same result.
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1
    # "9.0" -> "90"; several distinct GPU types -> "80;90"
    DETECTED_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd ';')
else
    DETECTED_ARCHS=""
fi
CUDA_ARCHS=${CUDA_ARCHS:-$DETECTED_ARCHS}
if [ "$USE_MEGATRON" -eq 1 ] && [ -z "$CUDA_ARCHS" ]; then
    echo "ERROR: no GPU detected; set CUDA_ARCHS (e.g. CUDA_ARCHS=90) to build TE/apex, or USE_MEGATRON=0."
    exit 1
fi

if [ -e "$ENV_DIR" ]; then
    if [ "$FORCE" -eq 1 ]; then
        echo "FORCE=1: removing existing $ENV_DIR"
        rm -rf "$ENV_DIR"
    else
        echo "ERROR: $ENV_DIR already exists (and may be the production env)."
        echo "Pick another path (bash $0 /new/path or ENV_DIR=...) or pass FORCE=1 to recreate it."
        exit 1
    fi
fi

echo "1. Create venv at $ENV_DIR (Python $PYTHON_VERSION) and install torch 2.11.0+cu129"
uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
source "$ENV_DIR/bin/activate"
PY="$ENV_DIR/bin/python"
SP=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
uv pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 --index-url "$TORCH_IDX"

# Constraints for every later step: nothing may move torch or vllm; setuptools must satisfy both vllm (<81)
# and nvidia-modelopt (>=80).
CONSTRAINTS="$ENV_DIR/constraints.txt"
cat > "$CONSTRAINTS" <<EOF
torch==2.11.0+cu129
torchvision==0.26.0+cu129
torchaudio==2.11.0+cu129
vllm @ ${VLLM_WHEEL}
setuptools>=80,<81
EOF
# PyPI first; the cu129 index only supplies the +cu129 torch builds pinned above.
IDX=(--index-url https://pypi.org/simple --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match)

echo "2. Install vLLM 0.24.0 (cu129), verl's requirements and the extras"
# requirements*.txt: verl's own deps (incl. TransferQueue); the rest are setup.py extras (test, math, geo)
# and Dockerfile.stable.vllm additions. datasets>=3.0.0: without it uv backtracks to datasets 1.1.1.
uv pip install "${IDX[@]}" -c "$CONSTRAINTS" \
    "vllm @ ${VLLM_WHEEL}" \
    -r "$REPO_ROOT/requirements.txt" \
    -r "$REPO_ROOT/requirements-test.txt" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "datasets>=3.0.0" "setuptools>=80,<81" \
    pytest py-spy pytest-asyncio pytest-rerunfailures math-verify \
    mathruler qwen-vl-utils==0.0.14 liger-kernel \
    codetiming pylatexenc cachetools nvtx matplotlib ninja nvidia-mathdx pybind11 wheel onnxscript
# freeze the resolved transformers so later steps cannot move it
echo "transformers==$("$PY" -c 'import importlib.metadata as m; print(m.version("transformers"))')" >> "$CONSTRAINTS"

echo "3. Install trl 0.27.0, torchcodec, vllm-omni, cupy and FlashAttention (prebuilt cp312/torch2.11/cu12)"
uv pip install --no-deps trl==0.27.0     # the image installs it without deps too
uv pip install --no-deps torchcodec==0.16.0 --index-url "$TORCH_IDX"
uv pip install "${IDX[@]}" -c "$CONSTRAINTS" nvidia-cudnn-frontend
uv pip install "${IDX[@]}" -c "$CONSTRAINTS" vllm-omni==0.24.0   # pins accelerate==1.12.0 (downgrade, expected)
uv pip install "${IDX[@]}" -c "$CONSTRAINTS" cupy-cuda12x==14.0.1   # nccl checkpoint engine (fully-async weight sync)
uv pip install --no-deps "flash-attn @ ${FLASH_ATTN_WHEEL}"

if [ "$USE_MEGATRON" -eq 1 ] || [ "$USE_DEEPGEMM" -eq 1 ]; then
    echo "4. CUDA 12.9 toolkit (TE/apex builds, DeepGEMM JIT): $CUDA_TOOLKIT"
    if [ ! -x "$CUDA_TOOLKIT/bin/nvcc" ] || [ ! -f "$CUDA_TOOLKIT/include/cuda_profiler_api.h" ]; then
        mkdir -p "$CUDA_TOOLKIT" "$CUDA_TOOLKIT/.archives"
        for p in "${CUDA_ARCHIVES[@]}"; do
            f="$CUDA_TOOLKIT/.archives/$(basename "$p")"
            [ -s "$f" ] || curl -fsSL --retry 3 -o "$f" "$CUDA_REDIST/$p"
            tar -xJf "$f" -C "$CUDA_TOOLKIT" --strip-components=1
        done
        [ -e "$CUDA_TOOLKIT/lib64" ] || ln -s lib "$CUDA_TOOLKIT/lib64"   # CMake / torch look for lib64
    fi
    "$CUDA_TOOLKIT/bin/nvcc" --version | tail -1
fi

if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "5. Install Megatron-Core core_v0.18.0, Megatron-Bridge r0.5.0 and mbridge"
    uv pip install "${IDX[@]}" -c "$CONSTRAINTS" "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}"
    uv pip install --no-deps --no-build-isolation \
        "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge.git@${MEGATRON_BRIDGE_COMMIT}" \
        "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM.git@${MEGATRON_CORE_COMMIT}"

    echo "6. Build TransformerEngine $TE_VERSION and apex for CUDA_ARCHS=$CUDA_ARCHS"
    # Every pip nvidia wheel contributes headers/libs the extensions include. nvidia/cu13 (a CUDA-13 nvcc wheel
    # that flashinfer pulls in through cuda-tile) is excluded on purpose.
    NV_INC=""; NV_LIB=""
    for d in "$SP"/nvidia/*/include; do [[ -d "$d" && "$d" != *"/nvidia/cu13/"* ]] && NV_INC="$NV_INC:$d"; done
    for d in "$SP"/nvidia/*/lib;     do [[ -d "$d" && "$d" != *"/nvidia/cu13/"* ]] && NV_LIB="$NV_LIB:$d"; done
    # pip wheels ship only versioned .so files; the linker needs the unversioned names
    ln -sf libcudart.so.12 "$SP/nvidia/cuda_runtime/lib/libcudart.so"
    ln -sf libcudnn.so.9   "$SP/nvidia/cudnn/lib/libcudnn.so"
    ln -sf libnvrtc.so.12  "$SP/nvidia/cuda_nvrtc/lib/libnvrtc.so"
    ln -sf libnccl.so.2    "$SP/nvidia/nccl/lib/libnccl.so"
    for n in cublas cublasLt cusparse cusolver curand cufft nvJitLink; do
        f=$(ls "$SP"/nvidia/*/lib/lib${n}.so.* 2>/dev/null | grep -v "/nvidia/cu13/" | head -1) || true
        [ -n "${f:-}" ] && ln -sf "$(basename "$f")" "$(dirname "$f")/lib${n}.so"
    done
    (
        # subshell: the build-only variables must not leak into the sanity check below
        export CUDA_HOME=$CUDA_TOOLKIT CUDA_PATH=$CUDA_TOOLKIT
        export CUDNN_PATH=$SP/nvidia/cudnn CUDNN_HOME=$SP/nvidia/cudnn
        export PATH=$CUDA_TOOLKIT/bin:$PATH
        export CPATH="$CUDA_TOOLKIT/include$NV_INC"
        export LIBRARY_PATH="$CUDA_TOOLKIT/lib64$NV_LIB"
        export LD_LIBRARY_PATH="${NV_LIB#:}"
        export NVTE_CUDA_ARCHS=$CUDA_ARCHS NVTE_FRAMEWORK=pytorch MAX_JOBS=$MAX_JOBS
        # "80;90" -> "8.0;9.0" for torch's extension builder (apex)
        export TORCH_CUDA_ARCH_LIST
        TORCH_CUDA_ARCH_LIST=$(echo "$CUDA_ARCHS" | tr ';' '\n' | sed -E 's/^([0-9]+)([0-9])$/\1.\2/' | paste -sd ';')
        cd /tmp
        # Prebuilt cu12 core + metapackage, then the torch extension from source. All --no-deps: PyPI's
        # transformer_engine_torch metadata requires transformer_engine_cu13, whose files would overwrite
        # the cu12 core's (same paths).
        uv pip install --no-deps "transformer-engine-cu12==$TE_VERSION" "transformer-engine==$TE_VERSION"
        uv pip install --no-deps --no-build-isolation "transformer-engine-torch==$TE_VERSION"
        # apex with its C++/CUDA extensions, as the image builds it (APEX_CPP_EXT / APEX_CUDA_EXT are the
        # env-var forms of --cpp_ext / --cuda_ext; uv has no --build-option)
        APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=8 NVCC_APPEND_FLAGS="--threads 4" \
            uv pip install --no-deps --no-build-isolation "apex @ git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}"
    )
    # transformer-engine-torch's remaining runtime deps
    uv pip install "${IDX[@]}" -c "$CONSTRAINTS" einops onnxscript onnx pydantic nvdlfw-inspect

    echo "7. Install nvidia-modelopt $MODELOPT_VERSION"
    uv pip install "${IDX[@]}" -c "$CONSTRAINTS" "nvidia-modelopt==$MODELOPT_VERSION"

    # TE's loader treats "nvrtc and curand not found on the system" as "no CUDA toolkit" and then loads
    # cudart from nvidia/cu13 first -> libcudart.so.13 next to torch's libcudart.so.12 in one process.
    # Pointing NVRTC_HOME / CURAND_HOME at the venv's CUDA 12 libs skips that branch. Written into the
    # generated activate script (Ray workers inherit it from the launching shell) and used by the check below.
    export NVRTC_HOME="$SP/nvidia/cuda_nvrtc" CURAND_HOME="$SP/nvidia/curand"
fi

echo "8. Sanity check"
DG_ENV=()
if [ "$USE_DEEPGEMM" -eq 1 ]; then
    mkdir -p "$DG_CACHE_DIR"
    DG_ENV=(CUDA_HOME="$CUDA_TOOLKIT" DG_JIT_CACHE_DIR="$DG_CACHE_DIR")
fi
env "${DG_ENV[@]}" USE_MEGATRON=$USE_MEGATRON USE_DEEPGEMM=$USE_DEEPGEMM PYTHONPATH="$REPO_ROOT" python - <<'EOF'
import importlib, os
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
mods = ["vllm", "vllm_omni", "transformers", "trl", "datasets", "ray", "tensordict", "transfer_queue",
        "flash_attn", "flashinfer", "liger_kernel", "torchcodec", "verl", "cupy",
        "verl.checkpoint_engine.nccl_checkpoint_engine"]   # fully-async's weight sync; needs cupy
if os.environ["USE_MEGATRON"] == "1":
    mods += ["megatron.core", "megatron.bridge", "mbridge", "transformer_engine", "modelopt",
             "fused_weight_gradient_mlp_cuda", "amp_C"]   # the last two: apex CUDA extensions
failed = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:32s} {getattr(mod, '__version__', '')}")
    except Exception as e:
        failed.append(m)
        print(f"  FAIL {m:32s} {type(e).__name__}: {str(e)[:120]}")
if torch.cuda.is_available():
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    from flash_attn import flash_attn_func
    assert torch.isfinite(flash_attn_func(q, q, q, causal=True)).all()
    if os.environ["USE_MEGATRON"] == "1" and "transformer_engine" not in failed:
        import transformer_engine.pytorch as te
        lin = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
        lin(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).sum().backward()
        libs = {l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps") if "libcudart" in l}
        if any("libcudart.so.13" in l for l in libs):
            failed.append("te-cudart")
            print("  FAIL TE loaded the CUDA 13 runtime:", sorted(libs))
    if "cupy" not in failed:
        # the nccl checkpoint engine broadcasts weights through a cupy NCCL communicator
        from cupy.cuda import nccl
        comm = nccl.NcclCommunicator(1, nccl.get_unique_id(), 0)
        assert comm.rank_id() == 0
        print(f"  ok   GPU: cupy NCCL {nccl.get_version()} communicator")
    print("  ok   GPU: flash-attn" + (" + TransformerEngine" if os.environ["USE_MEGATRON"] == "1" else ""))
if os.environ["USE_DEEPGEMM"] == "1":
    # vLLM's block-FP8 linear path: 1x128 activation groups, 128x128 weight blocks, bf16 output
    try:
        import vllm.third_party.deep_gemm  # noqa: F401  (fails with AssertionError when CUDA_HOME is missing)
        from vllm.utils import deep_gemm as vdg
        print(f"  ok   {'deep_gemm':32s} supported={vdg.is_deep_gemm_supported()}")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
            from vllm.third_party.deep_gemm.testing.numeric import calc_diff
            from vllm.third_party.deep_gemm.utils.math import per_block_cast_to_fp8
            e8m0 = vdg.is_deep_gemm_e8m0_used()
            x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16) / 2048**0.5
            q, s = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=e8m0)
            w_q, w_s = per_block_cast_to_fp8(w, use_ue8m0=e8m0)
            out = torch.empty(64, 4096, device="cuda", dtype=torch.bfloat16)
            vdg.fp8_gemm_nt((q, s), (w_q, w_s), out, is_deep_gemm_e8m0_used=e8m0)  # JIT-compiles with nvcc
            diff = float(calc_diff(out.float(), x.float() @ w.float().t()))
            if diff > 1e-2:
                failed.append("deep_gemm-numerics")
                print(f"  FAIL DeepGEMM FP8 GEMM calc_diff={diff:.2e}")
            else:
                print(f"  ok   GPU: DeepGEMM FP8 GEMM (JIT compile + run), calc_diff={diff:.2e}")
    except Exception as e:
        failed.append("deep_gemm")
        print(f"  FAIL {'deep_gemm':32s} {type(e).__name__}: {str(e)[:120]}")
if failed:
    raise SystemExit(f"sanity check failed: {failed}")
EOF

ACTIVATE="$ENV_DIR/activate_vcpo.sh"
{
    echo "# Generated by scripts/setup_uv_env.sh: puts uv on PATH, activates the venv and enters the repo."
    if [ -n "$UV_SOURCED" ]; then echo "source \"$UV_SOURCED\""; else echo "export PATH=\"$(dirname "$(command -v uv)"):\$PATH\""; fi
    for v in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_LINK_MODE; do
        [ -n "${!v:-}" ] && echo "export $v=\"${!v}\""
    done
    echo "source \"$ENV_DIR/bin/activate\""
    echo "# a system LD_LIBRARY_PATH would shadow the venv's CUDA 12.9 libs"
    echo "unset LD_LIBRARY_PATH"
    if [ "$USE_MEGATRON" -eq 1 ]; then
        echo "# keep TransformerEngine on torch's CUDA 12 runtime (see step 7 of scripts/setup_uv_env.sh)"
        echo "export NVRTC_HOME=\"$SP/nvidia/cuda_nvrtc\""
        echo "export CURAND_HOME=\"$SP/nvidia/curand\""
    fi
    if [ "$USE_DEEPGEMM" -eq 1 ]; then
        echo "# DeepGEMM (vLLM block-FP8 rollout GEMMs) JIT-compiles with this toolkit; kernels cached persistently"
        echo "export CUDA_HOME=\"$CUDA_TOOLKIT\""
        echo "export DG_JIT_CACHE_DIR=\"$DG_CACHE_DIR\""
        echo "# FlashInfer's block-FP8 JIT needs cuBLAS/cuRAND headers the toolkit lacks; DeepGEMM serves all batch sizes"
        echo "export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0"
    fi
    echo "cd \"$REPO_ROOT\""
} > "$ACTIVATE"

echo "Done. Activate with: source $ACTIVATE"
echo "      (or just the venv: source $ENV_DIR/bin/activate, then export the variables set in that file)"
echo "NOTE: verl itself is not installed; run training from the repo root (or with PYTHONPATH=$REPO_ROOT)."
