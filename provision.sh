#!/usr/bin/env bash
#
# Provision script for "MuseTalk — синхронизация губ" (slug: musetalk)
# on cloudcompute.ru.
#
# Turns a fresh Ubuntu + NVIDIA Vast box (jupyter-pytorch template) into a
# working MuseTalk 1.5 inference box. Runs the bundled Gradio app on port 7860.
#
# Env vars injected by Instance::buildProvisioningBootstrap before this runs:
#
#   CC_PROVISION_URL   POST endpoint for stage updates
#                      (https://app.cloudcompute.ru/api/agent/provision)
#   CC_AGENT_TOKEN     bearer token authenticating us to that endpoint
#
# Both are optional — if absent, report_stage is a silent no-op so the script
# works for local manual testing (`bash provision.sh` inside a fresh container).
#
# Stage IDs reported here MUST match config/applications.php's
# provisioning.stages for the `musetalk` slug:
#   install_runtime   clone + Python 3.10 venv + torch + deps + MMLab packages
#   download_models   ~5 GB of model weights (MuseTalk, VAE, Whisper, DWPose,
#                     SyncNet, face-parse-bisent) — idempotent, skips if present
#   start_server      Gradio UI on :7860 with all models loaded (ready gate)
#
# PERSISTENCE NOTE: no volume yet — model weights re-download on every fresh
# box (~5 GB, ~10 min) until the volume feature lands. The HF cache lives at
# /workspace/.cache/huggingface (the large workspace disk, not container root).
#
# IMPORTANT — Python isolation
# ----------------------------
# Vast.ai's jupyter-pytorch template floats its tag and has shipped images as
# new as Python 3.14. MuseTalk requires Python 3.10 (mmcv/mmpose/torch 2.0.1
# have no prebuilt wheels for newer interpreters). We pin via `uv` (downloads a
# standalone CPython), same pattern as the `tts` provision script.

set -euo pipefail

CC_PROVISION_URL="${CC_PROVISION_URL:-}"
CC_AGENT_TOKEN="${CC_AGENT_TOKEN:-}"

# --- pinned upstream --------------------------------------------------------

# Official MuseTalk upstream. Pin to a commit SHA after first successful
# deployment — using `main` for initial rollout.
MUSETALK_UPSTREAM="https://github.com/TMElyralab/MuseTalk.git"
MUSETALK_SHA="main"

MUSETALK_PORT="${MUSETALK_PORT:-7860}"
MUSETALK_DIR="${MUSETALK_DIR:-/workspace/MuseTalk}"
VENV_DIR="${MUSETALK_DIR}/.venv"
PYVERSION="${CC_MUSETALK_PYTHON_VERSION:-3.10}"

# torch 2.0.1 + cu118 is the version MuseTalk officially supports.
# Vast boxes run driver-forward-compat so cu118 wheels work on CUDA 12.x hosts.
TORCH_INDEX="https://download.pytorch.org/whl/cu118"

# Populated by setup_python; every install/run goes through these.
PY=""
PIP=""

# HF cache on the large workspace disk to avoid filling the container root.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Enables hf_transfer fast-path in huggingface-cli downloads.
export HF_HUB_ENABLE_HF_TRANSFER=1

# Tracks current stage for the ERR trap.
CURRENT_STAGE="install_runtime"

# --- helpers ----------------------------------------------------------------

# report_stage <json-payload>
#
# Best-effort POST to /api/agent/provision. Failures are swallowed: a missed
# update is far preferable to crashing provisioning.
report_stage() {
    if [ -z "$CC_PROVISION_URL" ] || [ -z "$CC_AGENT_TOKEN" ]; then
        return 0
    fi
    curl -fsS \
        -X POST "$CC_PROVISION_URL" \
        -H "Authorization: Bearer $CC_AGENT_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$1" \
        --max-time 5 \
        >/dev/null 2>&1 || true
}

log() {
    echo "[cc-provision] $*"
}

# report_log <short-status-line>
#
# Best-effort POST of a single live status line shown under the active
# stage's progress bar in the customer dashboard. Keeps the user informed
# without adding new infrastructure — reuses the same provision endpoint
# and WebSocket broadcast as report_stage. 200 chars max (enforced by the
# backend); safe to call frequently since failures are swallowed.
report_log() {
    local line="$1"
    local safe
    safe="$(printf '%s' "$line" | sed 's/\\/\\\\/g; s/"/'"'"'/g')"
    report_stage "{\"stage\":\"${CURRENT_STAGE}\",\"log_line\":\"${safe}\"}"
}

# send_log_tail
#
# Best-effort POST of the last 200 lines of /var/log/cc-provision.log as
# the `log_tail` field on provision_state. Persists across subsequent stage
# updates (sticky-merged on the backend) so the Provision Log tab in the
# dashboard always shows the most recent snapshot. Uses system python3 for
# correct JSON encoding — safe to call before $PY / $PIP are set up.
send_log_tail() {
    if [ -z "$CC_PROVISION_URL" ] || [ -z "$CC_AGENT_TOKEN" ]; then return 0; fi
    local encoded
    encoded="$(tail -n 200 /var/log/cc-provision.log 2>/dev/null \
        | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' \
        2>/dev/null)" || return 0
    [ -z "$encoded" ] && return 0
    report_stage "{\"stage\":\"${CURRENT_STAGE}\",\"log_tail\":${encoded}}"
}

# send_app_log_tail
#
# Best-effort POST of the last 200 lines of the application runtime log. This
# is where app.py tracebacks and Gradio/model-load output land, so it is more
# useful than cc-provision.log once the start_server stage begins.
send_app_log_tail() {
    if [ -z "$CC_PROVISION_URL" ] || [ -z "$CC_AGENT_TOKEN" ]; then return 0; fi
    local encoded
    encoded="$(tail -n 200 /var/log/cc-musetalk.log 2>/dev/null \
        | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' \
        2>/dev/null)" || return 0
    [ -z "$encoded" ] && return 0
    report_stage "{\"stage\":\"${CURRENT_STAGE}\",\"app_log_tail\":${encoded}}"
}

# fail <human-message>
#
# Report a fatal error against the current stage and exit. A non-empty
# `message` on provision_state is the backend's signal to flip the instance
# to ERROR immediately so the user sees a real error instead of the stepper
# hanging until timeout (Instance::hasApplicationProvisioningFailed).
fail() {
    local msg="$1"
    log "FAILED at stage=${CURRENT_STAGE}: ${msg}"
    local safe
    safe="$(printf '%s' "$msg" | sed 's/\\/\\\\/g; s/"/'"'"'/g')"
    local log_tail_enc
    log_tail_enc="$(tail -n 100 /var/log/cc-provision.log 2>/dev/null \
        | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' \
        2>/dev/null || echo 'null')"
    local app_log_tail_enc
    app_log_tail_enc="$(tail -n 100 /var/log/cc-musetalk.log 2>/dev/null \
        | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' \
        2>/dev/null || echo 'null')"
    report_stage "{\"stage\":\"${CURRENT_STAGE}\",\"message\":\"${safe}\",\"log_tail\":${log_tail_enc},\"app_log_tail\":${app_log_tail_enc}}"
    exit 1
}

trap 'fail "Установка прервана на этапе ${CURRENT_STAGE}. Подробности в /var/log/cc-provision.log на инстансе."' ERR

# setup_python
#
# Install uv and create a seeded Python 3.10 venv at $VENV_DIR. uv downloads a
# standalone CPython, decoupling us from whatever interpreter the base image has.
setup_python() {
    if ! command -v uv >/dev/null 2>&1; then
        log "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
            || fail "Не удалось установить uv (менеджер Python). Проверьте сеть."
        export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    fi
    command -v uv >/dev/null 2>&1 || fail "uv не найден в PATH после установки."

    log "creating seeded Python ${PYVERSION} venv at ${VENV_DIR}"
    uv venv --seed --python "${PYVERSION}" "${VENV_DIR}" \
        || fail "Не удалось создать виртуальное окружение Python ${PYVERSION}."

    PY="${VENV_DIR}/bin/python"
    PIP="${VENV_DIR}/bin/pip"
    [ -x "$PY" ] || fail "Python из venv не найден: ${PY}"
    [ -x "$PIP" ] || fail "pip из venv не найден: ${PIP}"
}

# clone_musetalk
#
# Fetch the upstream MuseTalk repo. Idempotent: skips if already cloned.
# Retries 3x on network failures before giving up.
clone_musetalk() {
    if [ -d "${MUSETALK_DIR}/.git" ]; then
        log "MuseTalk already cloned — skipping"
        return 0
    fi
    local clog="/var/log/cc-musetalk-clone.log"
    local n
    for n in 1 2 3; do
        rm -rf "$MUSETALK_DIR"
        if git clone "$MUSETALK_UPSTREAM" "$MUSETALK_DIR" >"$clog" 2>&1; then
            if [ "$MUSETALK_SHA" != "main" ]; then
                git -C "$MUSETALK_DIR" checkout --quiet "$MUSETALK_SHA" >>"$clog" 2>&1
            fi
            return 0
        fi
        log "git clone attempt ${n} failed; retrying in 5s"
        sleep 5
    done
    local tail_msg
    tail_msg="$(tail -c 500 "$clog" 2>/dev/null | tr -d '\r' | tr '\n' ' ' | sed 's/"/'"'"'/g')" || true
    fail "Не удалось склонировать MuseTalk: ${tail_msg}"
}

mkdir -p "$HF_HOME"

# --- stage 1: install_runtime -----------------------------------------------

CURRENT_STAGE="install_runtime"
log "stage: install_runtime"
report_stage '{"stage":"install_runtime","progress_pct":0}'

# System deps: ffmpeg for video I/O, libgl1/libglib for OpenCV headless,
# build-essential for any C extension builds (e.g. mmcv), git and curl.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y --no-install-recommends \
    ffmpeg build-essential git curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    >/dev/null 2>&1 || true

report_stage '{"stage":"install_runtime","progress_pct":5}'
report_log "system deps installed (ffmpeg, build-essential, libgl1)"

clone_musetalk

report_stage '{"stage":"install_runtime","progress_pct":15}'
report_log "MuseTalk cloned"

setup_python

report_stage '{"stage":"install_runtime","progress_pct":20}'
report_log "Python 3.10 venv ready"

# Torch 2.0.1 + cu118 — MuseTalk's officially supported stack. cu118 wheels
# run on any CUDA 11.8+ driver (Vast forward-compat covers 12.x boxes).
# Install before requirements.txt so pip doesn't replace these wheels with
# CPU-only torch pulled from PyPI.
log "installing torch 2.0.1+cu118"
report_log "installing torch 2.0.1+cu118..."
"$PIP" install --no-warn-script-location \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url "$TORCH_INDEX" \
    || fail "Не удалось установить PyTorch 2.0.1+cu118. Попробуйте другой инстанс."

report_stage '{"stage":"install_runtime","progress_pct":40}'
report_log "torch 2.0.1+cu118 installed"

# MuseTalk's own dependencies (diffusers, transformers, gradio, librosa,
# moviepy, etc.). Run from the repo dir so relative paths in the file resolve.
log "installing MuseTalk requirements"
report_log "installing MuseTalk requirements (diffusers, gradio, librosa…)"
"$PIP" install --no-warn-script-location -r "${MUSETALK_DIR}/requirements.txt" \
    || fail "Не удалось установить зависимости MuseTalk."

# gdown is imported by app.py at runtime; pin to a version that works without
# Google auth for public files. May already be in requirements.txt — harmless
# if so.
"$PIP" install --no-warn-script-location "gdown>=5.1.0" >/dev/null 2>&1 || true

# tensorflow 2.12 and onnx/other deps fight over protobuf; force a version
# that satisfies tensorflow's >=3.9.2,<5 constraint AND newer packages.
"$PIP" install --no-warn-script-location "protobuf>=4.25.0,<5.0.0" \
    >/dev/null 2>&1 || true

report_stage '{"stage":"install_runtime","progress_pct":55}'
report_log "MuseTalk requirements installed"
send_log_tail

# openmim + MMLab packages for face detection and pose estimation (DWPose).
# These MUST be installed in this exact version combination to be compatible
# with torch 2.0.1.
#
# `mmcv-lite` installs cleanly from PyPI, but it does NOT include mmcv._ext.
# DWPose imports mmdet, which imports mmcv.ops.nms at runtime, so MuseTalk needs
# the real OpenMMLab CUDA wheel with compiled ops.
log "installing MMLab packages"
report_log "installing openmim, mmengine…"
"$PIP" install --no-warn-script-location --no-cache-dir -U openmim \
    || fail "Не удалось установить openmim."
"$PIP" install --no-warn-script-location mmengine \
    || fail "Не удалось установить mmengine."

report_log "installing mmcv==2.0.1 with CUDA ops (OpenMMLab wheel)…"
"$PIP" uninstall -y mmcv mmcv-lite >/dev/null 2>&1 || true
MMCV_LOG="/tmp/cc-mmcv.log"
"$PIP" install --no-warn-script-location --no-cache-dir --timeout 60 --retries 10 \
    "mmcv==2.0.1" \
    -f "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html" \
    2>&1 | tee "$MMCV_LOG" || {
    tail_msg="$(tail -c 450 "$MMCV_LOG" 2>/dev/null \
        | tr -d '\r' | tr '\n' ' ' | sed 's/"/'"'"'/g')" || true
    fail "Не удалось установить mmcv==2.0.1 с CUDA ops: ${tail_msg}"
}

"$PY" - << 'PYSMOKE' || fail "mmcv установлен, но mmcv.ops не импортируется."
from mmcv.ops import batched_nms
print("mmcv ops import ok")
PYSMOKE

report_stage '{"stage":"install_runtime","progress_pct":60}'
report_log "mmcv CUDA ops ready; installing mmdet==3.1.0…"
send_log_tail
"$PIP" install --no-warn-script-location "mmdet==3.1.0" \
    || fail "Не удалось установить mmdet==3.1.0."

report_stage '{"stage":"install_runtime","progress_pct":68}'
report_log "mmdet done; installing xtcocotools, mmpose==1.1.0…"
send_log_tail

# xtcocotools is a Cython extension required by mmpose. Versions 1.12–1.13 fail
# to compile from source with Cython 3.x (path bug in setup.py); 1.14+ fixes
# this. Pre-install explicitly so pip resolves the >=1.12 constraint from
# mmpose to the working version rather than trying the broken older ones.
"$PIP" install --no-warn-script-location "xtcocotools>=1.14" \
    || fail "Не удалось установить xtcocotools."

# chumpy is an ancient package (last release 2019) whose setup.py does
# `import pip` — a long-deprecated anti-pattern. pip's build isolation
# creates a subprocess that has setuptools/wheel but NOT pip, so the build
# crashes with "No module named 'pip'". --no-build-isolation exposes the
# current venv's pip to the build process, avoiding the error.
# `|| true`: chumpy is only used by mmpose for SMPL body-model fitting;
# DWPose (what MuseTalk calls) never imports it, so a skip is safe.
"$PIP" install --no-warn-script-location --no-build-isolation "chumpy" \
    >/dev/null 2>&1 || true

# mmpose pulls in scipy, matplotlib, pycocotools — can take several minutes.
# Use tee so pip output still reaches cc-provision.log (for log_tail) while
# also being captured for the error message.
MMPOSE_LOG="/tmp/cc-mmpose.log"
"$PIP" install --no-warn-script-location "mmpose==1.1.0" \
    2>&1 | tee "$MMPOSE_LOG" || {
    tail_msg="$(tail -c 450 "$MMPOSE_LOG" 2>/dev/null \
        | tr -d '\r' | tr '\n' ' ' | sed 's/"/'"'"'/g')" || true
    fail "Не удалось установить mmpose==1.1.0: ${tail_msg}"
}

report_stage '{"stage":"install_runtime","progress_pct":80}'
report_log "mmcv, mmdet, mmpose installed"

# Bundled whisper encoder package — present in some MuseTalk checkouts, not
# all. The editable install teaches Python to find the vendored encoder; it's
# harmless to skip if the setup.py isn't there (current v1.5 uses transformers'
# WhisperModel for the same job).
if [ -f "${MUSETALK_DIR}/musetalk/whisper/setup.py" ]; then
    log "installing bundled musetalk/whisper package"
    "$PIP" install --no-warn-script-location \
        --editable "${MUSETALK_DIR}/musetalk/whisper" \
        >/dev/null 2>&1 || true
fi

# Pin huggingface-cli before the download stage. The bundled download_weights.sh
# calls `pip install -U huggingface_hub[cli]` which can pull a broken version
# (known issue, see MuseTalk #xxx); installing here at a pinned version avoids
# the breakage in all subsequent hf CLI calls.
"$PIP" install --no-warn-script-location \
    "huggingface_hub[cli]==0.30.2" hf_transfer \
    >/dev/null 2>&1 || true

report_stage '{"stage":"install_runtime","progress_pct":100}'
send_log_tail

# --- stage 2: download_models -----------------------------------------------

CURRENT_STAGE="download_models"
log "stage: download_models"
send_log_tail
report_stage '{"stage":"download_models","progress_pct":0}'

MODELS_DIR="${MUSETALK_DIR}/models"
HF_CLI="${VENV_DIR}/bin/huggingface-cli"

mkdir -p \
    "${MODELS_DIR}/musetalk" \
    "${MODELS_DIR}/musetalkV15" \
    "${MODELS_DIR}/syncnet" \
    "${MODELS_DIR}/dwpose" \
    "${MODELS_DIR}/face-parse-bisent" \
    "${MODELS_DIR}/sd-vae" \
    "${MODELS_DIR}/whisper"

# MuseTalk V1.0 + V1.5 weights (one HF repo, two subdirs).
# huggingface-cli download is idempotent: compares hashes, skips present files.
if [ ! -f "${MODELS_DIR}/musetalkV15/unet.pth" ]; then
    log "downloading MuseTalk V1.0 + V1.5 weights"
    report_log "downloading MuseTalk V1.0 + V1.5 weights…"
    "$HF_CLI" download TMElyralab/MuseTalk \
        --local-dir "${MODELS_DIR}" \
        --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" \
                  "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
        || fail "Не удалось скачать веса MuseTalk."
fi

report_stage '{"stage":"download_models","progress_pct":20}'
report_log "MuseTalk weights ready"

# SD-VAE (ft-mse). The HF repo is `stabilityai/sd-vae-ft-mse` but the local
# directory MUST be `models/sd-vae` — that is the path app.py and inference.py
# reference via vae_type="sd-vae".
if [ ! -f "${MODELS_DIR}/sd-vae/config.json" ]; then
    log "downloading SD VAE weights"
    report_log "downloading SD VAE (stabilityai/sd-vae-ft-mse)…"
    "$HF_CLI" download stabilityai/sd-vae-ft-mse \
        --local-dir "${MODELS_DIR}/sd-vae" \
        --include "config.json" "diffusion_pytorch_model.bin" \
        || fail "Не удалось скачать веса SD VAE."
fi

report_stage '{"stage":"download_models","progress_pct":35}'
report_log "SD VAE ready"

# Whisper-tiny weights (audio feature extraction via transformers.WhisperModel).
if [ ! -f "${MODELS_DIR}/whisper/pytorch_model.bin" ]; then
    log "downloading Whisper-tiny weights"
    report_log "downloading openai/whisper-tiny…"
    "$HF_CLI" download openai/whisper-tiny \
        --local-dir "${MODELS_DIR}/whisper" \
        --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
        || fail "Не удалось скачать веса Whisper-tiny."
fi

report_stage '{"stage":"download_models","progress_pct":50}'
report_log "Whisper-tiny ready"

# DWPose (whole-body pose estimation used for face landmark detection).
if [ ! -f "${MODELS_DIR}/dwpose/dw-ll_ucoco_384.pth" ]; then
    log "downloading DWPose weights"
    report_log "downloading DWPose (yzd-v/DWPose)…"
    "$HF_CLI" download yzd-v/DWPose \
        --local-dir "${MODELS_DIR}/dwpose" \
        --include "dw-ll_ucoco_384.pth" \
        || fail "Не удалось скачать веса DWPose."
fi

report_stage '{"stage":"download_models","progress_pct":65}'
report_log "DWPose ready"

# SyncNet (LatentSync lip-sync quality metric used during inference).
if [ ! -f "${MODELS_DIR}/syncnet/latentsync_syncnet.pt" ]; then
    log "downloading SyncNet weights"
    report_log "downloading SyncNet (ByteDance/LatentSync)…"
    "$HF_CLI" download ByteDance/LatentSync \
        --local-dir "${MODELS_DIR}/syncnet" \
        --include "latentsync_syncnet.pt" \
        || fail "Не удалось скачать веса SyncNet."
fi

report_stage '{"stage":"download_models","progress_pct":80}'
report_log "SyncNet ready"

# BiSeNet face parser. The official download_weights.sh uses gdown (Google
# Drive) for 79999_iter.pth, which is unreliable in headless provision (quota
# limits, auth redirects). Using the HuggingFace community mirror instead —
# same approach as download_weights.bat. Falls back to the PyTorch model zoo
# for the ResNet-18 backbone if the HF repo doesn't include it.
if [ ! -f "${MODELS_DIR}/face-parse-bisent/79999_iter.pth" ]; then
    log "downloading face-parse-bisent weights"
    report_log "downloading face-parse-bisent (BiSeNet)…"
    "$HF_CLI" download ManyOtherFunctions/face-parse-bisent \
        --local-dir "${MODELS_DIR}/face-parse-bisent" \
        --include "79999_iter.pth" "resnet18-5c106cde.pth" \
        || fail "Не удалось скачать веса face-parse-bisent."
fi

# ResNet-18 backbone — belt-and-suspenders fallback if the HF repo above
# doesn't ship it (the canonical source is PyTorch model zoo).
if [ ! -f "${MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    log "downloading ResNet-18 backbone (fallback)"
    curl -fL "https://download.pytorch.org/models/resnet18-5c106cde.pth" \
        -o "${MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth" \
        || fail "Не удалось скачать веса ResNet-18."
fi

report_stage '{"stage":"download_models","progress_pct":100}'
report_log "all model weights ready"
send_log_tail

# --- stage 3: start_server --------------------------------------------------

CURRENT_STAGE="start_server"
log "stage: start_server"
send_log_tail
report_stage '{"stage":"start_server"}'
report_log "launching Gradio app, loading models into GPU…"

# app.py checks `ffmpeg -version` at inference time; ensure it's in PATH.
# System ffmpeg (installed above via apt-get) is sufficient on Linux.
export FFMPEG_PATH="${FFMPEG_PATH:-$(command -v ffmpeg 2>/dev/null || echo '')}"

# Launch the Gradio web app.
#   --ip 0.0.0.0   required for Vast.ai port mapping (default 127.0.0.1 is
#                  unreachable from outside the container).
#   --use_float16  halves VRAM usage (~2-3 GB total in fp16 vs ~5 GB in fp32)
#                  with minimal perceptual quality difference at 256×256.
# app.py loads all model weights synchronously before binding the port, so the
# port-check loop below is the real ready gate (model load takes ~60-120 s).
(
    cd "${MUSETALK_DIR}"
    nohup "$PY" app.py \
        --ip 0.0.0.0 \
        --port "${MUSETALK_PORT}" \
        --use_float16 \
        > /var/log/cc-musetalk.log 2>&1 &
    echo $! > "${MUSETALK_DIR}/.app.pid"
)
APP_PID="$(cat "${MUSETALK_DIR}/.app.pid" 2>/dev/null || echo '')"

# Wait for the Gradio server to answer HTTP before reporting success.
# 300s timeout: model loading from disk + Gradio startup is ~90s on fast NVMe,
# up to ~3 min on slow container storage.
BIND_TIMEOUT_S=300
_elapsed=0
_next_report=15
for _ in $(seq 1 "$BIND_TIMEOUT_S"); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${MUSETALK_PORT}/" >/dev/null 2>&1; then
        send_log_tail
        send_app_log_tail
        report_stage '{"stage":"start_server","progress_pct":100}'
        log "provisioning complete — MuseTalk Gradio UI on port ${MUSETALK_PORT}"
        exit 0
    fi
    if [ -n "$APP_PID" ] && ! kill -0 "$APP_PID" 2>/dev/null; then
        log "app.py exited before binding port ${MUSETALK_PORT}"
        tail_msg="$(tail -c 500 /var/log/cc-musetalk.log 2>/dev/null | tr -d '\r' | tr '\n' ' ' | sed 's/"/'"'"'/g')" || true
        fail "MuseTalk приложение упало при запуске: ${tail_msg}"
    fi
    _elapsed=$((_elapsed + 1))
    if [ "$_elapsed" -ge "$_next_report" ]; then
        # Surface the last non-empty line from the Gradio app log so the user
        # can see what the model loader is doing without SSHing in.
        _app_line="$(grep -v '^\s*$' /var/log/cc-musetalk.log 2>/dev/null \
            | tail -1 | sed 's/\\/\\\\/g; s/"/'"'"'/g' | cut -c1-140)" || true
        if [ -n "$_app_line" ]; then
            report_log "loading models (${_elapsed}s): ${_app_line}"
        else
            report_log "loading models into GPU… (${_elapsed}s)"
        fi
        send_app_log_tail
        _next_report=$((_next_report + 15))
    fi
    sleep 1
done

fail "MuseTalk Gradio UI не запустился за ${BIND_TIMEOUT_S}с. Смотрите /var/log/cc-musetalk.log на инстансе."
