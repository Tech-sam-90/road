#!/bin/bash
# Shared paths + module-loading logic for the Narval runners.
# Source this from setup_env.sh, prefetch_models.sh, and every submit_*.sb —
# do not execute it directly.
#
# Narval sets $HOME, $PROJECT, $SCRATCH automatically per-user; we build our
# paths off those rather than hardcoding numeric project IDs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT

: "${PROJECT:?PROJECT is not set — are you on a Narval login/compute node?}"
: "${SCRATCH:?SCRATCH is not set — are you on a Narval login/compute node?}"

export PROJECT_REPO="${PROJECT_REPO:-$PROJECT/road-barbados-htr}"
export SCRATCH_REPO="${SCRATCH_REPO:-$SCRATCH/road-barbados-htr}"
export MODEL_CACHE="${MODEL_CACHE:-$PROJECT_REPO/model_cache}"

export VENV_SHARED="${VENV_SHARED:-$PROJECT_REPO/venvs/shared}"
export VENV_KRAKEN="${VENV_KRAKEN:-$PROJECT_REPO/venvs/kraken}"
export VENV_VLM="${VENV_VLM:-$PROJECT_REPO/venvs/vlm}"

# Compute nodes have no internet — force cache-only reads for HF + kraken.
export HF_HOME="$MODEL_CACHE/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export XDG_DATA_HOME="$MODEL_CACHE/xdg_data"

if [ "$REPO_ROOT" != "$PROJECT_REPO" ]; then
    echo "[WARN] this checkout lives at $REPO_ROOT, not \$PROJECT_REPO ($PROJECT_REPO)." >&2
    echo "[WARN] /home has a small quota and is not meant for a repo with code + venvs." >&2
    echo "[WARN] clone/rsync the repo into \$PROJECT before running setup_env.sh, e.g.:" >&2
    echo "[WARN]   rsync -a --exclude data/images --exclude .venv-* $REPO_ROOT/ $PROJECT_REPO/" >&2
fi

# Picks the newest available module matching a preferred-order candidate
# list, by grepping `module avail` output. We don't hardcode a version we
# can't verify from here — this runs for real on the login node.
_pick_module() {
    local candidates=("$@")
    local c
    for c in "${candidates[@]}"; do
        if module avail "$c" 2>&1 | grep -qi "$c"; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# Loads StdEnv + python + cuda + cudnn, preferring the newest known-good
# combination and falling back to whatever `module avail` actually reports.
# Exports NARVAL_STDENV / NARVAL_PYTHON / NARVAL_CUDA / NARVAL_CUDNN so
# callers can log exactly what got loaded.
load_narval_modules() {
    module purge

    local stdenv
    stdenv=$(_pick_module StdEnv/2023 StdEnv/2020) || {
        echo "[ERROR] no StdEnv module found via 'module avail StdEnv'. Run it manually and edit _common.sh." >&2
        return 1
    }
    module load "$stdenv"

    local pymod
    pymod=$(_pick_module python/3.11 python/3.12 python/3.10) || {
        echo "[ERROR] no python module found via 'module avail python'. Run it manually and edit _common.sh." >&2
        return 1
    }
    module load "$pymod"

    local cudamod
    cudamod=$(_pick_module cuda/12.2 cuda/12.0 cuda/11.8) || {
        echo "[WARN] none of the preferred cuda modules matched — falling back to the newest 'cuda/*' module avail reports." >&2
        cudamod=$(module avail cuda 2>&1 | grep -oE 'cuda/[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -V | uniq | tail -1)
    }
    if [ -n "$cudamod" ]; then
        module load "$cudamod"
    else
        echo "[ERROR] no cuda module found at all via 'module avail cuda'." >&2
        return 1
    fi

    local cudnnmod
    cudnnmod=$(module avail cudnn 2>&1 | grep -oE 'cudnn/[0-9.]+' | sort -V | uniq | tail -1)
    if [ -n "$cudnnmod" ]; then
        module load "$cudnnmod"
    else
        echo "[WARN] no cudnn module found via 'module avail cudnn' — continuing without an explicit cudnn module (many StdEnv cuda modules bundle it already)." >&2
    fi

    export NARVAL_STDENV="$stdenv" NARVAL_PYTHON="$pymod" NARVAL_CUDA="$cudamod" NARVAL_CUDNN="${cudnnmod:-none}"
    echo "[modules] $NARVAL_STDENV  $NARVAL_PYTHON  $NARVAL_CUDA  cudnn=$NARVAL_CUDNN"
}

# Copies data/*.csv + data/images from /project (persistent) to /scratch
# (fast, local to the compute node's I/O path) once, then skips on repeat
# calls. Kraken/VLM dataloaders should always read from $SCRATCH_REPO/data,
# never re-read 5472 small files over the /project filesystem every step.
stage_data_to_scratch() {
    local marker="$SCRATCH_REPO/data/.staged_ok"
    if [ -f "$marker" ]; then
        echo "[stage] data already present on scratch, skipping"
        return 0
    fi
    echo "[stage] copying data/ to scratch for fast I/O..."
    mkdir -p "$SCRATCH_REPO/data"
    rsync -a "$PROJECT_REPO/data/images/" "$SCRATCH_REPO/data/images/"
    cp "$PROJECT_REPO"/data/*.csv "$SCRATCH_REPO/data/"
    touch "$marker"
}

# Finds the most recent HF Trainer checkpoint-N/ subdirectory under $1, if
# any, for --resume_from_checkpoint. Prints nothing if none exists.
find_latest_hf_checkpoint() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    ls -d "$dir"/checkpoint-*/ 2>/dev/null \
        | sed -E 's#.*checkpoint-([0-9]+)/?#\1 &#' \
        | sort -n \
        | tail -1 \
        | awk '{print $2}' \
        | sed 's:/*$::'
}

# Finds the most recent kraken .mlmodel checkpoint under $1 (ketos writes
# <prefix>_<epoch>.mlmodel each epoch by default).
find_latest_kraken_checkpoint() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    ls -t "$dir"/*.mlmodel 2>/dev/null | head -1
}
