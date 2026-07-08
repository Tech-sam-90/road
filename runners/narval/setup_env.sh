#!/bin/bash
# Run ONCE on a Narval LOGIN node (has internet, can install packages).
#
# Loads the best available module toolchain, creates three virtualenvs under
# $PROJECT (never $HOME — small quota, not for this), and installs each
# stack's requirements.txt. For every package we try `pip install --no-index`
# against the Alliance wheelhouse first (hardware-optimized builds, no
# network needed later) and only fall back to a normal PyPI install — which
# requires this to run on a login node — when the wheelhouse doesn't have it.
#
# We do NOT hardcode module versions we can't verify from outside Narval:
# this script queries `module avail` itself and picks the newest match from
# a preferred-order list (see _common.sh). Re-run any time to pick up new
# module releases or requirements.txt changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

case "$(hostname -f 2>/dev/null || hostname)" in
    *login*|*narval[0-9]*) ;;
    *) echo "[WARN] hostname doesn't look like a Narval login node — this script needs internet access, which compute nodes don't have." >&2 ;;
esac

load_narval_modules

mkdir -p "$PROJECT_REPO" "$MODEL_CACHE"
if [ "$REPO_ROOT" = "$PROJECT_REPO" ]; then
    :
elif [ ! -d "$PROJECT_REPO/src" ]; then
    echo "[setup] no repo checkout found at $PROJECT_REPO — copying this checkout there now." >&2
    rsync -a --exclude 'data/images/' --exclude '.venv-*/' --exclude '.git/' "$REPO_ROOT/" "$PROJECT_REPO/"
fi

WHEELHOUSE_HITS=()
PYPI_HITS=()
FAILED=()

# $1 = venv dir, $2 = requirements.txt path
create_and_install_venv() {
    local venv_dir="$1"
    local req_file="$2"
    local label
    label="$(basename "$venv_dir")"

    echo ""
    echo "=== [$label] venv: $venv_dir ==="
    if [ ! -f "$req_file" ]; then
        echo "[$label] no requirements file at $req_file, skipping"
        return 0
    fi

    if [ ! -d "$venv_dir" ]; then
        # Alliance docs: use `virtualenv --no-download`, not stdlib `venv` —
        # the module-provided virtualenv preseeds pip to point at the local
        # wheelhouse mirror instead of PyPI.
        mkdir -p "$(dirname "$venv_dir")"
        virtualenv --no-download "$venv_dir"
    fi
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    pip install --no-index --upgrade pip setuptools wheel

    while IFS= read -r line; do
        line="$(echo "$line" | sed 's/#.*//' | xargs)"
        [ -z "$line" ] && continue

        local pkg_name
        pkg_name="$(echo "$line" | sed -E 's/[<>=!~; ].*//')"

        local have_wheel=""
        if command -v avail_wheels >/dev/null 2>&1; then
            avail_wheels "$pkg_name" 2>/dev/null | grep -qi "$pkg_name" && have_wheel=1
        fi

        if [ -n "$have_wheel" ] && pip install --no-index "$line" >/tmp/pip_install_$$.log 2>&1; then
            echo "  [wheelhouse] $line"
            WHEELHOUSE_HITS+=("$label:$line")
        elif pip install "$line" >/tmp/pip_install_$$.log 2>&1; then
            echo "  [pypi]       $line"
            PYPI_HITS+=("$label:$line")
        else
            echo "  [FAILED]     $line (see /tmp/pip_install_$$.log)"
            FAILED+=("$label:$line")
        fi
        rm -f /tmp/pip_install_$$.log
    done < "$req_file"

    deactivate
}

create_and_install_venv "$VENV_SHARED" "$PROJECT_REPO/requirements.txt"
create_and_install_venv "$VENV_KRAKEN" "$PROJECT_REPO/src/kraken/requirements.txt"
create_and_install_venv "$VENV_VLM" "$PROJECT_REPO/src/vlm/requirements.txt"

echo ""
echo "=================== SUMMARY ==================="
echo "Modules loaded: $NARVAL_STDENV  $NARVAL_PYTHON  $NARVAL_CUDA  cudnn=$NARVAL_CUDNN"
echo "From wheelhouse (--no-index, hardware-optimized): ${#WHEELHOUSE_HITS[@]}"
printf '  %s\n' "${WHEELHOUSE_HITS[@]}"
echo "From regular PyPI (needed a login node's internet): ${#PYPI_HITS[@]}"
printf '  %s\n' "${PYPI_HITS[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "FAILED installs (fix before submitting jobs): ${#FAILED[@]}"
    printf '  %s\n' "${FAILED[@]}"
fi
echo "================================================="
echo ""
echo "Venvs ready at:"
echo "  shared: $VENV_SHARED"
echo "  kraken: $VENV_KRAKEN"
echo "  vlm:    $VENV_VLM"
echo ""
echo "Next: run prefetch_models.sh (also on a login node) before submitting any jobs."
