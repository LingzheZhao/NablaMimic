#!/usr/bin/env bash
# Pinned analytic-adjoint MuJoCo Warp plus this repository's five patches.
#   bash scripts/setup_mujoco_warp.sh [DEST] [--no-install]
# Existing modified trees are accepted only if their package files exactly match
# the declared revision plus patches. Use a fresh destination for other builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/mujoco_warp"
INSTALL=1
for a in "$@"; do
  case "$a" in
    --no-install) INSTALL=0 ;;
    *) DEST="$a" ;;
  esac
done
PY="${PYTHON:-python}"
URL=https://github.com/etaoxing/mujoco_warp.git
BASE=357a75d60a56d67d476942a1b6e54b3045ee8e87
HEAD=4af093f02a3f48f0009ca7c916674b928136f310
PATCHES=()
for number in 1 2 3 4 5; do
  matches=("$ROOT"/patches/mujoco_warp/000"$number"-*.diff)
  [[ ${#matches[@]} == 1 && -f "${matches[0]}" ]] || {
    echo "Expected exactly one patch numbered 000$number" >&2; exit 1; }
  PATCHES+=("${matches[0]}")
done

if [[ ! -d "$DEST/.git" && ! -f "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone -q "$URL" "$DEST"
fi
git -C "$DEST" cat-file -e "$HEAD^{commit}" 2>/dev/null || git -C "$DEST" fetch -q origin pull/1/head
git -C "$DEST" cat-file -e "$HEAD^{commit}"
git -C "$DEST" merge-base --is-ancestor "$BASE" "$HEAD"
if [[ "$(git -C "$DEST" rev-parse HEAD)" != "$HEAD" ]]; then
  [[ -z "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]] || {
    echo "$DEST has local changes; use a fresh directory" >&2; exit 1; }
  git -C "$DEST" -c advice.detachedHead=false checkout -q --detach "$HEAD"
fi

# Reconstruct the expected package in temporary storage and compare actual bytes,
# including resources. Runtime bytecode is excluded; extra source files are not.
REFERENCE="$(mktemp -d)"
trap 'rm -rf -- "$REFERENCE"' EXIT
mkdir "$REFERENCE/base" "$REFERENCE/patched"
git -C "$DEST" archive "$HEAD" mujoco_warp | tar -x -C "$REFERENCE/base"
cp -a "$REFERENCE/base/mujoco_warp" "$REFERENCE/patched/"
for p in "${PATCHES[@]}"; do
  git -C "$REFERENCE/patched" apply "$p"
done
same_package() {
  diff -qr --exclude='__pycache__' --exclude='*.pyc' -- "$1/mujoco_warp" "$2/mujoco_warp" >/dev/null
}
if ! same_package "$REFERENCE/patched" "$DEST"; then
  same_package "$REFERENCE/base" "$DEST" || {
    echo "$DEST differs from both the pinned base and the complete patch set; use a fresh directory" >&2
    exit 1
  }
  for p in "${PATCHES[@]}"; do
    git -C "$DEST" apply "$p"
    echo "applied $(basename "$p")"
  done
fi
same_package "$REFERENCE/patched" "$DEST" || {
  echo "MuJoCo Warp package differs from the declared revision plus patches" >&2; exit 1; }
echo "MuJoCo Warp package OK: $DEST"
if [[ "$INSTALL" == 1 ]]; then
  "$PY" -m pip install --no-deps -e "$DEST"
fi
