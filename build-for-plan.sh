#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

prefix="${KATAGO_PREFIX:-}"
if [[ -z "${prefix}" && -r "${SCRIPT_DIR}/runtime-prefix.txt" ]]; then
  prefix="$(<"${SCRIPT_DIR}/runtime-prefix.txt")"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/repo" ]]; then
  prefix="${SCRIPT_DIR}"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/runtime" ]]; then
  prefix="${SCRIPT_DIR}/runtime"
fi
if [[ -z "${prefix}" && -d "${SCRIPT_DIR}/.final-migration-env" ]]; then
  prefix="${SCRIPT_DIR}/.final-migration-env"
fi
repo="${prefix:+${prefix}/repo}"
[[ -n "${repo}" && -r "${repo}/python/cuda_tactic_workflow.py" ]] || repo="${SCRIPT_DIR}"
helper="${SCRIPT_DIR}/build_for_plan.py"
autotune="${SCRIPT_DIR}/autotune.py"
if [[ ! -r "${helper}" ]]; then
  helper="${SCRIPT_DIR}/final-migration/autotune/build_for_plan.py"
fi
if [[ ! -r "${autotune}" ]]; then
  autotune="${SCRIPT_DIR}/final-migration/autotune/autotune.py"
fi
[[ -r "${helper}" && -r "${autotune}" ]] || {
  printf '[build-for-plan] ERROR: build-only workflow files are missing\n' >&2
  exit 1
}

python="${KATAGO_PYTHON:-}"
if [[ -z "${python}" && -n "${prefix}" && -x "${prefix}/venv/bin/python" ]]; then
  python="${prefix}/venv/bin/python"
fi
if [[ -z "${python}" ]]; then
  python="$(command -v python3 || command -v python || true)"
fi
[[ -n "${python}" && -x "${python}" ]] || {
  printf '[build-for-plan] ERROR: Python 3 is missing; set KATAGO_PYTHON\n' >&2
  exit 1
}

if [[ -z "${prefix}" ]]; then
  prefix="${SCRIPT_DIR}/.final-migration-env"
fi

declare -a command=(
  "${python}" "${helper}"
  --prefix "${prefix}" --repo "${repo}" --autotune "${autotune}"
  --python "${python}"
)
for root in "${SCRIPT_DIR}/plans" "${repo}/final-migration/plans"; do
  if [[ -d "${root}" ]]; then
    command+=(--plans-root "${root}")
    break
  fi
done
exec "${command[@]}" "$@"
