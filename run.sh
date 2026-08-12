#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MODEL_NAME="b11c768h12nbt3tflrs-fson-silu.bin.gz"

usage() {
  cat <<'EOF'
Usage: ./run.sh [OPTIONS] [-- KATAGO_GTP_ARGS...]

Start KataGo GTP. A CUDA tactic plan is an optional optimization preset; it
does not own the model, physical batch size, lane count, or runtime settings.

Options:
  --device N          CUDA device ordinal (default: 0)
  --model PATH        any KataGo model
  --config PATH       GTP config (default: cpp/configs/gtp_example.cfg)
  --plan PATH         tactic preset (default: auto-detect by CUDA architecture)
  --no-plan           run the normal CUDA backend without a tactic preset
  --tactic-batch N    choose a mapping when a plan contains multiple mappings
  --batch N           override nnMaxBatchSize (otherwise use the config)
  --lanes N           override numNNServerThreadsPerModel (otherwise use config)
  --katago PATH       executable to run (no plan-binary hash restriction)
  --search-threads N  override numSearchThreads (otherwise use the config)
  --print-command     print the final command without starting GTP
  -h, --help          show this help

The same values may be supplied through KATAGO_DEVICE, KATAGO_MODEL,
KATAGO_CONFIG, KATAGO_PLAN, KATAGO_BIN, KATAGO_TACTIC_BATCH, KATAGO_BATCH,
KATAGO_LANES, and KATAGO_SEARCH_THREADS.
EOF
}

die() { printf '[katago-run] ERROR: %s\n' "$*" >&2; exit 1; }

device="${KATAGO_DEVICE:-0}"
model="${KATAGO_MODEL:-}"
config="${KATAGO_CONFIG:-}"
plan="${KATAGO_PLAN:-}"
katago="${KATAGO_BIN:-}"
tactic_batch="${KATAGO_TACTIC_BATCH:-}"
batch="${KATAGO_BATCH:-}"
lanes="${KATAGO_LANES:-}"
search_threads="${KATAGO_SEARCH_THREADS:-}"
no_plan=0
print_command=0
declare -a gtp_args=()

while (( $# > 0 )); do
  case "$1" in
    --device) [[ $# -ge 2 ]] || die "--device requires a value"; device="$2"; shift 2 ;;
    --model) [[ $# -ge 2 ]] || die "--model requires a path"; model="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config requires a path"; config="$2"; shift 2 ;;
    --plan) [[ $# -ge 2 ]] || die "--plan requires a path"; plan="$2"; shift 2 ;;
    --no-plan) no_plan=1; shift ;;
    --tactic-batch) [[ $# -ge 2 ]] || die "--tactic-batch requires a value"; tactic_batch="$2"; shift 2 ;;
    --batch) [[ $# -ge 2 ]] || die "--batch requires a value"; batch="$2"; shift 2 ;;
    --lanes) [[ $# -ge 2 ]] || die "--lanes requires a value"; lanes="$2"; shift 2 ;;
    --katago) [[ $# -ge 2 ]] || die "--katago requires a path"; katago="$2"; shift 2 ;;
    --search-threads) [[ $# -ge 2 ]] || die "--search-threads requires a value"; search_threads="$2"; shift 2 ;;
    --print-command) print_command=1; shift ;;
    --) shift; gtp_args+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) gtp_args+=("$1"); shift ;;
  esac
done

[[ "${device}" =~ ^[0-9]+$ ]] || die "device ordinal must be non-negative"
for value_name in tactic_batch batch lanes search_threads; do
  value="${!value_name}"
  [[ -z "${value}" || "${value}" =~ ^[1-9][0-9]*$ ]] \
    || die "${value_name//_/-} must be positive"
done
(( no_plan == 0 )) || plan=""

repo="${SCRIPT_DIR}"
[[ -r "${repo}/cpp/configs/gtp_example.cfg" ]] || die "repository files are missing"
python="${KATAGO_PYTHON:-$(command -v python3 || command -v python || true)}"

if [[ -z "${model}" ]]; then
  for candidate in "${SCRIPT_DIR}/assets/${MODEL_NAME}" "${SCRIPT_DIR}/models/${MODEL_NAME}"; do
    [[ -r "${candidate}" ]] && model="${candidate}" && break
  done
fi
[[ -r "${model}" ]] || die "model not found; pass --model PATH"
model="$(readlink -m -- "${model}")"
config="${config:-${repo}/cpp/configs/gtp_example.cfg}"
[[ -r "${config}" ]] || die "GTP config not found: ${config}"
config="$(readlink -m -- "${config}")"

if (( no_plan == 0 )) && [[ -z "${plan}" ]]; then
  [[ -n "${python}" && -x "${python}" ]] || die "Python 3 is needed only for automatic plan selection"
  architecture="$(PYTHONPATH="${repo}/python" "${python}" -c '
import sys
from portable_cuda_device import query_cuda_device
cc = tuple(query_cuda_device(int(sys.argv[1]))["compute_capability"])
try: print({(8,6): "sm86", (8,9): "sm89", (12,0): "sm120"}[cc])
except KeyError: raise SystemExit(f"unsupported CUDA compute capability {cc}")
' "${device}")" || die "cannot identify CUDA architecture for device ${device}"
  mapfile -t matching_plans < <(find "${SCRIPT_DIR}/final-migration/plans/${architecture}" \
    -type f -name best-tactic-plan.json -print 2>/dev/null | sort)
  if (( ${#matching_plans[@]} == 1 )); then
    plan="${matching_plans[0]}"
  elif (( ${#matching_plans[@]} > 1 )); then
    die "multiple ${architecture} presets are available; select one with --plan"
  else
    printf '[katago-run] WARNING: no %s tactic preset found; using normal CUDA\n' "${architecture}" >&2
  fi
fi
if [[ -n "${plan}" ]]; then
  plan="$(readlink -m -- "${plan}")"
  [[ -r "${plan}" ]] || die "plan not found: ${plan}"
fi

if [[ -n "${katago}" ]]; then
  katago="$(readlink -m -- "${katago}")"
  [[ -x "${katago}" ]] || die "KataGo executable is not executable: ${katago}"
else
  for candidate in \
    "${SCRIPT_DIR}/katago" "${SCRIPT_DIR}/bin/katago" \
    "${SCRIPT_DIR}/cpp/build/katago" "${SCRIPT_DIR}/build/katago"; do
    [[ -x "${candidate}" ]] && katago="$(readlink -m -- "${candidate}")" && break
  done
fi
[[ -n "${katago}" ]] || die "KataGo executable not found; pass --katago PATH"

declare -a overrides=("cudaDeviceToUse=${device}")
[[ -z "${plan}" ]] || overrides+=("cudaTacticPlanFile=${plan}")
[[ -z "${tactic_batch}" ]] || overrides+=("cudaTacticPlanBatch=${tactic_batch}")
[[ -z "${batch}" ]] || overrides+=("nnMaxBatchSize=${batch}")
[[ -z "${lanes}" ]] || overrides+=("numNNServerThreadsPerModel=${lanes}")
[[ -z "${search_threads}" ]] || overrides+=("numSearchThreads=${search_threads}")
override_string="$(IFS=,; printf '%s' "${overrides[*]}")"

declare -a command=(
  "${katago}" gtp -model "${model}" -config "${config}"
  -override-config "${override_string}" "${gtp_args[@]}"
)
printf '[katago-run] device=%s plan=%s model=%s\n' \
  "${device}" "${plan:-normal-cuda}" "${model}" >&2
if (( print_command )); then
  printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
exec "${command[@]}"
