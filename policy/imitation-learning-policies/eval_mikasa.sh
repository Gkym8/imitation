#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
    echo "Usage: bash eval_mikasa.sh ENV_ID CHECKPOINT GPU [mikasa_eval.py options]" >&2
    exit 2
fi
policy_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_id=$1
checkpoint=$(realpath -e -- "$2")
export CUDA_VISIBLE_DEVICES=$3
shift 3
sim_root=${MIKASA_SIM_ROOT:-"$policy_root/../../../gated-memory-policy/mikasa-robo-env"}
output="$(dirname "$checkpoint")/eval_mikasa/$(date +%Y%m%d-%H%M%S-%N)"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
exec "${MIKASA_PYTHON:-python}" "$sim_root/eval/mikasa_eval.py" \
    --policy-root "$policy_root" --env-id "$env_id" --checkpoint "$checkpoint" \
    --abs-joint-pos --output-dir "$output" "$@"
