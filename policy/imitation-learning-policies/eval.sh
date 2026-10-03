#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "Usage: bash eval.sh TASK_NAME TASK_CONFIG CKPT_SETTING EXPERT_DATA_NUM SEED GPU_ID CHECKPOINT_PATH" >&2
    exit 2
fi

policy_name=imitation-learning-policies
task_name=$1
task_config=$2
ckpt_setting=$3
expert_data_num=$4
seed=$5
gpu_id=$6
checkpoint_path=$7

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
if [[ ${checkpoint_path} != /* ]]; then
    checkpoint_path="$(cd -- "$(dirname -- "${checkpoint_path}")" && pwd)/$(basename -- "${checkpoint_path}")"
fi
if [[ ! -f ${checkpoint_path} ]]; then
    echo "Checkpoint not found: ${checkpoint_path}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mGPU to use: ${gpu_id}\033[0m"
echo "Checkpoint: ${checkpoint_path}"

cd "${project_root}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --policy_name "${policy_name}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --checkpoint_path "${checkpoint_path}"
