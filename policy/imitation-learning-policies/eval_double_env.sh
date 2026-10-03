#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 8 ]]; then
    echo "Usage: bash eval_double_env.sh TASK_NAME TASK_CONFIG CKPT_SETTING EXPERT_DATA_NUM SEED GPU_ID POLICY_CONDA_ENV CHECKPOINT_PATH" >&2
    exit 2
fi

policy_name=imitation-learning-policies
task_name=$1
task_config=$2
ckpt_setting=$3
expert_data_num=$4
seed=$5
gpu_id=$6
policy_conda_env=$7
checkpoint_path=$8

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
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${project_root}"

echo "policy_conda_env is '$policy_conda_env'"

# Find an available port
FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# Start the server in the background
echo -e "\033[32m[server] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[server] Launching policy_model_server (PID will be recorded)...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py \
    --port "${FREE_PORT}" \
    --policy_conda_env "${policy_conda_env}" \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --checkpoint_path "${checkpoint_path}" &
SERVER_PID=$!

# Ensure the server is killed when this script exits
trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null" EXIT

conda deactivate

# Start the client in the foreground
echo -e "\033[34m[client] Starting eval_policy_client on port ${FREE_PORT}...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --port "${FREE_PORT}" \
    --policy_conda_env "${policy_conda_env}" \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --checkpoint_path "${checkpoint_path}"

echo -e "\033[33m[main] eval_policy_client has finished; the server will be terminated.\033[0m"
