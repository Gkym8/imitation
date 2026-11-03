#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "Usage: bash train.sh TASK_NAME TASK_CONFIG EXPERT_DATA_NUM SEED ACTION_DIM GPU_IDS [RUN_TAG]" >&2
    echo "GPU_IDS accepts one GPU (for example 0) or a comma-separated list (for example 0,1)." >&2
    echo "RUN_TAG is optional and is appended to the complete task run name." >&2
    echo "Select camera views with workspace.train_dataset.image_keys in imitation_learning/configs/task/rmbench.yaml." >&2
    echo "Converted datasets are stored under RMBench/data/imitation_learning_policies by default." >&2
    exit 2
fi

task_name=$1
task_config=$2
expert_data_num=$3
seed=$4
action_dim=$5
gpu_ids=$6
run_tag=${7:-}

if [[ -n ${run_tag} && ! ${run_tag} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dots, underscores, and hyphens: ${run_tag}" >&2
    exit 2
fi

if [[ ! ${gpu_ids} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be one GPU ID or a comma-separated list, got: ${gpu_ids}" >&2
    exit 2
fi

IFS=',' read -r -a gpu_id_list <<< "${gpu_ids}"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_id_list[@]}"; do
    if [[ -n ${seen_gpu_ids[${gpu_id}]:-} ]]; then
        echo "GPU_IDS contains a duplicate GPU ID: ${gpu_id}" >&2
        exit 2
    fi
    seen_gpu_ids[${gpu_id}]=1
done
num_processes=${#gpu_id_list[@]}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
processed_data_dir=${RMBENCH_PROCESSED_DATA_DIR:-${project_root}/data/imitation_learning_policies}
dataset_name="${task_name}-${task_config}-${expert_data_num}"
dataset_path="${processed_data_dir}/${dataset_name}.zarr"
normalizer_path="${processed_data_dir}/${dataset_name}_normalizer.json"
run_name="${dataset_name}-seed${seed}"
if [[ -n ${run_tag} ]]; then
    run_name="${run_name}_${run_tag}"
fi
run_name_suffix="_${run_name}"

# Use the Python executable from the currently activated environment by
# default. Overrides remain available when conversion and policy training must
# use different environments, without embedding machine-specific paths here.
default_python=${PYTHON_BIN:-python}
rmbench_python=${RMBENCH_PYTHON:-${default_python}}
policy_python=${POLICY_PYTHON:-${default_python}}

if ! command -v "${rmbench_python}" >/dev/null 2>&1; then
    echo "RMBench Python executable not found: ${rmbench_python}" >&2
    echo "Activate the desired environment or set RMBENCH_PYTHON." >&2
    exit 1
fi
if ! command -v "${policy_python}" >/dev/null 2>&1; then
    echo "Policy Python executable not found: ${policy_python}" >&2
    echo "Activate the desired environment or set POLICY_PYTHON." >&2
    exit 1
fi

echo "RMBench source data: ${project_root}/data/data"
echo "Converted policy data: ${processed_data_dir}"
echo "Run name: ${run_name}"

if [[ ! -d ${dataset_path} ]]; then
    "${rmbench_python}" "${script_dir}/process_data.py" \
        "${task_name}" "${task_config}" "${expert_data_num}" \
        --output-path "${dataset_path}"
fi

if [[ ! -f ${normalizer_path} ]]; then
    "${policy_python}" "${script_dir}/fit_rmbench_normalizer.py" \
        "${dataset_path}" "${action_dim}"
fi

export HYDRA_FULL_ERROR=1
if [[ -n ${PYTHONPATH:-} ]]; then
    export PYTHONPATH="${script_dir}:${PYTHONPATH}"
else
    export PYTHONPATH="${script_dir}"
fi

cd "${script_dir}"
train_args=(
    scripts/train_policy.py
    +task_name="${task_name}"
    +task_yaml_name=rmbench
    +policy_name=diffusion_memory_transformer
    +project_name=rmbench
    +logger_project_name=rmbench
    +run_name="${run_name}"
    +run_name_suffix="${run_name_suffix}"
    +train_server_name=localhost
    +seed="${seed}"
    +workspace.train_dataset.root_dir="${processed_data_dir}"
    +workspace.train_dataset.compressed_dir="${processed_data_dir}"
    +workspace.train_dataset.normalizer_dir="${processed_data_dir}"
    +workspace.train_dataset.name="${dataset_name}"
    +workspace.train_dataset.source_data_meta.agent_pos.shape="[${action_dim}]"
    +workspace.train_dataset.source_data_meta.action.shape="[${action_dim}]"
    +workspace.train_dataset.output_data_meta.agent_pos.shape="[${action_dim}]"
    +workspace.train_dataset.output_data_meta.action.shape="[${action_dim}]"
)

if (( num_processes == 1 )); then
    export CUDA_VISIBLE_DEVICES=${gpu_ids}
    echo "Starting single-GPU training on GPU ${gpu_ids}."
    "${policy_python}" "${train_args[@]}"
else
    if ! "${policy_python}" -c "import accelerate" >/dev/null 2>&1; then
        echo "The policy Python environment does not contain Accelerate: ${policy_python}" >&2
        echo "Activate an environment with Accelerate or set POLICY_PYTHON." >&2
        exit 1
    fi
    main_process_port=${RMBENCH_MAIN_PROCESS_PORT:-$((29500 + RANDOM % 1000))}
    echo "Starting DDP training on GPUs ${gpu_ids} (${num_processes} processes, port ${main_process_port})."
    "${policy_python}" -m accelerate.commands.launch \
        --multi_gpu \
        --gpu_ids "${gpu_ids}" \
        --num_machines 1 \
        --num_processes "${num_processes}" \
        --main_process_port "${main_process_port}" \
        "${train_args[@]}"
fi
