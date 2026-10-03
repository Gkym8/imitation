#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: bash process_data.sh TASK_NAME TASK_CONFIG EXPERT_DATA_NUM [DATA_ROOT] [OUTPUT_PATH]" >&2
    exit 2
fi

task_name=$1
task_config=$2
expert_data_num=$3
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../.." && pwd)"
data_root=${4:-${RMBENCH_DATA_ROOT:-${project_root}/data/data}}
processed_data_dir=${RMBENCH_PROCESSED_DATA_DIR:-${project_root}/data/imitation_learning_policies}
output_path=${5:-${processed_data_dir}/${task_name}-${task_config}-${expert_data_num}.zarr}

python_bin=${PYTHON_BIN:-python}

echo "Source RMBench data: ${data_root}"
echo "Converted policy data: ${output_path}"

"${python_bin}" "${script_dir}/process_data.py" \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --data-root "${data_root}" \
    --output-path "${output_path}"
