#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${POLICY_PYTHON:-python}" "${script_dir}/scripts/train_mikasa.py" "$@"
