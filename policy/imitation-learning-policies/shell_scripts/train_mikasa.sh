#!/usr/bin/env bash
set -euo pipefail

policy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "${policy_dir}/train_mikasa.sh" "$@"
