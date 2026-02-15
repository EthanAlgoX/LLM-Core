#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Activate conda environment for reproducible runs.
if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Please install conda or activate env manually."
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetune

python -m pip install -r requirements.txt
python grpo_demo.py
