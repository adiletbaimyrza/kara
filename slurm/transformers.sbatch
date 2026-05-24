#!/bin/bash -l
# Cyfronet Helios transformer fine-tuning job (single GH200 GPU, aarch64).
# Fine-tunes mBERT + XLM-R on the Kyrgyz hate-speech dataset.
# Expected wallclock: ~20-30 min total (~10 min per model).
#
# Usage (from $SCRATCH/kyrgyz-hsd):
#   sbatch slurm/transformers.sbatch

#SBATCH --job-name=khsd-tfm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200       # ← replace with your Helios grant id
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

ml ML-bundle/25.10

set -euo pipefail

SCRATCH_ROOT=${SCRATCH}/kyrgyz-hsd
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH}/hf_home

mkdir -p logs "${HF_HOME}"

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"

ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=========================================="
echo "Kyrgyz-HSD transformer job"
echo "=========================================="
echo "Repo:      ${REPO_DIR}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo unknown)"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Ensure train/val/test splits exist (idempotent — quick CPU step).
# Required because clean.sh removes data/splits/ between runs.
echo ""
echo "=== [0/1] Ensuring splits exist ==="
python -m src.build_dataset

echo ""
echo "=== [1/1] Transformer fine-tuning (mBERT + XLM-R) ==="
python -m src.experiments.exp_transformer --model both

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
