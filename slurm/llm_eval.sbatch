#!/bin/bash -l
# Cyfronet Helios LLM job (single GH200 GPU, aarch64).
# Runs (1) LLM-as-annotator, (2) zero-shot eval, (3) few-shot eval.
# Aya-Expanse-8B weights fit comfortably on a GH200 96GB in bf16.
# Expected wallclock: ~40-60 min total.
#
# Usage (from $SCRATCH/kara):
#   sbatch slurm/llm_eval.sbatch

#SBATCH --job-name=kara-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgnlpkara-gpu-gh200       # ← replace with your Helios grant id
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err

ml ML-bundle/25.10

set -euo pipefail

SCRATCH_ROOT=${SCRATCH}/kara
REPO_DIR=${SCRATCH_ROOT}
VENV_DIR=${SCRATCH_ROOT}/venv
HF_HOME=${SCRATCH}/hf_home

mkdir -p logs "${HF_HOME}"

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}"

# Load HF_TOKEN (and any other secrets) from .env if present.
ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=========================================="
echo "Kara LLM job"
echo "=========================================="
echo "Repo:      ${REPO_DIR}"
echo "Venv:      ${VENV_DIR}"
echo "HF_HOME:   ${HF_HOME}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader || echo unknown)"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

echo ""
echo "=== [1/3] LLM second-annotator ==="
python -m src.annotate_llm

echo ""
echo "=== [2/3] Zero-shot eval ==="
python -m src.experiments.exp_llm_zeroshot

echo ""
echo "=== [3/3] Few-shot eval ==="
python -m src.experiments.exp_llm_fewshot

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
