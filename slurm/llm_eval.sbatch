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
#SBATCH --account=plgunhype-gpu-gh200       # ← replace with your Helios grant id
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
# Aya-Expanse-8B is a gated Cohere model — HF_TOKEN is REQUIRED.
# Steps:
#   1. Sign in at https://huggingface.co and accept license on the model page:
#      https://huggingface.co/CohereForAI/aya-expanse-8b
#   2. Create a token at https://huggingface.co/settings/tokens
#   3. Add it to $SCRATCH/kara/.env as: HF_TOKEN=hf_xxxxxx
ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set." >&2
    echo "Aya-Expanse-8B is a gated model and requires HuggingFace authentication." >&2
    echo "  1. Accept the license at https://huggingface.co/CohereForAI/aya-expanse-8b" >&2
    echo "  2. Create a token at https://huggingface.co/settings/tokens" >&2
    echo "  3. Add it to ${ENV_FILE} as: HF_TOKEN=hf_xxxxxx" >&2
    exit 1
fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

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
echo "=== [1/4] LLM second-annotator ==="
python -m src.annotate_llm

# After the annotator finishes, rebuild the dataset — this picks up the new
# LLM labels and computes Cohen's kappa (human vs LLM) automatically.
# Also ensures train/val/test splits exist before the eval steps below.
echo ""
echo "=== [2/4] Build dataset + compute IAA (Cohen's kappa) ==="
python -m src.build_dataset

echo ""
echo "=== [3/4] Zero-shot eval ==="
python -m src.experiments.exp_llm_zeroshot

echo ""
echo "=== [4/4] Few-shot eval ==="
python -m src.experiments.exp_llm_fewshot

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
