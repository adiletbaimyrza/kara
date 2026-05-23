#!/bin/bash
#SBATCH --job-name=kara-llm
#SBATCH --account=plgnlp-gpu          # ← replace with your Athena grant id (e.g. plg<grantname>-gpu)
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/llm-%j.out
#SBATCH --error=slurm/logs/llm-%j.err

# ── Cyfronet Athena LLM job (single A100 80GB) ──
# Runs (1) LLM-as-annotator, (2) zero-shot eval, (3) few-shot eval.
# All produce GPU-bound work. Aya-Expanse-8B weights fit comfortably on
# Athena's A100-80GB in bf16. Expected wallclock: ~35-45 min total
# (model loading ~2 min + ~1 sec/item × 1202 items + experiments).

set -euo pipefail
mkdir -p slurm/logs

module purge
module load Python/3.11.5 CUDA/12.2

# Activate venv (assumed pre-created with `python -m venv .venv` and pip install -r requirements.txt)
source .venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

echo "=== LLM second-annotator ==="
python -m src.annotate_llm

echo "=== Zero-shot eval ==="
python -m src.experiments.exp_llm_zeroshot

echo "=== Few-shot eval ==="
python -m src.experiments.exp_llm_fewshot
