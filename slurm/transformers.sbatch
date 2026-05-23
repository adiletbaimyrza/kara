#!/bin/bash
#SBATCH --job-name=kara-tfm
#SBATCH --account=plgnlp-gpu          # ← replace with your Athena grant id (e.g. plg<grantname>-gpu)
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/tfm-%j.out
#SBATCH --error=slurm/logs/tfm-%j.err

# Fine-tune mBERT + XLM-R on the Kyrgyz hate-speech dataset.
# Single A100 80GB on Cyfronet Athena. Expected wallclock ~20-30 min total
# (~10 min per model × 2 models + setup).

set -euo pipefail
mkdir -p slurm/logs

module purge
module load Python/3.11.5 CUDA/12.2

source .venv/bin/activate
cd "$SLURM_SUBMIT_DIR"

python -m src.experiments.exp_transformer --model both
