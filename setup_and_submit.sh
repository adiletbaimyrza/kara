#!/bin/bash -l
# One-shot setup and job submission for Kara on Helios.
# Safe to rerun — all steps are idempotent.
#
# Usage (from $SCRATCH/kara):
#   bash setup_and_submit.sh                # runs: setup venv (if needed) → llm + transformer jobs
#   bash setup_and_submit.sh llm            # only LLM job
#   bash setup_and_submit.sh transformer    # only transformer job

ml ML-bundle/25.10

set -euo pipefail

WHICH="${1:-all}"

SCRATCH_ROOT="${SCRATCH}/kara"
REPO_DIR="${SCRATCH_ROOT}"
VENV_DIR="${SCRATCH_ROOT}/venv"
HF_HOME="${SCRATCH}/hf_home"

echo "=========================================="
echo "Kara setup-and-submit"
echo "=========================================="
echo "Mode:             ${WHICH}"
echo "Scratch root:     ${SCRATCH_ROOT}"
echo "HF cache:         ${HF_HOME}"
echo "=========================================="
echo ""

# ── [1/4] Directories (idempotent) ────────────────────────────────────────────
echo "[1/4] Creating directories..."
cd "${REPO_DIR}"
mkdir -p logs
mkdir -p "${HF_HOME}"
mkdir -p results figures tables
echo "  OK"
echo ""

# ── [2/4] Check .env / HF_TOKEN (optional but recommended) ────────────────────
echo "[2/4] Checking .env..."
ENV_FILE="${SCRATCH_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
    set -a; source "${ENV_FILE}"; set +a
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "  HF_TOKEN found."
    else
        echo "  .env present but HF_TOKEN not set (Aya is openly licensed, so this is optional)."
    fi
else
    echo "  No .env file. Aya-Expanse is openly licensed, so this is OK."
    echo "  If you want HF auth (e.g. for rate limits or future gated models):"
    echo "    cp .env.example ${ENV_FILE}"
    echo "    nano ${ENV_FILE}"
fi
echo ""

# ── [3/4] Venv (submit setup job if not ready) ────────────────────────────────
echo "[3/4] Checking Python venv..."
VENV_MARKER="${VENV_DIR}/.kara_deps_installed"
SETUP_DEP=""
if [ ! -f "${VENV_MARKER}" ]; then
    echo "  Venv not ready — submitting setup job on a GH200 compute node..."
    SETUP_JOB_ID=$(sbatch --parsable slurm/setup_venv.sh)
    echo "  Setup job: ${SETUP_JOB_ID}"
    echo "  Subsequent jobs will wait via --dependency=afterok:${SETUP_JOB_ID}"
    SETUP_DEP="--dependency=afterok:${SETUP_JOB_ID}"
else
    echo "  Venv ready."
fi
echo ""

# ── [4/4] Submit experiment jobs ──────────────────────────────────────────────
# In "all" mode, jobs run SEQUENTIALLY: LLM → transformer.
# Reason: the LLM annotator produces annotations_llm.csv, which build_dataset
# (called inside each job) uses to compute Cohen's kappa. If the transformer
# job runs in parallel, its build_dataset only sees partial LLM labels and
# IAA becomes meaningless.
echo "[4/4] Submitting experiment jobs..."

LLM_DEP=""
if [ "${WHICH}" = "llm" ] || [ "${WHICH}" = "all" ]; then
    LLM_JOB_ID=$(sbatch --parsable ${SETUP_DEP} slurm/llm_eval.sbatch)
    echo "  LLM job:         ${LLM_JOB_ID}"
    LLM_DEP="--dependency=afterok:${LLM_JOB_ID}"
fi

if [ "${WHICH}" = "transformer" ] || [ "${WHICH}" = "all" ]; then
    # If we just submitted the LLM job, chain transformer after it via
    # afterok. Otherwise (transformer-only mode) just use setup dep.
    if [ -n "${LLM_DEP}" ]; then
        TFM_JOB_ID=$(sbatch --parsable ${LLM_DEP} slurm/transformers.sbatch)
        echo "  Transformer job: ${TFM_JOB_ID}  (waits for LLM job ${LLM_JOB_ID})"
    else
        TFM_JOB_ID=$(sbatch --parsable ${SETUP_DEP} slurm/transformers.sbatch)
        echo "  Transformer job: ${TFM_JOB_ID}"
    fi
fi

echo ""
echo "=========================================="
echo "Queue status:"
squeue -u $USER
echo "=========================================="
echo ""
echo "Watch logs with:"
echo "  tail -f logs/kara-llm-*.log"
echo "  tail -f logs/kara-tfm-*.log"
