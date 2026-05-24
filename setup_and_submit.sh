#!/bin/bash -l
# One-shot setup and job submission for kyrgyz-hsd on Helios.
# Safe to rerun — all steps are idempotent.
#
# Usage (from $SCRATCH/kyrgyz-hsd):
#   bash setup_and_submit.sh                # runs: setup venv (if needed) → llm + transformer jobs
#   bash setup_and_submit.sh llm            # only LLM job
#   bash setup_and_submit.sh transformer    # only transformer job

ml ML-bundle/25.10

set -euo pipefail

WHICH="${1:-all}"

SCRATCH_ROOT="${SCRATCH}/kyrgyz-hsd"
REPO_DIR="${SCRATCH_ROOT}"
VENV_DIR="${SCRATCH_ROOT}/venv"
HF_HOME="${SCRATCH}/hf_home"

echo "=========================================="
echo "kyrgyz-hsd setup-and-submit"
echo "=========================================="
echo "Mode:             ${WHICH}"
echo "Scratch root:     ${SCRATCH_ROOT}"
echo "HF cache:         ${HF_HOME}"
echo "=========================================="
echo ""

echo "[1/4] Creating directories..."
cd "${REPO_DIR}"
mkdir -p logs
mkdir -p "${HF_HOME}"
mkdir -p results figures tables
echo "  OK"
echo ""

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

echo "[3/4] Checking Python venv..."
VENV_MARKER="${VENV_DIR}/.khsd_deps_installed"
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

# Skip-if-done logic:
#   - LLM job is SKIPPED entirely if annotations_llm.csv has all 1202 entries
#     AND both exp7/exp8 metrics.json exist. Set KARA_FORCE=1 to override.
#   - The individual scripts inside each job ALSO skip done sub-steps:
#       annotate_llm.py skips already-labelled IDs
#       Experiment.run_logged() skips if metrics.json already exists
#     So even if you do submit, only the missing work runs.
#
# In --all mode, jobs run SEQUENTIALLY: LLM → transformer (afterok dep).
# This guarantees build_dataset sees the COMPLETE LLM annotations
# (otherwise IAA would be computed on a partial labelset).
echo "[4/4] Detecting completion state and submitting jobs..."

# ── Check what's done ──
LLM_CSV="${REPO_DIR}/data/annotated/annotations_llm.csv"
LLM_CSV_ROWS=0
if [ -f "${LLM_CSV}" ]; then
    LLM_CSV_ROWS=$(($(wc -l < "${LLM_CSV}") - 1))      # minus header
fi
LLM_ANNOTATOR_DONE=0
if [ "${LLM_CSV_ROWS}" -ge 1202 ]; then
    LLM_ANNOTATOR_DONE=1
fi

[ -f "${REPO_DIR}/results/exp7_llm_zeroshot/metrics.json" ] && EXP7_DONE=1 || EXP7_DONE=0
[ -f "${REPO_DIR}/results/exp8_llm_fewshot/metrics.json"  ] && EXP8_DONE=1 || EXP8_DONE=0
[ -f "${REPO_DIR}/results/exp5_mbert/metrics.json"        ] && EXP5_DONE=1 || EXP5_DONE=0
[ -f "${REPO_DIR}/results/exp6_xlmr/metrics.json"         ] && EXP6_DONE=1 || EXP6_DONE=0

echo "  State:"
echo "    LLM annotator  : $([ "${LLM_ANNOTATOR_DONE}" = 1 ] && echo "DONE (${LLM_CSV_ROWS}/1202)" || echo "incomplete (${LLM_CSV_ROWS}/1202)")"
echo "    exp7 zero-shot : $([ "${EXP7_DONE}" = 1 ] && echo DONE || echo missing)"
echo "    exp8 few-shot  : $([ "${EXP8_DONE}" = 1 ] && echo DONE || echo missing)"
echo "    exp5 mBERT     : $([ "${EXP5_DONE}" = 1 ] && echo DONE || echo missing)"
echo "    exp6 XLM-R     : $([ "${EXP6_DONE}" = 1 ] && echo DONE || echo missing)"

FORCE_ALL=0
if [ "${KARA_FORCE:-0}" = "1" ]; then
    FORCE_ALL=1
    echo "  KARA_FORCE=1 set — submitting all jobs regardless of state."
fi
echo ""

# ── Decide LLM job ──
LLM_JOB_ID=""
LLM_DEP=""
if [ "${WHICH}" = "llm" ] || [ "${WHICH}" = "all" ]; then
    if [ "${LLM_ANNOTATOR_DONE}" = 1 ] && [ "${EXP7_DONE}" = 1 ] && [ "${EXP8_DONE}" = 1 ] && [ "${FORCE_ALL}" = 0 ]; then
        echo "  LLM job:         SKIPPED (annotator + exp7 + exp8 all done; set KARA_FORCE=1 to override)"
    else
        LLM_JOB_ID=$(sbatch --parsable ${SETUP_DEP} slurm/llm_eval.sbatch)
        echo "  LLM job:         ${LLM_JOB_ID}"
        LLM_DEP="--dependency=afterok:${LLM_JOB_ID}"
    fi
fi

# ── Decide transformer job ──
if [ "${WHICH}" = "transformer" ] || [ "${WHICH}" = "all" ]; then
    if [ "${EXP5_DONE}" = 1 ] && [ "${EXP6_DONE}" = 1 ] && [ "${FORCE_ALL}" = 0 ]; then
        echo "  Transformer job: SKIPPED (exp5 + exp6 both done; set KARA_FORCE=1 to override)"
    else
        # Chain after LLM if we just submitted it; else use setup dep alone
        if [ -n "${LLM_DEP}" ]; then
            TFM_JOB_ID=$(sbatch --parsable ${LLM_DEP} slurm/transformers.sbatch)
            echo "  Transformer job: ${TFM_JOB_ID}  (waits for LLM job ${LLM_JOB_ID})"
        else
            TFM_JOB_ID=$(sbatch --parsable ${SETUP_DEP} slurm/transformers.sbatch)
            echo "  Transformer job: ${TFM_JOB_ID}"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Queue status:"
squeue -u $USER
echo "=========================================="
echo ""
echo "Watch logs with:"
echo "  tail -f logs/khsd-llm-*.log"
echo "  tail -f logs/khsd-tfm-*.log"
