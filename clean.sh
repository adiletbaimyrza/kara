#!/bin/bash
# Reset Kara project state for a clean re-run.
# Safe on both Mac (local dev) and Helios (cluster).
#
# Cleanup levels (cumulative — each includes the previous):
#
#   bash clean.sh                # outputs only:   logs, results, figures, tables, splits
#   bash clean.sh --deep         # + venv marker:  forces pip reinstall on next run
#   bash clean.sh --llm-labels   # + LLM annotations CSV (forces re-running LLM annotator)
#   bash clean.sh --nuke         # + venv dir + HF cache (full reset, requires model re-download)
#
# Optional flags:
#   --yes / -y       skip the confirmation prompt
#   --dry-run        print what would be deleted, don't actually delete
#
# ALWAYS PRESERVED (never deleted by this script):
#   - data/raw/               original YouTube comments
#   - data/comments_filtered.csv, data/candidates_*.csv
#   - data/annotated/annotations_human.csv   ← 8 hours of manual labelling
#   - .env (HF_TOKEN, secrets)
#   - source code, SLURM scripts, paper drafts, DISCOVERIES.md
#
# On Helios: cancels only your kara-* SLURM jobs (leaves cpt_experiments etc alone).

set -euo pipefail

# ── Flag parsing ──────────────────────────────────────────────────────────────
DEEP=0
LLM_LABELS=0
NUKE=0
ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --deep)        DEEP=1 ;;
        --llm-labels)  LLM_LABELS=1 ;;
        --nuke)        DEEP=1; LLM_LABELS=1; NUKE=1 ;;
        --yes|-y)      ASSUME_YES=1 ;;
        --dry-run|-n)  DRY_RUN=1 ;;
        -h|--help)
            sed -n 's/^# \{0,1\}//p' "$0" | head -30
            exit 0 ;;
        *) echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Where are we? ─────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_DIR}"

# Detect Helios vs Mac
IS_HELIOS=0
if [ -n "${SCRATCH:-}" ] && command -v squeue >/dev/null 2>&1; then
    IS_HELIOS=1
fi

# ── Build the target list ─────────────────────────────────────────────────────
declare -a TARGETS=()

# Always: experiment outputs (cheap to regenerate)
TARGETS+=(
    "${REPO_DIR}/logs"
    "${REPO_DIR}/slurm/logs"
    "${REPO_DIR}/results"
    "${REPO_DIR}/figures"
    "${REPO_DIR}/tables"
    "${REPO_DIR}/data/splits"
)

# __pycache__ and .ipynb_checkpoints everywhere (bash 3.2 compatible — no mapfile)
while IFS= read -r line; do
    [ -n "$line" ] && TARGETS+=("$line")
done < <(find "${REPO_DIR}" -type d \( -name __pycache__ -o -name '.ipynb_checkpoints' \) 2>/dev/null)

# --deep: also remove venv marker (forces full pip reinstall on next setup_venv)
if [ "${DEEP}" -eq 1 ]; then
    if [ "${IS_HELIOS}" -eq 1 ]; then
        TARGETS+=("${SCRATCH}/kara/venv/.kara_deps_installed")
    fi
fi

# --llm-labels: also remove the LLM-annotator output (forces re-running it on GPU)
if [ "${LLM_LABELS}" -eq 1 ]; then
    TARGETS+=("${REPO_DIR}/data/annotated/annotations_llm.csv")
fi

# --nuke: also wipe venv dir + HF cache (full reset, redownloads model on next run)
if [ "${NUKE}" -eq 1 ]; then
    if [ "${IS_HELIOS}" -eq 1 ]; then
        TARGETS+=(
            "${SCRATCH}/kara/venv"
            "${SCRATCH}/hf_home"
        )
    else
        # Mac: don't touch ~/.cache/huggingface globally — it's shared across projects.
        # If you really want to wipe it, do it manually: rm -rf ~/.cache/huggingface
        echo "  (On Mac, --nuke does not touch ~/.cache/huggingface — wipe manually if needed)"
    fi
fi

# ── Show the plan ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "Kara clean — plan"
echo "=========================================="
echo "Where:   ${REPO_DIR}"
echo "Helios:  $( [ "${IS_HELIOS}" -eq 1 ] && echo yes || echo no )"
echo "Level:   $( [ "${NUKE}" -eq 1 ] && echo nuke \
              || ([ "${LLM_LABELS}" -eq 1 ] && [ "${DEEP}" -eq 1 ] && echo "deep + llm-labels") \
              || ([ "${LLM_LABELS}" -eq 1 ] && echo "llm-labels") \
              || ([ "${DEEP}" -eq 1 ] && echo deep) \
              || echo outputs )"
echo "Dry run: $( [ "${DRY_RUN}" -eq 1 ] && echo yes || echo no )"
echo ""
echo "Will remove:"
ANYTHING=0
for t in "${TARGETS[@]}"; do
    [ -z "$t" ] && continue
    if [ -e "$t" ]; then
        size=$(du -sh "$t" 2>/dev/null | cut -f1)
        printf "  %-10s %s\n" "${size:-?}" "$t"
        ANYTHING=1
    fi
done
if [ "${ANYTHING}" -eq 0 ]; then
    echo "  (nothing to remove — already clean)"
fi
echo ""
echo "Will preserve (never deleted):"
echo "  data/raw/                                  (original YouTube data)"
echo "  data/comments_filtered.csv                 (filter output)"
echo "  data/candidates_*.csv                      (candidate pools)"
echo "  data/annotated/annotations_human.csv       ★ your manual labels"
if [ "${LLM_LABELS}" -eq 0 ]; then
    echo "  data/annotated/annotations_llm.csv         (use --llm-labels to also remove)"
fi
echo "  .env                                       (HF_TOKEN, secrets)"
echo "  source code, slurm/*.sh, paper/, DISCOVERIES.md"
if [ "${IS_HELIOS}" -eq 1 ]; then
    if [ "${NUKE}" -eq 0 ]; then
        echo "  ${SCRATCH}/kara/venv                       (use --nuke to also remove)"
        echo "  ${SCRATCH}/hf_home                         (use --nuke to also remove)"
    fi
fi
echo ""
if [ "${IS_HELIOS}" -eq 1 ]; then
    echo "Will also cancel queued/running kara-* SLURM jobs (leaves other jobs alone)."
    echo ""
fi
echo "=========================================="

# ── Confirm ───────────────────────────────────────────────────────────────────
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "Dry run — nothing was deleted."
    exit 0
fi
if [ "${ANYTHING}" -eq 0 ] && [ "${IS_HELIOS}" -eq 0 ]; then
    exit 0
fi
if [ "${ASSUME_YES}" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# ── Cancel Kara SLURM jobs first (Helios only) ────────────────────────────────
if [ "${IS_HELIOS}" -eq 1 ]; then
    echo ""
    echo "Cancelling kara-* SLURM jobs..."
    # Only cancel jobs whose name starts with "kara-" or matches "setup_venv"
    KARA_JOBS=$(squeue -u "$(whoami)" -h -o '%i %j' 2>/dev/null \
                | awk '$2 ~ /^(kara-|setup_venv)/ {print $1}')
    if [ -n "${KARA_JOBS}" ]; then
        echo "${KARA_JOBS}" | xargs scancel
        echo "  Cancelled: $(echo "${KARA_JOBS}" | wc -w | tr -d ' ') jobs"
    else
        echo "  No kara jobs in queue."
    fi
fi

# ── Remove targets ────────────────────────────────────────────────────────────
echo ""
echo "Removing files..."
removed=0
for t in "${TARGETS[@]}"; do
    [ -z "$t" ] && continue
    if [ -e "$t" ]; then
        rm -rf "$t"
        echo "  removed: $t"
        removed=$((removed + 1))
    fi
done
echo "  ${removed} target(s) removed."

# ── Final word ────────────────────────────────────────────────────────────────
echo ""
echo "Done."
if [ "${IS_HELIOS}" -eq 1 ]; then
    if [ "${NUKE}" -eq 1 ]; then
        echo "Next: bash setup_and_submit.sh"
        echo "  (will re-create venv from scratch and re-download Aya — ~10-15 min before jobs run)"
    elif [ "${DEEP}" -eq 1 ]; then
        echo "Next: bash setup_and_submit.sh"
        echo "  (will reinstall pip deps but keep HF cache)"
    else
        echo "Next: bash setup_and_submit.sh"
    fi
else
    echo "Next (Mac): python3 -m src.run_all --only exp1,exp2,exp3,exp4"
fi
