# Reproducing on Cyfronet Helios

End-to-end guide for running the dataset construction, training, and
evaluation pipeline on Cyfronet's Helios GH200 cluster. Local-only steps
(annotation, classical baselines) are noted inline.

## Prerequisites

- Helios account with access to the `plgrid-gpu-gh200` partition and a
  valid `--account=...` grant.
- Hugging Face account with a read token, license accepted at
  https://huggingface.co/CohereForAI/aya-expanse-8b (Aya is openly
  licensed; the token is recommended for rate-limit reasons).
- Python 3.11+ available via `ml ML-bundle/25.10` on Helios.

## 1. Clone and configure

From the Helios login node:

```bash
cd "$SCRATCH"
git clone https://github.com/adiletbaimyrza/kyrgyz-hsd.git
cd kyrgyz-hsd
cp .env.example .env
# edit .env and set HF_TOKEN=hf_xxx
```

Open every `slurm/*.sbatch` file and replace
`--account=plgunhype-gpu-gh200` with your own grant id. The same line
appears in `slurm/setup_venv.sh`, `slurm/transformers.sbatch`, and
`slurm/llm_eval.sbatch`.

## 2. Build the Python venv

Run once per fresh checkout. The venv must be built on a compute node
because the login node has no PyPI access and a different architecture.

```bash
sbatch slurm/setup_venv.sh
```

This installs `torch` from the PyTorch cu126 aarch64 wheel index
(the default PyPI ships CPU-only torch for aarch64), then installs the
rest of `requirements.txt`. A marker file at `venv/.khsd_deps_installed`
skips reinstallation on subsequent runs.

## 3. Prepare the data (one-off, CPU-only)

The raw YouTube comments live in `data/raw/`. The two derivative
files are not committed; regenerate them on the login node:

```bash
python filter_comments.py        # data/raw/  -> data/comments_filtered.csv
python find_hate_candidates.py   # comments_filtered.csv -> candidates_hate.csv, candidates_random.csv
```

## 4. Annotate (one-off, local or login node)

The human annotation CSV (`data/annotated/annotations_human.csv`) is
already committed. To re-annotate from scratch, run the CLI locally
(not on Helios — it needs a TTY):

```bash
python -m src.annotate_cli
```

For the LLM second-annotator, submit the GPU job described in step 6.

## 5. Submit-all wrapper

For a full GPU run (venv setup if needed, then the LLM and transformer
jobs in sequence), use:

```bash
bash setup_and_submit.sh           # all GPU jobs
bash setup_and_submit.sh llm       # only LLM job
bash setup_and_submit.sh transformer # only transformer job
```

The wrapper checks the venv marker and chains a setup job before the
training jobs if needed.

## 6. Individual SLURM jobs

| Job | Script | GPU | Wallclock | What it produces |
|---|---|---|---|---|
| LLM annotation + zero/few-shot eval | `slurm/llm_eval.sbatch` | 1× GH200 | ~40-60 min | `data/annotated/annotations_llm.csv`, `results/exp7_llm_zeroshot/`, `results/exp8_llm_fewshot/` |
| mBERT and XLM-RoBERTa fine-tuning | `slurm/transformers.sbatch` | 1× GH200 | ~20-30 min | `results/exp5_mbert/`, `results/exp6_xlmr/` |

Submit individually:

```bash
sbatch slurm/llm_eval.sbatch
sbatch slurm/transformers.sbatch
```

The LLM job also runs `src.build_dataset` after annotation so Cohen's κ
and the train/val/test splits land in `data/annotated/dataset_final.csv`
and `data/splits/`. The transformer job calls `src.build_dataset` first
to make sure splits exist.

## 7. Classical baselines (CPU, local or login node)

Classical TF-IDF + LogReg / SVM experiments run in roughly 5 minutes on
a laptop CPU:

```bash
python -m src.run_all --only exp1,exp2,exp3,exp4
```

Or run the full pipeline (classical + transformer + LLM) on a single
GPU machine:

```bash
python -m src.run_all
```

## 8. Build figures and tables

After all experiments have produced their `results/exp*/` directories:

```bash
python -m src.make_figures
```

This reads `results/summary.csv` and `data/splits/` and writes the
10 paper figures into `figures/` and 3 markdown/LaTeX tables into
`tables/`.

## Troubleshooting

- **fp16 training collapse.** XLM-RoBERTa under fp16 mixed precision
  collapsed to predicting the majority class on this dataset (macro-F1
  = 0.20). The transformer recipe in `src/experiments/exp_transformer.py`
  uses `bf16` instead, which has the same memory footprint but fp32's
  exponent range and resolves the LayerNorm overflow. Do not switch
  back to fp16.
- **HF token errors.** Aya-Expanse-8B requires HuggingFace
  authentication for rate-limit reasons. Verify
  `echo $HF_TOKEN` inside the job script, and re-check the license
  acceptance at https://huggingface.co/CohereForAI/aya-expanse-8b.
- **Login-node `pip install` fails on aarch64.** Helios login nodes
  are x86_64; the GH200 compute nodes are aarch64. Always run
  `setup_venv.sh` as a SLURM job, never on the login node.
- **HF cache evicted between jobs.** Set `HF_HOME=$SCRATCH/hf_home` in
  every job script (already done in the SLURM templates) so model
  weights persist across jobs.

## Reproducibility

`RNG_SEED=42` is set in every script (`src/paths.py`). Classical
experiments produce identical metrics across runs. Transformer training
is seeded but retains small non-determinism from GPU kernels; variance
is reported in [REPORT.md](REPORT.md).
