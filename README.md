# Kyrgyz Hate Speech Detection

NLP Master's course project. We build the first Kyrgyz hate-speech detection
dataset from ~13.9k YouTube comments and benchmark **8 systems** across
classical ML, fine-tuned multilingual transformers, and LLM prompting.

The pipeline is split into two stages on purpose so the user can re-run
analysis without re-training:

1. **`python -m src.run_all`** trains every model and writes per-experiment
   outputs under `results/<exp_name>/` plus a global `results/summary.csv`.
2. **`python -m src.make_figures`** reads `results/` + `data/splits/` and
   writes every figure under `figures/` and every table under `tables/`.

The notebook `notebooks/final.ipynb` is the submission artefact and surfaces
everything top-to-bottom.

---

## Repository layout

```
data/
  raw/                            # 13 YouTube .info.json files
  comments_filtered.csv           # 13,902 clean comments
  candidates_hate.csv             # 702 keyword-matched candidates
  candidates_random.csv           # 500 random non-matching comments
  annotated/                      # built by you + LLM
  splits/                         # train.csv / val.csv / test.csv
src/
  preprocess.py
  annotate_cli.py                 # terminal labelling UI
  annotate_llm.py                 # Aya-Expanse-8B as 2nd annotator
  build_dataset.py                # merge labels, Cohen's κ, splits
  experiments/                    # exp1..exp8
  run_all.py                      # orchestrator
  make_figures.py                 # figures + tables
  error_analysis.py
slurm/                            # Cyfronet Helios SLURM scripts
notebooks/                        # 01_data_eda, 02_annotation_eda,
                                  # 03_results_viz, final
paper/outline.md                  # ACL-style section skeleton
DISCOVERIES.md                    # running log of surprising findings
results/, figures/, tables/       # built by the scripts
filter_comments.py                # legacy: collected raw -> filtered
find_hate_candidates.py           # legacy: filtered -> candidate pools
```

---

## Reproduction

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data already exists

`data/comments_filtered.csv`, `data/candidates_hate.csv`, and
`data/candidates_random.csv` are committed — produced by
`filter_comments.py` and `find_hate_candidates.py` from the raw YouTube data
in `data/raw/`. You don't need to re-run those unless you collect more data.

### 3. Annotate

```bash
python -m src.annotate_cli                    # ~2–3 hours of your time
                                              # keys: h o n s u q
python -m src.annotate_cli --stats            # progress report
```

The CLI shows one comment at a time and you label it with a single keystroke.
Every label is flushed to disk immediately, so crashes don't lose progress.

In parallel, on a GPU node (locally or on Helios), run the LLM as the second
annotator:

```bash
# locally (needs a GPU and the heavy deps)
python -m src.annotate_llm

# OR on Helios
sbatch slurm/llm_eval.sbatch
```

Then build the dataset (merges human + LLM labels, computes Cohen's κ, writes
splits):

```bash
python -m src.build_dataset
```

### 4. Run experiments

```bash
# Classical baselines (CPU, ~5 min)
python -m src.run_all --only exp1,exp2,exp3,exp4

# Transformer fine-tuning (Helios A100, ~20 min)
sbatch slurm/transformers.sbatch

# LLM zero/few-shot (Helios A100, ~30 min)
sbatch slurm/llm_eval.sbatch

# Or everything at once on a beefy GPU machine:
python -m src.run_all
```

### 5. Build figures + tables

```bash
python -m src.make_figures
```

### 6. Open the submission notebook

```bash
jupyter notebook notebooks/final.ipynb
```

---

## Helios notes

The two SLURM scripts in `slurm/` are templates — replace `--account=plgnlp-gpu`
with your actual grant id. They assume:

- a virtual environment at `.venv/` with `pip install -r requirements.txt`,
- `Python/3.11.5` and `CUDA/12.2` modules loaded.

Aya-Expanse-8B weights fit comfortably on a 40 GB A100 in bf16. The first run
will download the model from Hugging Face; set `HF_HOME` to a persistent scratch
dir if you want to cache it across jobs.

---

## What the experiments answer

- **RQ1**: Does aggressive preprocessing help TF-IDF for Kyrgyz?  *(Exp 1 vs 2)*
- **RQ2**: Do char n-grams beat word n-grams on a morphologically rich language?  *(Exp 1 vs 3)*
- **RQ3**: How far does classical ML go on ~1k examples?  *(Exps 1–4)*
- **RQ4**: Do multilingual transformers fine-tuned on ~1k examples beat TF-IDF?  *(Exp 4 vs 5/6)*
- **RQ5**: mBERT vs XLM-R — which wins?  *(Exp 5 vs 6)*
- **RQ6**: Can a zero-shot LLM substitute for fine-tuning?  *(Exp 5/6 vs 7)*
- **RQ7**: Do in-context examples close the gap?  *(Exp 7 vs 8)*

Answers appear in `tables/results_main.md` and are summarised in the paper.

---

## Reproducibility

Every script seeds `numpy` / `random` / `torch` with `RNG_SEED=42` (`src/paths.py`).
Re-running `python -m src.run_all` produces identical metrics for classical
experiments. Transformer experiments are seeded but training has small
non-determinism from GPU kernels — variance is reported in the paper.
