# Kyrgyz Hate Speech Detection — 3-Day Project Plan

## Paper Structure

```
1. Abstract
2. Introduction
3. Related Work
4. Dataset Construction
   4.1 Data Collection
   4.2 Annotation Schema & Guidelines
   4.3 Inter-Annotator Agreement
   4.4 Dataset Statistics
5. Models & Experimental Setup
   5.1 Baselines
   5.2 Transformer-based models
   5.3 LLM evaluation
6. Results
7. Error Analysis
8. Conclusion & Future Work
9. References (target: 20+ citations)
```

---

## Day 1 — Dataset Construction

**Goal: Collect and annotate ~1,000–1,500 Kyrgyz examples**

### Morning (3–4 hrs): Data Collection

Sources to scrape (prioritize by ease and Kyrgyz content density):
- YouTube comments — Kyrgyz political/sports channels (`yt-dlp` or YouTube Data API)
- VK.com — popular in Kyrgyzstan; public group comments (VK API)
- 24.kg, kabar.kg, akipress.com — news portal comments
- Telegram public channels (`telethon`)

Target: collect 3,000–5,000 raw posts, then annotate a curated subset.

### Afternoon (3–4 hrs): Annotation

Schema (binary): `hate` / `non-hate`
- Optional: add `offensive` as a third class if data permits

Steps:
1. Write 1-page annotation guidelines (definition, Kyrgyz examples, edge cases)
2. Self-annotate all ~1,500 samples
3. Recruit 1 second annotator for 200–300 sample subset
4. Compute Cohen's Kappa or Krippendorff's alpha (target: α > 0.6)

Tool: Label Studio (free, local) or Google Sheets for the second annotator.

### Evening (2–3 hrs): EDA & Preprocessing

- Class distribution, text length, vocab size
- Kyrgyz-specific preprocessing: Cyrillic normalization, remove URLs/emojis
- Train / val / test split: 70 / 10 / 20
- Save as CSV + HuggingFace Dataset format

**Deliverables:**
- `data/raw/` — collected posts
- `data/annotated/kyrgyz_hate_speech.csv`
- `data/splits/` — train/val/test
- `notebooks/01_data_analysis.ipynb`

---

## Day 2 — Model Training & Experiments

**Goal: Run 7 experiments across classical ML, transformers, and LLM**

### Morning (3–4 hrs): Classical ML Baselines

| # | Experiment | Model |
|---|---|---|
| 1 | Baseline | TF-IDF + Logistic Regression (no preprocessing) |
| 2 | Preprocessing ablation | TF-IDF + Logistic Regression (with preprocessing) |
| 3 | Classical ML | TF-IDF + SVM (LinearSVC) |

Metrics: F1 (macro), Precision, Recall, Accuracy

### Afternoon (3–4 hrs): Transformer Fine-tuning

| # | Experiment | Model |
|---|---|---|
| 4 | Multilingual BERT | `bert-base-multilingual-cased` fine-tuned |
| 5 | XLM-RoBERTa | `xlm-roberta-base` fine-tuned |

Setup: HuggingFace `Trainer` API, 3–5 epochs, batch size 16, AdamW. Use Google Colab free GPU if needed.

### Evening (2–3 hrs): LLM Evaluation

| # | Experiment | Approach |
|---|---|---|
| 6 | Zero-shot | Prompt LLM with task definition only |
| 7 | Few-shot | Add 3–5 labeled examples to prompt |

Free LLM options (no paid API):
- `Aya-23` / `Aya-Expanse` (multilingual, HuggingFace)
- `Qwen2.5` or `mT0`
- Ollama locally (`llama3.2` or `mistral`)

**Deliverables:**
- `notebooks/02_baselines.ipynb`
- `notebooks/03_transformers.ipynb`
- `notebooks/04_llm_eval.ipynb`
- `results/results_table.csv`

---

## Day 3 — Analysis & Paper Writing

**Goal: Full paper draft (6–8 pages)**

### Morning (3 hrs): Error Analysis + Visualizations

- Examine false positives and false negatives from best model (likely XLM-RoBERTa)
- Categorize at least 10 error examples:
  - Sarcasm / irony misclassified
  - Code-switching (Kyrgyz/Russian mixed)
  - Cultural context the model misses
  - Ambiguous annotation cases
- Create: confusion matrix, per-class F1 bar chart, training loss curve

### Afternoon (4 hrs): Write the Paper

| Section | Time |
|---|---|
| Introduction | 45 min |
| Related Work | 45 min |
| Dataset Construction | 45 min |
| Experiments & Results | 50 min |
| Error Analysis | 20 min |
| Conclusion | 15 min |
| Abstract (write last) | 20 min |
| References | 20 min |

**Introduction** — state 3 contributions:
1. First Kyrgyz hate speech dataset
2. Benchmark of ML / transformer / LLM models
3. Error analysis for a low-resource setting

**Related Work** — 4 paragraphs:
1. Hate speech detection in general
2. Low-resource hate speech datasets
3. Turkic & Central Asian NLP
4. LLMs for hate speech

### Evening (2 hrs): Polish & Cleanup

- Notebook runs end-to-end from top
- Add `requirements.txt`
- Proofread; verify all metric numbers match notebook output

**Deliverables:**
- `paper/kyrgyz_hate_speech.pdf` (or Overleaf `.tex`)
- Final cleaned notebooks
- `requirements.txt`

---

## Results Table (fill in on Day 2)

| Model | F1 (macro) | Precision | Recall | Accuracy |
|---|---|---|---|---|
| TF-IDF + LogReg (no preprocessing) | — | — | — | — |
| TF-IDF + LogReg (with preprocessing) | — | — | — | — |
| TF-IDF + SVM | — | — | — | — |
| mBERT (fine-tuned) | — | — | — | — |
| XLM-RoBERTa (fine-tuned) | — | — | — | — |
| LLM zero-shot | — | — | — | — |
| LLM few-shot | — | — | — | — |

---

## Research Questions

1. Does text preprocessing improve hate speech detection in Kyrgyz? *(Exp 1 vs 2)*
2. Do multilingual transformers outperform classical ML on a low-resource dataset? *(Exp 3 vs 5)*
3. Can zero/few-shot LLM prompting substitute for fine-tuning when data is scarce? *(Exp 5 vs 6/7)*

---

## Tools & Libraries

| Purpose | Tool |
|---|---|
| Data collection | `yt-dlp`, VK API, `telethon`, `requests` |
| Annotation | Label Studio (local) or Google Sheets |
| Classical ML | `scikit-learn` |
| Transformers | `transformers`, `datasets`, `torch` |
| LLM | `ollama` or HuggingFace `pipeline` |
| Metrics | `sklearn.metrics` |
| Paper | Overleaf (ACL template) or Google Docs |

---

## Grading Alignment

| Requirement | How it's covered |
|---|---|
| Sensible problem + data | Novel Kyrgyz dataset, clearly described |
| Baseline | TF-IDF + LogReg (Experiment 1) |
| ≥ 3 experiments | 7 experiments total |
| Impact of design decisions | Preprocessing ablation + model comparison |
| Error analysis | Day 3 morning |
| Conclusions | Paper Section 8 |
| Readable notebook | Day 3 cleanup pass |
| Bonus (Kyrgyz language) | Confirm with instructor |
