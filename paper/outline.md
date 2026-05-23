# Paper Outline — Kyrgyz Hate Speech Detection

Target venue style: ACL short paper (4–6 pages + references). The final
`paper/PAPER.md` will be filled in once all experiments are run.

---

## 1. Abstract  (~150 words)
- Kyrgyz is low-resource; no public hate-speech dataset exists.
- We construct a 1k-scale annotated dataset from YouTube comments.
- We benchmark 8 systems across classical ML, fine-tuned multilingual
  transformers, and zero-/few-shot LLM prompting.
- Headline finding (filled after results): which family wins, by how much,
  and what fails most.

## 2. Introduction  (~0.75 pages)
Contributions stated as 3 bullets:
1. **First Kyrgyz hate-speech dataset** (~1k examples, 3-class).
2. **Benchmark** of TF-IDF, transformer, and LLM approaches.
3. **Error analysis** for a morphologically rich, low-resource Turkic
   language — what hate looks like in Kyrgyz YouTube comments.

## 3. Related Work  (~0.75 pages)
Paragraph plan (see `SOURCES.md` for the citation map):
- Hate speech detection in general.
- Low-resource hate speech.
- Turkic & Central Asian NLP (Kazakh, Turkish, Kyrgyz).
- Transformers & LLMs for hate-speech classification.

## 4. Dataset  (~1.25 pages)
- 4.1 Source: 13 Kyrgyz YouTube videos, comments scraped with yt-dlp.
- 4.2 Filtering pipeline (`filter_comments.py`): 11 stages, retention rate.
- 4.3 Candidate sampling (`find_hate_candidates.py`): keyword lexicon
  motivated by Davidson et al. (2017); biased pool + random pool.
- 4.4 Annotation: schema (3 classes), guidelines, self-annotation + LLM
  second-annotator (Aya-Expanse-8B), Cohen's κ.
- 4.5 Final dataset statistics: per-class counts, splits, vocabulary size.
- **Figure: pipeline_flowchart.png + annotation_flowchart.png**
- **Tables: dataset_stats.tex + annotation_iaa.tex**

## 5. Models & Experimental Setup  (~0.5 pages)
- 5.1 Classical baselines: TF-IDF + LogReg (with/without preprocessing,
  word/char n-grams) and TF-IDF + LinearSVC.
- 5.2 Fine-tuned transformers: mBERT and XLM-RoBERTa-base, 5 epochs,
  bs 16, AdamW lr 2e-5.
- 5.3 LLM prompting: Aya-Expanse-8B zero-shot and 5-shot.
- Metrics: macro-F1 (primary), per-class F1, accuracy.
- Compute: Cyfronet Helios A100 for transformers and LLM; CPU for TF-IDF.

## 6. Results  (~0.75 pages)
- **Table: results_main.tex** — model × {Acc, macro-F1, macro-P, macro-R}.
- **Figure: results_f1_bar.png** — headline F1 comparison.
- **Figure: training_curves.png** — transformer convergence.
- Subsections:
  - 6.1 RQ1: preprocessing effect (Exp 1 vs 2).
  - 6.2 RQ2: char n-grams (Exp 1 vs 3).
  - 6.3 RQ3: classical ML ceiling (Exp 1–4).
  - 6.4 RQ4–5: transformers (Exp 5 vs 6).
  - 6.5 RQ6–7: LLM (Exp 7 vs 8).

## 7. Error Analysis  (~0.5 pages)
- Confusion matrix of best model.
- Categorised error examples (sarcasm, code-switching, cultural context,
  ambiguous annotation, quoted slur).
- **Figure: confusion_matrices.png + error_categories.png**
- **Table: error_examples.tex**

## 8. Discussion & Conclusion  (~0.5 pages)
Pull from `DISCOVERIES.md`. Acknowledge limitations: single annotator on
gold labels, narrow domain (YouTube political content), small size.
Future work: more annotators, more domains, more data.

## 9. References (20+)
From `SOURCES.md`.
