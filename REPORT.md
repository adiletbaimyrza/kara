# Hate Speech Detection in Kyrgyz: A Dataset, Benchmark, and Annotation Schema

**Author:** Adilet Baimyrza Uulu
**Course:** Natural Language Processing (Master's), Jagiellonian University
**Repository:** https://github.com/adiletbaimyrza/kyrgyz-hsd

---

## Abstract

Kyrgyz is a low-resource Turkic language with roughly 5 million speakers
and a vibrant online political discourse spanning women's rights, party
politics, inter-ethnic tension, and LGBT issues, all of which surface
hate-speech behaviour in the corpus I collect. No publicly available
hate-speech detection dataset exists. The situation differs sharply from
English: where MetaHate (Piot et al., 2024) alone aggregates 36 English
benchmark corpora and a taxonomy has converged across Davidson,
HateXplain, and Founta, Kyrgyz lacks both labelled data and a taxonomy
that fits its rhetorical register. Turkic-Islamic curse formulas,
post-colonial ethnic-loyalty pejoratives (манкурт), and Central Asian
inter-ethnic targeting do not transfer from US-English benchmarks. I
construct a 1,079-comment 3-class
annotated dataset built from Kyrgyz-language YouTube comments using a
Davidson-style biased+random candidate-sampling pipeline. I benchmark eight
systems spanning classical machine learning, fine-tuned multilingual
transformers, and zero-/few-shot LLM prompting. The headline finding is that
**classical TF-IDF char-n-gram + Logistic Regression (macro-F1 = 0.646)
outperforms every neural system tested**: fine-tuned
mBERT (0.557) and XLM-RoBERTa (0.431), and Aya-Expanse-8B both zero-shot
(0.354) and 5-shot (0.393). Character n-grams improve macro-F1 by +12 points
over word n-grams, which I attribute to the observation that 89.8% of
comments in my corpus use the Russian keyboard rather than the Kyrgyz-specific
Cyrillic letters (Ң, Ө, Ү), creating an orthographic mismatch with the
Wikipedia/CommonCrawl pretraining distributions of all neural systems I
test. Inter-annotator agreement between the human author and Aya-Expanse-8B
as a second annotator was κ = 0.100 on n = 1,079, with the LLM systematically
over-predicting `hate`: 46% of human-labelled `non_hate` items were called
`hate` by the LLM.

---

## 1. Introduction

Hate speech detection is a mature NLP subfield, but its empirical foundation
is heavily concentrated on English data from US-centric platforms (Davidson
et al., 2017; Mathew et al., 2021; Bourgeade et al., 2024). Low-resource
Turkic and Central Asian languages remain underrepresented, despite
politically active online communities where detection is operationally
relevant.

Neighbouring Turkic languages have begun to receive attention. Turkish has a
labelled hate-speech corpus and detection system (Toraman et al., 2022;
Yilmaz, 2025), and Kazakh has cyberbullying and offensive-language datasets
(Bekmagambetov et al., 2021). Kyrgyz has had no equivalent resource, and
this work builds the first one.

The paper is structured as follows. Section 2 describes the dataset
construction pipeline and annotation schema. Section 3 documents
culturally-specific hate-speech sub-registers observed in the corpus.
Section 4 describes experimental setup; Section 5 presents results.
Section 6 discusses broader implications and Section 7 concludes.

---

## 2. Dataset

### 2.1 Source

I collected approximately 15,000 comments from 13 Kyrgyz-language YouTube
videos using `yt-dlp`. The video selection was constrained to
politically-active channels (news, current events, political commentary)
because hate-speech base-rate is materially higher in political comment threads
than in general entertainment content. Channel-level diversity was prioritised:
the 13 videos span four distinct content domains (presidential commentary,
border-conflict reporting, opposition political commentary, social-issue
discussion).

### 2.2 Filtering pipeline

A multi-stage filtering pipeline (`filter_comments.py`) was applied to remove
noise. The 11 filtering stages, in order: (1) empty/whitespace, (2) word-count
< 3 or > 100, (3) letter-ratio < 40%, (4) Cyrillic-letter-ratio < 60%, (5)
excessive-character repeats, (6) within-comment token repetition ≥ 70%, (7)
URL / Telegram-link, (8) phone-number spam, (9) ≥3 @-mentions, (10) ≥3
hashtags, (11) canonical-form near-duplicate. After filtering, **13,902 unique
comments** remain (retention rate ~93%).

The largest single drop reason after empty/spam filtering was
near-duplicate removal, Kyrgyz political comment threads contain heavy
template-style repetition (often nationalist slogans), and post-deduplication
the corpus contains substantially more linguistic variation than raw
comment-count would suggest.

A separate orthographic observation: **only 10.2% (1,423 / 13,902) of
filtered comments contain Kyrgyz-specific Cyrillic letters** (Ң, Ө, Ү);
the remaining 89.8% are written using the Russian keyboard alone,
substituting Н, О, У for the missing letters. This is the dominant
orthographic feature of the YouTube comment register and the
post-hoc explanation for the classical-vs-neural performance gap I
observe in §5.

![Dataset construction pipeline: 13 YouTube videos → 15k raw comments → 11-stage filter → 13.9k clean comments → keyword and random candidate sampling → 1,202 annotation pool.](figures/pipeline_flowchart.png)

### 2.3 Candidate sampling (Davidson 2017 methodology)

Annotating all 13,902 comments is infeasible for a single annotator. Following
Davidson et al. (2017) I apply **biased+random candidate sampling**:

- **Biased pool**: comments matching ≥1 keyword from a curated 5-category
  slur/profanity lexicon (Russian profanity, Kyrgyz profanity, anti-LGBT slurs,
  ethnic slurs, political insults). The lexicon performs Latin→Cyrillic
  normalisation and strips common obfuscation patterns (e.g., `кот.к` →
  `коток`). **702 comments matched (5.0% of the filtered set).**
- **Random pool**: 500 comments randomly sampled from the 13,200 non-matching
  comments. This pool catches slur-free hate the lexicon misses (e.g., ethnic
  essentialism without slurs) and provides genuine `non_hate` content for
  class balance.

Total annotation candidates: **1,202**.

![Keyword-pool composition by lexicon category. Profanity (Kyrgyz and Russian) dominate; ethnic, anti-LGBT, and political-insult categories provide diverse coverage of likely-hate content.](figures/keyword_category_hits.png)

### 2.4 Annotation schema

I adopt the **3-class Davidson schema** with two principled extensions:

- **`hate`**: language targeting people based on a protected attribute
  (ethnicity, nationality, religion, sexual orientation, gender, disability).
- **`offensive`**: profanity, insults, or vulgar expression without
  protected-attribute targeting.
- **`non_hate`**: clean text, neutral or critical without insult or profanity.
- **`skip`**: annotator-uncertainty mark, dropped from gold set.

The two extensions are documented and justified:

1. **Slur-as-generic-insult ⇒ `hate`**: protected-attribute slurs used as
   generic invective (e.g., anti-LGBT slur `пидор` at male politicians,
   gendered slur `жалап` for societal decay) count as `hate` regardless of
   surface target. This follows HateXplain and Facebook Community Standards
   Tier 1.
2. **Incitement-of-violence ⇒ `hate`**: explicit calls for execution/death
   against any identifiable human target count as `hate`. This aligns Davidson
   with Founta (2018), the EU Code of Conduct, and the Council of Europe
   definition.

Three explicit carve-outs from the bare incitement-rule (justified in
Section 3):

a) Dehumanisation without protected-attribute target stays `offensive`.
b) Optative cosmic/religious curses (Turkic curse-formula register) against
   political/non-protected targets stay `offensive`. The Roma case is a
   refinement: optative curses targeting a protected ethnic minority remain
   `hate`.
c) Violent imperatives against criminal-behaviour categories (pedophiles,
   murderers, terrorists, corrupt politicians-framed-as-criminals) stay
   `offensive` when the speech act reads as death-penalty / judicial-execution
   opinion rather than vigilante incitement.

![Annotation pipeline: 1,202 candidates labelled by the author and (independently) by Aya-Expanse-8B; the human label is the gold; Cohen's κ is computed on overlapping items.](figures/annotation_flowchart.png)

### 2.5 Final dataset statistics

A single annotator (me) labelled all 1,202 candidates over
approximately 8 hours of focused work (median 8.6 seconds per item). After
dropping `skip` items, the final labelled set is **1,079 comments**:

| Class | Count | % | Train | Val | Test |
|---|---|---|---|---|---|
| `non_hate` | 472 | 43.7% | 330 | 47 | 95 |
| `offensive` | 343 | 31.8% | 240 | 35 | 68 |
| `hate` | 264 | 24.5% | 185 | 26 | 53 |
| **Total** | **1,079** | 100% | **755** | **108** | **216** |
| Skipped | 123 | 10.2% of candidates | | | |

Stratified 70/10/20 train/val/test split, random seed = 42. The skip rate of
10.2% is itself a corpus characteristic, these were items the annotator could
not confidently label in <5 seconds, typically due to garbled text, ambiguous
target reference, or stance ambiguity (e.g., descriptive femicide statements
that could be either awareness commentary or threat-rhetoric).

**Inter-annotator agreement.** Aya-Expanse-8B was chosen because it
explicitly supports Kyrgyz among its 23 instruction-tuned languages
(CohereForAI, 2024), making it the strongest open-weights multilingual LLM
with native Kyrgyz coverage. It was deployed as a second annotator on all
1,202 candidates using the same Davidson-extended schema prompt (Section 2.4). Cohen's κ on the 1,079 items where the human label was
not `skip` is **κ = 0.100** (raw agreement = 0.361), which falls in the
"slight agreement" range under Landis & Koch (1977). The statistic is

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is the observed proportion of items both annotators labelled
identically and $p_e$ is the expected chance agreement, computed as
$p_e = \sum_{k} P_{\text{human}}(k) \cdot P_{\text{LLM}}(k)$ over the three
classes from the two annotators' marginal label distributions. Substituting
$p_o = 0.361$ and the back-computed $p_e \approx 0.290$ gives
$\kappa \approx 0.100$.

Per-class agreement (human label → LLM-said-same-label):

| Human label | Agreement | n |
|---|---|---|
| `non_hate` | 0.148 | 472 |
| `offensive` | 0.452 | 343 |
| `hate` | 0.625 | 264 |

The LLM agrees with the human most often on `hate`
and almost never on `non_hate`. The full human-vs-LLM confusion matrix
(Figure: `iaa_confusion.png`) shows that **the LLM systematically
over-predicts `hate`**: of the 472 items the human labelled `non_hate`,
the LLM called 219 (46%) `hate` and only 70 (15%) `non_hate`. Of the 343
human-`offensive` items, the LLM called 175 (51%) `hate`. The LLM is
effectively running a 2-class hate-vs-non-hate split heavily biased toward
the positive class, rather than a 3-class Davidson schema.

I interpret this as evidence that **the Davidson schema with my two
extensions and three carve-outs is non-trivial and cannot be reliably
reproduced from a prompt alone**, even by a multilingual LLM that explicitly
supports Kyrgyz. The schema requires the kind of consistent corner-case
adjudication that only a human annotator (or one fine-tuned to the schema)
can provide. This is a methodological finding, not a failure: low-resource
hate-speech work cannot offload annotation to LLMs without losing schema
fidelity.

![Human vs LLM annotator confusion matrix (n=1,079). The
dark-blue band along the LLM-`hate` column reveals the systematic
over-prediction: 46% of human-`non_hate` and 51% of human-`offensive`
items are labelled `hate` by Aya-Expanse-8B.](figures/iaa_confusion.png)

---

## 3. Hate-Speech Sub-Registers in the Corpus

Several culturally-specific sub-registers recur in the corpus. They are
largely absent from US-English benchmark datasets (Davidson, HateXplain,
MetaHate).

### 3.1 Turkic-Islamic curse-formula register

Approximately half of all `offensive`-class comments contain at least one
Turkic-Islamic curse-formula marker. The register has six recurring markers:

- Divine invocations: `Аллахым жазаңарды берсин`, `Кудайа жазаңарды берсин`
- Arabic-origin curse words: `наалат`, `каргыш`
- Prayer closings: `омийин` / `oooмийин`
- Eschatological framing: `эки дүйнөдө` ("in both worlds")
- Praying-hands emoji (🤲) as ritual marker
- Negation-of-Muslim-status framing

Within this register, five sub-patterns are distinguishable: family-scope
curses (`тукумуңар курут болуп`), cosmic curses (`жер жарылып соруп кетсин`),
body-part curses (`мойну үзүлсүн`, `колуңар шал болсун`), exile-burial denial
(`өлсөң сөөгүң келбесин`), and religious imprecations (`АЛЛАХ ЖАЗАЛАЙТ`).

These curses are uniformly in optative grammatical mood, they channel divine
or cosmic punishment rather than commanding human action. My schema labels
them `offensive` despite their content severity (some, like `тукумуңар курут
болуп`, are genocidal in scope). The grammatical-mood discipline keeps the
schema operationally reproducible.

### 3.2 Misogynistic-targeting cluster

A single named female politician (Nadira Narmatova, MP) attracts a
*severity-graded cluster* of misogynistic comments in my queue: from mild
retirement-demand (`Надира кетип калчы`) through gendered-pejoratives
(`Надира жезкемпир` = "old witch") and body-shaming (`корова Надира`) to
explicit elimination imperatives (`Нарматованы жойуш керек` = "Narmatova
must be eliminated", `hate`). This pattern, targeted misogynistic
harassment of a female public figure scaling from mockery to elimination ,
is documented in misogyny scholarship (Amnesty International's *Toxic
Twitter* report) but rarely captured in NLP corpora because most datasets
aggregate by post rather than tracking single-target harassment.

### 3.3 Anti-LGBT / Western-conspiracy stacked rhetoric

Anti-LGBT comments in this corpus consistently link three features: anti-LGBT
attack, ethnic-loyalty failure (`манкурт`, `кыргыз урпагы эмес`), and
Western-influence conspiracy (`американын жугундусу`, `батыштын акчасы`).
This forms the *"LGBT as foreign threat to Kyrgyz identity"* rhetorical
pattern, present globally in conservative/nationalist anti-LGBT discourse
(Russian "traditional values", Polish "LGBT-free zones", US "groomer"
discourse) with culturally specific Kyrgyz vocabulary.

### 3.4 Ethnic-targeting comments

The Kyrgyz-Tajik border conflicts (2021, 2022) produced a sub-corpus of
ethnic-targeted hate that dominates the ethnic-hate slice of this dataset.
Three feature patterns emerge: (i) ethnic essentialism (`Тажиктерден
жакшылык келбейт`); (ii) celebration of mass killing (`200 дой таджиктерди
олтуруптур го`); and (iii) ethnic-collective insult (`Акмактар тажиктер кол
салган`). All three are `hate`-labelled but via different feature-paths.
The *explicit ethnic identifier* in the text, not just contextual
implication, is the discriminator between `hate` and `offensive` in this
cluster.

Although this corpus is dominated by Tajik-targeting comments (a consequence
of when the data was collected relative to the 2021 and 2022 border
conflicts), ethnic hate in the broader Kyrgyz online discourse extends to
Chinese, Uzbek, Kazakh, and Russian targets as well. Hate based on ethnicity
is a common feature of Kyrgyz political comment threads in general.

---

## 4. Experimental Setup

I evaluate eight systems across three families:

| # | Model | Family | Description |
|---|---|---|---|
| 1 | TF-IDF word 1-2gram + LogReg | classical | baseline, minimal preprocessing |
| 2 | TF-IDF word 1-2gram + LogReg | classical | with full preprocessing |
| 3 | TF-IDF char 3-5gram + LogReg | classical | character n-grams |
| 4 | TF-IDF char 3-5gram + Linear SVM | classical | classifier swap |
| 5 | mBERT fine-tuned | transformer | bert-base-multilingual-cased |
| 6 | XLM-RoBERTa-base fine-tuned | transformer | xlm-roberta-base |
| 7 | Aya-Expanse-8B zero-shot | LLM | greedy decoding, max 8 tokens |
| 8 | Aya-Expanse-8B 5-shot/class | LLM | 15 in-context examples |

**Preprocessing profiles.** Two profiles are defined: `minimal` (NFC +
whitespace) and `full` (minimal + lowercase + URL/mention/hashtag strip +
Latin→Cyrillic mapping + punctuation strip + repeat-character collapse).

**Transformer training recipe.** 5 epochs, batch size 16, AdamW with learning
rate 2e-5, weight decay 0.01, warmup ratio 0.1, early stopping on val macro-F1
with patience 2. Same recipe for both mBERT (Devlin et al., 2019) and
XLM-RoBERTa (Conneau et al., 2020) to isolate the model-family effect. I use **bf16** mixed precision rather than fp16:
preliminary runs with fp16 caused XLM-R to collapse to predicting the
majority class (macro-F1 = 0.20 across all eval epochs) due to LayerNorm
overflow; bf16 has the same memory footprint as fp16 but fp32's exponent
range and resolves the instability.

**LLM prompting.** Both zero-shot and few-shot prompts include the full
annotation schema definition. The 5-shot configuration includes 5 examples per
class (15 total) sampled deterministically from the training split.

**Metrics.** Macro-F1 is the primary metric (chosen because hate is the
smallest class and I want all three classes to count equally in evaluation).
I additionally report accuracy, macro-precision, macro-recall, and per-class F1.

**Compute.** Classical experiments run on Apple M3 Pro (CPU). Transformer
fine-tuning and LLM inference run on Cyfronet Helios GH200 (120 GB).

**Reproducibility.** `RNG_SEED=42` is set in every script; classical experiments
are deterministic across runs. Code, data splits, and trained-model outputs
(including all 8 `metrics.json` files and the LLM annotator output) are
available at: https://github.com/adiletbaimyrza/kyrgyz-hsd.

---

## 5. Results

### 5.1 Classical-ML benchmark

The four classical experiments establish baseline performance:

| # | Model | Macro-F1 | Acc | Macro-P | Macro-R |
|---|---|---|---|---|---|
| 1 | TF-IDF word (no preproc) + LogReg | **0.510** | 0.528 | 0.509 | 0.510 |
| 2 | TF-IDF word (full preproc) + LogReg | **0.524** | 0.542 | 0.524 | 0.525 |
| 3 | TF-IDF char 3-5 + LogReg | **0.646** | 0.667 | 0.649 | 0.644 |
| 4 | TF-IDF char 3-5 + Linear SVM | **0.643** | 0.667 | 0.656 | 0.637 |

**RQ1.** Does preprocessing help? Marginally yes: aggressive preprocessing
adds +1.4 F1 points (0.510 → 0.524). The improvement is consistent but small.

**RQ2.** Do character n-grams help on morphologically rich Kyrgyz?
**Strongly yes**: char-ngrams add +12.2 F1 points (0.524 → 0.646). This is
the largest single-decision effect I observe. I attribute this to two
factors. First, Kyrgyz is agglutinative, word-level n-grams treat inflected
forms of the same stem as distinct tokens, fragmenting an already-small
training signal. Character n-grams partially recover stems across inflections.
Second, **89.8% of comments in my corpus are written using the Russian
keyboard rather than Kyrgyz-specific Cyrillic** (Ң ң, Ө ө, Ү ү). Word-level
n-grams treat `жолго тушусун` (Russian-keyboard) and `жолго түшүсүн`
(Kyrgyz-keyboard) as completely different token sequences; char-ngrams
recognise them as nearly identical. The +12 F1 jump is consistent with the
orthographic-prevalence observation.

**RQ3.** Classical ML ceiling. Char-ngram TF-IDF with either LogReg or
Linear SVM reaches ~0.65 macro-F1. The two classifiers are within 0.4 F1
points; the feature representation matters far more than the classifier
choice at this scale.

**Per-class breakdown** (best classical model, Exp 3):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `non_hate` | 0.73 | 0.78 | 0.75 | 95 |
| `offensive` | 0.65 | 0.59 | 0.62 | 68 |
| `hate` | 0.58 | 0.57 | 0.57 | 53 |

The classifier is systematically worst at `hate`, F1 = 0.57, likely because
`hate` is the smallest class (185 training examples) and heterogeneous,
spanning slur-based hate, ethnic-collective insult, incitement of violence,
misogynistic targeting, and anti-LGBT rhetoric (documented in §3). A
bag-of-words model trained on this little data cannot capture the semantic
patterns the `hate` class requires.

### 5.2 Transformer benchmark

| # | Model | Macro-F1 | Acc | F1 non_hate | F1 offensive | F1 hate |
|---|---|---|---|---|---|---|
| 5 | mBERT (bert-base-multilingual-cased) | **0.557** | 0.593 | 0.726 | 0.544 | 0.400 |
| 6 | XLM-RoBERTa-base | **0.431** | 0.514 | 0.684 | 0.420 | 0.188 |

**RQ4.** Do multilingual transformers fine-tuned on ~1k examples outperform
classical TF-IDF? **No.** mBERT (0.557) sits *below* the char-n-gram
baseline (0.646) by 9 F1 points; XLM-RoBERTa (0.431) sits 21 points below.
This contradicts the typical pattern seen on English low-resource
hate-speech tasks where multilingual transformers handily beat TF-IDF
once a few hundred labelled examples are available.

**RQ5.** mBERT vs XLM-R for Kyrgyz? **mBERT > XLM-R by 13 F1 points** on
this dataset, the opposite of what English NLP results would predict
(XLM-R typically wins on multilingual benchmarks). I hypothesise this is
because XLM-R's CommonCrawl-heavy Kyrgyz pretraining over-represents formal
Kyrgyz text using the full Cyrillic alphabet (Ң, Ө, Ү), whereas YouTube
comments overwhelmingly use the Russian-keyboard substitutes (89.8%, §2.2).
mBERT's narrower but Wikipedia-anchored
Kyrgyz pretraining may better match colloquial vocabulary after the
Trainer's standard lowercasing/tokenisation. The model with broader Kyrgyz
coverage performs worse on this dataset because its Kyrgyz pretraining is
on a different orthographic register from the corpus.

Per-class breakdown: XLM-R reaches F1 = 0.19 on the `hate` class,
indicating the model did not learn the minority-class boundary. mBERT
reaches F1 = 0.40 on `hate`, still below the char-n-gram baseline of 0.57.

### 5.3 LLM benchmark

| # | Model | Macro-F1 | Acc | F1 non_hate | F1 offensive | F1 hate |
|---|---|---|---|---|---|---|
| 7 | Aya-Expanse-8B zero-shot | **0.354** | 0.394 | 0.477 | 0.171 | 0.416 |
| 8 | Aya-Expanse-8B 5-shot/class | **0.393** | 0.486 | 0.655 | 0.108 | 0.415 |

**RQ6.** Can a zero-shot LLM substitute for fine-tuning? **No.**
Aya-Expanse-8B zero-shot reaches macro-F1 = 0.354, **29 F1 points below the
classical baseline** and 20 points below mBERT. Aya was selected as the
strongest off-the-shelf zero-shot candidate because it lists Kyrgyz among
its 23 supported languages.

**RQ7.** Does few-shot prompting close the gap? **Only marginally.**
5-shot per class (15 in-context examples drawn from the training split)
improves macro-F1 from 0.354 to 0.393, a +0.04 point gain. The few-shot
prompt does *not* close the gap to fine-tuned transformers, let alone the
classical baseline.

Both LLM configurations show the same prediction pattern: the model
**almost never predicts `offensive`** (recall = 0.103 zero-shot, 0.176
few-shot) and **over-predicts `hate`** (recall = 0.698 zero-shot, 0.585
few-shot). The LLM behaves as a 2-class hate-vs-non_hate classifier rather
than producing the three-class output the schema specifies. This
matches the IAA pattern observed in §2.5 where Aya as second annotator
also over-predicted `hate`. The failure is consistent across the
LLM's two roles (annotator and classifier) and is a feature of the model's
schema-comprehension, not of any specific prompt.

### 5.4 Combined headline

![Macro-F1 across all 8 systems on the test set (n=216).
Char-n-gram TF-IDF + LogReg leads at 0.646; Aya zero-shot trails at 0.354.
Classical (blue) outperforms transformer (orange) outperforms LLM
(red).](figures/results_f1_bar.png)

Reading top to bottom:

1. **Classical char-n-gram TF-IDF + LogReg: 0.646** (winner)
2. Classical char-n-gram TF-IDF + Linear SVM: 0.643
3. mBERT fine-tuned: 0.557
4. Classical word TF-IDF + LogReg with preprocessing: 0.524
5. Classical word TF-IDF + LogReg (baseline): 0.510
6. XLM-RoBERTa fine-tuned: 0.431
7. Aya-Expanse-8B 5-shot: 0.393
8. Aya-Expanse-8B zero-shot: 0.354

**Classical char-n-gram TF-IDF outperforms every neural system tested**,
by margins of 9 F1 (vs mBERT) to 29 F1 (vs Aya zero-shot). The result is
consistent with a **mismatch between the YouTube-comment register of my
corpus (89.8% Russian-keyboard Kyrgyz) and the formal-Kyrgyz pretraining
distributions** of mBERT, XLM-R, and Aya-Expanse. Character n-grams are
orthography-resilient by construction: they treat `жолго тушусун`
(Russian-keyboard) and `жолго түшүсүн` (Kyrgyz-keyboard) as nearly
identical token sequences. Neural systems trained on formal Kyrgyz cannot
match this resilience on a 1k-scale dataset.

The figure also makes a class-family observation visible: **all three LLM
and weak-transformer systems fail predominantly on the `offensive` class**,
while the classical systems are balanced across all three classes. The
hate-detection-as-binary failure mode is a risk when deploying LLMs for
low-resource hate-speech work.

![Per-class F1 across all 8 systems. The LLM and XLM-R systems reach near-zero
F1 on `offensive` (orange bars at far left); the classical systems (right)
are the only ones balanced across all three classes.](figures/per_class_metrics.png)

![Confusion matrices for all 8 systems, sorted by macro-F1. The classical
systems (top row) cluster correct predictions tightly on the diagonal; the
LLM systems (bottom right) have a dark off-diagonal band corresponding to
the over-prediction of `hate`.](figures/confusion_matrices.png)

![Training curves for mBERT (exp5) and XLM-R (exp6). mBERT converges
within 3 epochs to val macro-F1 ≈ 0.66; XLM-R stays near 0.33 (majority-
class baseline) and only weakly learns the minority classes even after
bf16 stabilisation.](figures/training_curves.png)

---

## 6. Discussion

### 6.1 Effective design choices

**Character n-grams over word n-grams** was the single most impactful design
decision in the classical-ML phase, and the post-hoc explanation, Russian-
keyboard orthographic prevalence, generalises to any Cyrillic-Turkic
low-resource language with similar keyboard-input patterns. This is a
finding researchers working on Kazakh, Uzbek, Tatar, and other related
languages should anticipate.

### 6.2 Results contrary to prior expectation

**Classical char-n-grams beat fine-tuned transformers and an 8B LLM on this
task.** My pre-registered expectation (Section 5.2 placeholder before the
runs) was that XLM-R would land around 0.70-0.78 macro-F1 based on
analogous low-resource benchmarks (Bengali, Turkish). The actual XLM-R
result was 0.431, *below* the simplest TF-IDF baseline. mBERT did better
(0.557) but still 9 F1 points below the char-n-gram winner. The 8B
multilingual LLM (Aya-Expanse, marketed as supporting Kyrgyz) performed
worst of all (0.354 zero-shot, 0.393 5-shot). The explanation is the orthographic-prevalence mismatch: the YouTube comment
register in my corpus uses Russian-keyboard Cyrillic (89.8%) while every
neural system tested was pretrained on formal-Kyrgyz text with full
Cyrillic alphabet (Ң, Ө, Ү). Character n-grams are resilient to this
mismatch by construction; subword tokenisers (BPE, SentencePiece) are not.

**The LLM annotator and LLM classifier exhibit the same failure mode.**
Aya-Expanse as second annotator over-predicted `hate` (46% of
human-`non_hate` items labelled `hate`); Aya-Expanse as zero-shot
classifier also over-predicts `hate` (recall = 0.70, only ~10% recall on
`offensive`). The schema is too complex for prompt-only application, and
fine-tuning is necessary to teach the model the three-way distinction.
This consistency across roles suggests the failure is in the model's
schema-understanding, not in any specific prompt, a finding that
generalises to other research that tries to use LLMs both as annotators
and as classifiers on the same multi-class hate-speech taxonomy.

**mBERT outperforms XLM-R on this dataset.** The conventional ordering
(XLM-R > mBERT) holds on most multilingual benchmarks but inverts here. I hypothesise this is because XLM-R's
CommonCrawl-scale Kyrgyz training over-represents formal-Kyrgyz text
with full Cyrillic letters, while mBERT's narrower Wikipedia-anchored
Kyrgyz pretraining is closer to YouTube-comment vocabulary after standard
lowercasing. The model with broader Kyrgyz coverage performs worse on this
dataset because its Kyrgyz pretraining is on a different orthographic
register from the corpus. Future work
should compare with Kyrgyz-specific monolingual models (e.g.
KyrgyzBERT, Mamasaidov & Shopokova 2025) to test whether monolingual
pretraining on YouTube-register Kyrgyz closes the gap to the classical
baseline.

**XLM-R requires bf16 mixed precision on GH200.** My initial XLM-R run
under fp16 collapsed to predicting the majority class (macro-F1 = 0.20
across all eval epochs, training loss stuck at ~1.08). Switching to bf16
fixed the collapse and produced the 0.431 result reported above. This is
a documented issue with XLM-R's LayerNorm under fp16, but it is not widely
discussed in the low-resource NLP literature, practitioners should be
aware that the "free" fp16 speed-up can silently degrade model quality.

### 6.3 Accepted trade-offs

**Single-annotator gold standard.** Time and access constraints prevented
recruiting a second human annotator. LLM-as-second-annotator (Aya-Expanse-8B)
gives a Cohen's κ measurement but cannot fully substitute for human
inter-rater reliability. Future work should add at least one Kyrgyz-speaking
second annotator on a 200-item subset.

**Schema gap: misogyny without lexical markers.** Davidson's lexicon-based
design under-captures content that's misogynistic in worldview but not in
vocabulary. Sweeping anti-women generalisations, harassment normalisation,
and prescriptive religious-conservative gender norms lack the slur/profanity
markers Davidson requires for `offensive`. A future v2 should add a
HatEval-style misogyny flag (Basile et al., 2019) orthogonal to the
hate/offensive axis.

**Schema gap: collective-punishment-via-curse.** The Turkic curse-formula
register includes generational and family-collective curses (`тукумуңар
курут болуп`, `үй-бүлөнүн кордугу балдарына тийсин`) that are content-genocidal
but grammatically optative. My schema labels them `offensive` for operational
reproducibility; Dangerous Speech (Benesch) would label them `hate`. This is
a deliberate trade-off named in §2.4 carve-out (b).

### 6.4 Cross-cultural observations

Several rhetorical patterns in this corpus have no direct analogue in
US-English benchmark datasets:

- **The Turkic-Islamic curse-formula register** is absent from
  HateXplain / Davidson / MetaHate. The 🤲 prayer-hands-emoji + Allah-invocation
  + `омийин` template ritualises the curse as prayer, a distinct speech
  act from secular profanity-driven invective.
- **The Tajik-conflict cluster** of comments is a post-war ethnic-tension
  corpus my work captures incidentally. Analogous data exists for
  Russian-Ukrainian, Armenian-Azerbaijani, Israeli-Palestinian, and other
  conflict pairs but is rarely systematically collected.
- **"LGBT as foreign threat to Kyrgyz identity"** stacks anti-LGBT + ethnic-
  loyalty + Western-conspiracy framing, same global pattern as Russian
  "traditional values" / Polish "LGBT-free zones" / US "groomer" rhetoric,
  with culturally specific Kyrgyz vocabulary.
- **Cursing is a culturally productive register, not a marked style.** A
  substantial fraction of `offensive`-class comments in my corpus consist
  primarily of curse-wishes (`наалат`, `жакшылык көрбөй өтсүн`, `тукум курут
  болуп`, `Аллах жазалайт`, `жер жутсун`) with little or no conventional
  insult vocabulary. The register layers pre-Islamic Turkic imagery (`жер
  жутсун` shamanic register), Arabic-origin Islamic curses (`наалат`,
  `омийин`, `Аллахтын буйругу болсун`), and post-Soviet political vocabulary
  (`шерменде`, `сатылган`) in the same speech acts. This layered register
  reflects Kyrgyz cultural history, centuries of Turkic-Islamic identity
  overlaid with seventy years of Russian/Soviet influence, and is absent
  from English-language hate-speech corpora.

### 6.5 Limitations

(a) **Domain narrowness.** All comments are from YouTube political content.
General-domain hate-speech behaviour may differ.

(b) **Single-annotator gold.** Inter-rater reliability is only available
against the LLM second-annotator.

(c) **Small dataset.** 1,079 labelled comments places `hate` at 185
training examples, likely below the threshold where fine-tuned transformers
reach their full potential.

(d) **Schema gaps documented but not closed.** Misogyny-without-lexical-
markers, collective-punishment-via-curse, and dehumanisation-without-protected
-target all fall into the `offensive` bucket despite content-severity that
modern Dangerous Speech / HatEval frameworks would label `hate`.

(e) **Lexicon coverage gaps.** Emerging phonetic-distortion slurs
(`джапджик`-style mutations of `тажик`) are not in my lexicon and likely
under-recruited.

(f) **Skip rate.** 10.2% of candidates were skipped. These are not random:
they cluster in garbled text, stance-ambiguity (e.g., femicide observation
vs femicide threat), and short-context comments. The dataset's class
distribution is conditional on the skip-rule.

---

## 7. Conclusion

I introduce the first publicly-described Kyrgyz hate-speech
detection dataset (1,079 annotated comments across `hate` / `offensive` /
`non_hate`), and an 8-system benchmark spanning classical ML, fine-tuned
multilingual transformers, and zero-/few-shot LLM prompting.

The **headline empirical finding**: classical character-n-gram TF-IDF +
Logistic Regression (macro-F1 = 0.646) outperforms every neural system I
tested, mBERT (0.557), XLM-RoBERTa
(0.431), and Aya-Expanse-8B both zero-shot (0.354) and 5-shot (0.393).
The classical-vs-neural gap is large (9 F1 over mBERT, 29 F1 over LLM
zero-shot) and consistent with my methodological story: 89.8% of
Kyrgyz YouTube comments use the Russian keyboard rather than the
Kyrgyz-specific Cyrillic alphabet (Ң, Ө, Ү), and the neural systems'
pretraining distributions assume the formal-Cyrillic register. Character
n-grams are orthography-resilient by construction; subword tokenisers are
not. I recommend **char-n-gram TF-IDF + LogReg as a default baseline for
any low-resource Cyrillic NLP task** where comment-style orthography
prevails, not just hate-speech detection.

Beyond the benchmark, the paper contributes:

1. A refined Davidson-based annotation schema with two principled
   extensions (slur-as-generic-insult, incitement-of-violence) and three
   documented carve-outs (dehumanisation-without-protected-target,
   optative-curse-register, criminal-behaviour-target).
2. Seven culturally-specific hate-speech sub-register characterisations
   absent from US-English benchmark corpora.
3. An IAA study (κ = 0.100, n = 1,079) between the author and
   Aya-Expanse-8B as second annotator, with the LLM systematically
   over-predicting `hate`, evidence that multi-class hate-speech schemas
   in low-resource languages cannot be reliably reproduced via prompting.
4. An operational pipeline (`src/run_all.py`, `src/make_figures.py`, the
   SLURM scripts) that reproduces all 8 experiments and 10 figures from
   raw YouTube comments to final paper-ready outputs.

**Recommended future work**:

- A v2 dataset with a HatEval-style misogyny flag orthogonal to the
  hate/offensive axis, target-level subtype labels for cross-cultural
  comparison with HateXplain, and a second human annotator.
- Comparison with KyrgyzBERT (Mamasaidov & Shopokova 2025) to test
  whether monolingual Kyrgyz pretraining closes the classical-vs-neural
  gap.
- An orthographic-normalisation ablation: re-running mBERT/XLM-R/Aya
  after pre-mapping Russian-keyboard substitutes to canonical Kyrgyz
  characters, to test whether the neural underperformance is fully
  explained by the orthographic mismatch.

The dataset, code, trained-model outputs, splits, and a complete
discoveries log are publicly available at
https://github.com/adiletbaimyrza/kyrgyz-hsd.

---

## References

### Hate-speech detection benchmarks and taxonomies

- **Davidson, T., Warmsley, D., Macy, M., & Weber, I.** (2017).
  Automated Hate Speech Detection and the Problem of Offensive Language.
  *ICWSM 2017*. https://arxiv.org/abs/1703.04009
- **Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., &
  Mukherjee, A.** (2021). HateXplain: A Benchmark Dataset for Explainable
  Hate Speech Detection. *AAAI 2021*.
  https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain
- **Founta, A.-M., Djouvas, C., Chatzakou, D., et al.** (2018). Large
  Scale Crowdsourcing and Characterization of Twitter Abusive Behavior.
  *ICWSM 2018*. https://arxiv.org/abs/1802.00393
- **Piot, T., et al.** (2024). MetaHate: A Dataset for Unifying Efforts on
  Hate Speech Detection. https://arxiv.org/html/2401.06526v1
- **Bourgeade, T., et al.** (2024). HateDay: Insights from a Global Hate
  Speech Dataset Representative of a Day on Twitter.
  https://arxiv.org/html/2411.15462v1
- **Basile, V., et al.** (2019). HatEval 2019: Multilingual Detection of
  Hate Speech Against Immigrants and Women in Twitter. *SemEval 2019
  Task 5*. https://aclanthology.org/S19-2007/

### Hate-speech detection in neighbouring Turkic and Central Asian languages

- **Toraman, C., et al.** (2022). A Turkish Hate Speech Dataset and
  Detection System. *LREC 2022*. https://aclanthology.org/2022.lrec-1.443/
- **Yilmaz, A.** (2025). TurkHSD: A Hate Speech Detection Model for
  Turkish Text Content. https://www.researchgate.net/publication/387920916
- **Bekmagambetov, B., et al.** (2021). Cyberbullying and Hate Speech
  Detection on Kazakh-Language Social Networks.
  https://www.researchgate.net/publication/352848223
- **Mamasaidov, R., & Shopokova, A.** (2025). KyrgyzBERT: A Compact,
  Efficient Language Model for Kyrgyz NLP.
  https://arxiv.org/html/2511.20182

### Models used in this benchmark

- **Devlin, J., et al.** (2019). BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding. *NAACL 2019* (mBERT release).
  https://arxiv.org/abs/1810.04805
- **Conneau, A., et al.** (2020). Unsupervised Cross-lingual Representation
  Learning at Scale. *ACL 2020* (XLM-RoBERTa).
  https://arxiv.org/abs/1911.02116
- **CohereForAI** (2024). Aya-Expanse-8B: A multilingual instruction-tuned
  language model. https://huggingface.co/CohereForAI/aya-expanse-8b

### Annotation methodology and policy

- **Landis, J. R., & Koch, G. G.** (1977). The Measurement of Observer
  Agreement for Categorical Data. *Biometrics* 33(1), 159-174.
- **Benesch, S.** (2012). Dangerous Speech: A Proposal to Prevent Group
  Violence. *Dangerous Speech Project*. https://dangerousspeech.org/guide/
- **Council of Europe** (2016). EU Code of Conduct on Countering Illegal
  Hate Speech Online.
  https://commission.europa.eu/strategy-and-policy/policies/justice-and-fundamental-rights/combatting-discrimination/racism-and-xenophobia/eu-code-conduct-countering-illegal-hate-speech-online_en
- **Amnesty International** (2018). *Toxic Twitter: Violence and Abuse
  Against Women Online*.
  https://www.amnesty.org/en/latest/research/2018/03/online-violence-against-women-chapter-1/

### Code and data

- **Project repository:** https://github.com/adiletbaimyrza/kyrgyz-hsd.
  Contains source code, annotated dataset (1,079 labelled comments), data
  splits (70/10/20 stratified), all 8 trained-model outputs
  (`results/exp*/`), regenerable figures (`figures/`), regenerable tables
  (`tables/`), and full run logs (`logs/`).

---

## Appendix A: Annotation guidelines decision tree

```
1. Explicit protected-attribute slur (жалап, пидор, чурка, сарт, жид,
   манкурт-when-ethnic-targeted)?                                  → hate

2. Explicit incitement of violence/death against an identifiable
   protected-attribute OR political-identity target
   (атуу керек, өлгүлө, кырылып кеткиле, ташбаранга алыш керек)?  → hate
   • carve-out: criminal-behaviour target (pedophiles etc.)        → offensive
   • carve-out: dehumanisation without protected target            → offensive
   • carve-out: optative curse against political target            → offensive

3. Ethnic-collective insult applied to a protected ethnic group
   (e.g. акмак тажиктер with explicit тажик identifier)?           → hate

4. Stacking of 2+ protected-attribute attacks
   (e.g. LGBT + ethnic, religion + ethnic-loyalty)?                → hate
                                                                     (high
                                                                      confidence)

5. Eliminationist rhetoric pattern (ethnic-stripping +
   social-separation + harsh-measures imperative)?                 → hate

6. Celebration of mass death of an ethnic group?                    → hate

7. Profanity / insult words / pejorative noun applied to a person,
   OR Turkic-Islamic curse formula?                                → offensive

8. Legal-action language (камалсын, жабыш керек, катуу жаза,
   кууп салыш керек) without insult?                                → non_hate

9. Situational critique without personal attack
   (мунун бары акмакчылык, бул иш уят)?                            → non_hate

10. Conspiracy theory / political opinion without slurs,
    insults, or violence calls?                                    → non_hate

11. Annotator hesitation > 5 seconds?                              → skip
```

---

## Appendix B: Expanded Kyrgyz hate-speech / offensive-language lexicon

While annotating the 1,202-item candidate pool, I logged every
hate-speech and offensive-language token that recurred but was missing
from the seed lexicon used by `find_hate_candidates.py`. The compiled
list (~120 tokens across 9 categories) is a standalone research artifact
in `data/lexicon_expanded.md`. It can be dropped into the seed lexicon
to recruit a higher-recall candidate pool in v2 of this dataset or in
any future Kyrgyz hate-speech work.

The expanded lexicon adds the following categories beyond Davidson's
original five (Russian profanity, Kyrgyz profanity, anti-LGBT, ethnic,
political):

| New category | Tokens added | Purpose |
|---|---|---|
| `violence_imperative` | ~20 phrases, `атыш керек`, `өлтүрүш керек`, `жок кылыш керек`, `ташбаран`, `өлгүлө`, `кырылып кеткиле`, `мойну үзүлсүн`, etc. | Recruits comments triggering the incitement-of-violence rule (Extension 2). |
| `curse_optative` | ~12 phrases, `наалат`, `каргыш`, `тукум курут болуп`, `омийин`, `жер жутсун`, etc. | Recruits the Turkic-Islamic curse-formula register (carve-out (b)). |

The expanded lexicon also extends the existing categories:

- **`profanity_ru`**: +15 tokens, `сучка`, `пздц`, `хуйбаш`, `хуесос`, `пиздабол`, `сволоч`, `стерва`, `канчык`, `далбойоп`, `заебал`, `опкосун`, `похуй`, `хуйня`, `Энен дурайын`, `оозуна сиктим`.
- **`profanity_ky`**: +18 tokens, `айван`/`айбан`, `мал`, `чочко`, `шакал`, `канкор`, `мыкачы`, `сасыган`, `арам`, `манка`, `чимкирик`, `ташак`, `былжырак`, `жинди`, `көкмээ`, `жетим`, `катын`, etc.
- **`lgbt`**: +6 tokens, `пидрлар`, `пидараз`, `пидарас`, `кызтеке`, `гейропа`, `лесбиян`, `транс`.
- **`ethnic`**: +3 tokens, `лөлү` (Roma), `африст` [?], `мырк`.

Each token is annotated in `data/lexicon_expanded.md` with a
transliteration / gloss and a category assignment, with uncertain
glosses explicitly marked `[?]`. The file also includes drop-in Python
code that extends `find_hate_candidates.KEYWORDS_BY_CATEGORY` with the
new tokens.

**Caveats** (also stated in the lexicon file):

1. Categories overlap, many tokens function as both gendered slurs
   and generic profanity depending on usage. The lexicon-membership
   tag is necessary but not sufficient for `hate` vs `offensive`
   classification; the schema decision tree above (Appendix A) is
   what disambiguates.
2. Tokens preserve the **orthographic variation** observed in the
   corpus (89.8% Russian-keyboard substitutes for Ң, Ө, Ү). Both
   forms are attested in YouTube comments and the lexicon includes
   both where applicable.
3. The lexicon is documented for **research purposes only**.
   Inclusion is a methodological contribution to low-resource Kyrgyz
   NLP, not an endorsement of any term.

This is the **first publicly-documented expanded hate-speech /
offensive-language lexicon for Kyrgyz** that I am aware of.
