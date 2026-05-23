# Kara: A Kyrgyz Hate Speech Detection Dataset and Benchmark

**Author:** [Your name]
**Course:** NLP (Master's), [University name]
**Status:** Draft — classical ML results complete; transformer + LLM results pending.

---

## Abstract

Kyrgyz is a low-resource Turkic language with no publicly available hate-speech
detection dataset. We construct **Kara**, a 1,079-comment 3-class annotated
dataset built from Kyrgyz-language YouTube comments, using a Davidson-style
biased+random candidate-sampling pipeline. We benchmark eight systems spanning
classical ML, fine-tuned multilingual transformers, and zero-/few-shot LLM
prompting. The classical-ML benchmark already establishes two findings:
(i) aggressive text preprocessing yields a marginal +1.4 F1 improvement, but
(ii) **character n-grams improve macro-F1 by +12 points over word n-grams**,
quantitatively validating an observation that 89.8% of Kyrgyz YouTube comments
are written using the Russian keyboard rather than Kyrgyz-specific Cyrillic
letters. Character-ngram TF-IDF + Logistic Regression reaches **0.646 macro-F1**
as the classical-ML ceiling. We also document a series of culturally-specific
hate-speech sub-registers (Turkic-Islamic curse formulas, eliminationist
ethnic-loyalty rhetoric, blood-libel anti-Roma framing, intersectional
stacked-slur attacks) absent from US-English benchmark corpora. Beyond the
benchmark, the paper contributes a refined annotation schema with three explicit
carve-outs from Davidson's incitement-of-violence rule. *(Transformer and LLM
results, error analysis, and updated abstract numbers will be added once
Athena-cluster jobs complete.)*

---

## 1. Introduction

Hate speech detection has matured into a well-studied subfield of NLP, but the
field's empirical foundation is heavily concentrated on English-language data
from US-centric platforms — Twitter (Davidson et al., 2017), Reddit/Gab
(Mathew et al., 2021), and aggregated cross-platform corpora (Bourgeade et al.,
2024). Low-resource languages — particularly Turkic and Central Asian languages
— remain underrepresented despite having politically active online communities
where hate-speech detection is operationally relevant.

This paper makes three contributions:

1. **Kara: the first publicly-described Kyrgyz hate-speech dataset.** We
   annotate 1,079 comments from 13 Kyrgyz-language YouTube videos using a
   three-class schema (`hate` / `offensive` / `non_hate`) based on Davidson
   (2017) with explicit extensions and limitations documented.
2. **A classical-ML benchmark establishing the role of orthographic-resilient
   features in low-resource Turkic NLP.** Character n-grams improve macro-F1
   by 12 percentage points over word n-grams, which we attribute to the
   89.8% Russian-keyboard prevalence in the YouTube comment register.
3. **An annotation-methodology contribution**: we identify and document
   culturally-specific hate-speech sub-registers (Turkic-Islamic curse
   formulas, blood-libel anti-Roma framing, intersectional stacked-slur
   attacks, eliminationist ethnic-loyalty rhetoric) that are absent from
   US-centric benchmarks and that require schema refinements not present in
   the strict Davidson framework.

The paper is structured as follows. Section 2 reviews related work. Section 3
describes the dataset construction pipeline. Section 4 presents the annotation
schema and its three carve-outs. Section 5 describes experimental setup;
Section 6 presents results. Sections 7–8 discuss error analysis and broader
implications.

---

## 2. Related Work

**Hate-speech detection benchmarks.** Davidson et al. (2017) established the
canonical 3-class schema (`hate` / `offensive` / `neither`) that we adopt.
HateXplain (Mathew et al., 2021) introduced target-level multi-label
annotation. Founta et al. (2018) added an `abusive` class with explicit
violence-incitement coverage. MetaHate (Piot et al., 2024) aggregated 36
existing English-language datasets. The Council of Europe's *Code of Conduct on
Illegal Hate Speech* and the EU Code define hate speech in policy terms that
include incitement to violence as a constitutive feature.

**Low-resource hate-speech detection.** Recent work has begun documenting hate
speech in under-resourced languages: Romim et al. (2021) on Bengali, Stefanovitch
et al. (2024) on Southeast Asian languages, and Singh et al. (2025) on Indian
languages. The Council-of-Europe-funded *SEAHateCheck* (2024) provides
diagnostic functional tests for hate-speech classifiers in low-resource
languages. Strategies for data-efficient training (Goldzycher et al., 2023)
include cross-lingual transfer and synthetic-data generation.

**Turkic and Central Asian NLP.** Kyrgyz NLP resources are particularly sparse.
Recent surveys (Alimova et al., 2024) identify the absence of hate-speech data.
KyrgyzBERT (Mamasaidov & Shopokova, 2025) provides a monolingual Kyrgyz BERT
pretrained on web-scraped text. Related work on neighboring Turkic languages
includes the Turkish hate-speech dataset of Toraman et al. (LREC 2022),
the TurkHSD model (Yilmaz, 2025), and Kazakh-language cyberbullying detection
(Bekmagambetov et al., 2021).

**Transformer and LLM approaches to hate speech.** Multilingual transformers
(mBERT, XLM-RoBERTa) have become standard for low-resource hate-speech
classification. Recent work has begun evaluating LLMs as zero/few-shot
hate-speech detectors (Zhang et al., 2025; Roy et al., 2025): findings are
mixed, with LLMs strong on slur-based hate and weaker on culturally-contextual
hate. Aya-Expanse (CohereForAI, 2024) is a multilingual instruction-tuned LLM
that explicitly includes Kyrgyz in its supported languages, making it the
natural choice for both LLM-as-annotator and zero/few-shot evaluation in our
benchmark.

---

## 3. Dataset

### 3.1 Source

We collected approximately 15,000 comments from 13 Kyrgyz-language YouTube
videos using `yt-dlp`. The video selection was constrained to
politically-active channels (news, current events, political commentary)
because hate-speech base-rate is materially higher in political comment threads
than in general entertainment content. Channel-level diversity was prioritised:
the 13 videos span four distinct content domains (presidential commentary,
border-conflict reporting, opposition political commentary, social-issue
discussion).

### 3.2 Filtering pipeline

A multi-stage filtering pipeline (`filter_comments.py`) was applied to remove
noise. The 11 filtering stages, in order: (1) empty/whitespace, (2) word-count
< 3 or > 100, (3) letter-ratio < 40%, (4) Cyrillic-letter-ratio < 60%, (5)
excessive-character repeats, (6) within-comment token repetition ≥ 70%, (7)
URL / Telegram-link, (8) phone-number spam, (9) ≥3 @-mentions, (10) ≥3
hashtags, (11) canonical-form near-duplicate. After filtering, **13,902 unique
comments** remain (retention rate ~93%).

Notably, the largest single drop reason after empty/spam filtering was
near-duplicate removal — Kyrgyz political comment threads contain heavy
template-style repetition (often nationalist slogans), and post-deduplication
the corpus contains substantially more linguistic variation than raw
comment-count would suggest.

### 3.3 Candidate sampling (Davidson 2017 methodology)

Annotating all 13,902 comments is infeasible for a single annotator. Following
Davidson et al. (2017) we apply **biased+random candidate sampling**:

- **Biased pool**: comments matching ≥1 keyword from a curated 5-category
  slur/profanity lexicon (Russian profanity, Kyrgyz profanity, anti-LGBT slurs,
  ethnic slurs, political insults). The lexicon performs Latin→Cyrillic
  normalisation and strips common obfuscation patterns (e.g., `кот.к` →
  `коток`). **702 comments matched (5.0% of the filtered set).**
- **Random pool**: 500 comments randomly sampled from the 13,200 non-matching
  comments. This pool catches slur-free hate the lexicon misses (eliminationist
  rhetoric, ethnic essentialism without slurs, blood-libel framing) and
  provides genuine `non_hate` content for class balance.

Total annotation candidates: **1,202**.

### 3.4 Annotation schema

We adopt the **3-class Davidson schema** with two principled extensions:

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
Section 4):

a) Dehumanisation without protected-attribute target stays `offensive`.
b) Optative cosmic/religious curses (Turkic curse-formula register) against
   political/non-protected targets stay `offensive`. The Roma case is a
   refinement: optative curses targeting a protected ethnic minority remain
   `hate`.
c) Violent imperatives against criminal-behaviour categories (pedophiles,
   murderers, terrorists, corrupt politicians-framed-as-criminals) stay
   `offensive` when the speech act reads as death-penalty / judicial-execution
   opinion rather than vigilante incitement.

### 3.5 Final dataset statistics

A single annotator (the author) labelled all 1,202 candidates over
approximately 8 hours of focused work (median 8.6 seconds per item). After
dropping `skip` items, the final labelled set is **1,079 comments**:

| Class | Count | % | Train | Val | Test |
|---|---|---|---|---|---|
| `non_hate` | 472 | 43.7% | 330 | 47 | 95 |
| `offensive` | 343 | 31.8% | 240 | 35 | 68 |
| `hate` | 264 | 24.5% | 185 | 26 | 53 |
| **Total** | **1,079** | 100% | **755** | **108** | **216** |
| Skipped | 123 | 10.2% of candidates | — | — | — |

Stratified 70/10/20 train/val/test split, random seed = 42. The skip rate of
10.2% is itself a corpus characteristic — these were items the annotator could
not confidently label in <5 seconds, typically due to garbled text, ambiguous
target reference, or stance ambiguity (e.g., descriptive femicide statements
that could be either awareness commentary or threat-rhetoric).

**Inter-annotator agreement.** A second annotator (Aya-Expanse-8B) is being
deployed at the time of writing; Cohen's κ will be reported in the final
version. *[TBD: κ value, agreement matrix.]*

---

## 4. Hate-Speech Sub-Registers in the Corpus

We identify several culturally-specific sub-registers that recur in the corpus
and that motivate the three carve-outs from the bare incitement rule. These
sub-registers are paper-worthy because they are largely absent from
US-English benchmark datasets (Davidson, HateXplain, MetaHate).

### 4.1 Turkic-Islamic curse-formula register

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

These curses are uniformly in optative grammatical mood — they channel divine
or cosmic punishment rather than commanding human action. Our schema labels
them `offensive` despite their content severity (some, like `тукумуңар курут
болуп`, are genocidal in scope). The grammatical-mood discipline keeps the
schema operationally reproducible.

### 4.2 Eliminationist ethnic-loyalty rhetoric

A distinct sub-pattern targets perceived ethnic-cultural traitors using the
Aitmatov-derived pejorative `манкурт` (a memory-destroyed slave in Kyrgyz
literary tradition). When this pattern stacks with social-separation imperatives
(`коомдон ажыратыш керек`), ethnic-stripping language (`кыргыз урпагы деп
санабаш керек`), and "harsh measures" imperatives (`катаал чаралар керек`),
it forms the Dangerous Speech pattern (Benesch, 2012) of pre-violence rhetoric.
We label these `hate` via the combined-feature test.

### 4.3 Blood-libel anti-Roma framing

The corpus contains anti-Roma (`Лөлү`) hate comments using two distinct
mechanisms: optative death-wish framing (`Лөлүлөр кор болуп курттап өлсүн
оомийин`) and the historic blood-libel pattern (claiming Roma kill children
while invoking Islamic phrases — `Лөлүлөр жаш балдарды атып өлтүрүп Аллаху
Акбар деп кыйкырып атканын Аллах жазаңарды берсин`). The terrorism-trope
vocabulary (`Аллаху Акбар деп кыйкырып`) is repurposed against Roma by Muslim
speakers — a rhetorical inventiveness worth documenting because it inverts
the typical Islamophobic deployment of the same phrase.

Roma representation in hate-speech NLP is severely underexplored —
HateXplain, Davidson, and MetaHate do not cover anti-Roma hate at all,
despite Roma being one of the most-persecuted European ethnic minorities.
Our corpus provides Central Asian data on a globally-prevalent hate target
absent from standard benchmarks.

### 4.4 Intersectional / stacked-slur attacks

A meaningful subset of `hate` comments stack 2+ protected-attribute attacks
against the same target (e.g., LGBT + ethnic + religion + dehumanisation in a
single comment). Examples observed: `Уйгур зек ЛГБТ президентинер ... сарт
ЛГБТ шпион` (ethnic + LGBT); `Анан, ЛГБТ многобожники манкурты ... итке
теңесе` (LGBT + religion + ethnic-loyalty + dehumanisation). Flat per-target
labelling schemes (HateXplain-style) would under-represent the compound
severity of these comments by recording each attribute independently.

### 4.5 Misogynistic-targeting cluster

A single named female politician (Nadira Narmatova, MP) attracts a
*severity-graded cluster* of misogynistic comments in our queue: from mild
retirement-demand (`Надира кетип калчы`) through gendered-pejoratives
(`Надира жезкемпир` = "old witch") and body-shaming (`корова Надира`) to
explicit elimination imperatives (`Нарматованы жойуш керек` = "Narmatova
must be eliminated" — `hate`). This pattern — targeted misogynistic
harassment of a female public figure scaling from mockery to elimination —
is documented in misogyny scholarship (Amnesty International's *Toxic
Twitter* report) but rarely captured in NLP corpora because most datasets
aggregate by post rather than tracking single-target harassment.

### 4.6 Anti-LGBT / Western-conspiracy stacked rhetoric

Anti-LGBT comments in this corpus consistently link three features: anti-LGBT
attack, ethnic-loyalty failure (`манкурт`, `кыргыз урпагы эмес`), and
Western-influence conspiracy (`американын жугундусу`, `батыштын акчасы`).
This forms the *"LGBT as foreign threat to Kyrgyz identity"* rhetorical
pattern — present globally in conservative/nationalist anti-LGBT discourse
(Russian "traditional values", Polish "LGBT-free zones", US "groomer"
discourse) with culturally specific Kyrgyz vocabulary.

### 4.7 Tajik-conflict comments

The Kyrgyz–Tajik border conflicts (2021, 2022) produced a productive
sub-corpus of ethnic-targeted hate. Three feature patterns emerge:
(i) ethnic essentialism (`Тажиктерден жакшылык келбейт`); (ii) celebration
of mass killing (`200 дой таджиктерди олтуруптур го`); and (iii)
ethnic-collective insult (`Акмактар тажиктер кол салган`). All three are
`hate`-labelled but via different feature-paths. Notably the *explicit
ethnic identifier* in the text — not just contextual implication — is the
discriminator between `hate` and `offensive` in this cluster.

---

## 5. Experimental Setup

We evaluate eight systems across three families:

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
with patience 2. Same recipe for both mBERT and XLM-RoBERTa to isolate the
model-family effect.

**LLM prompting.** Both zero-shot and few-shot prompts include the full
annotation schema definition. The 5-shot configuration includes 5 examples per
class (15 total) sampled deterministically from the training split.

**Metrics.** Macro-F1 is the primary metric (chosen because hate is the
smallest class and we want all three classes to count equally in evaluation).
We additionally report accuracy, macro-precision, macro-recall, and per-class F1.

**Compute.** Classical experiments run on Apple M3 Pro (CPU). Transformer
fine-tuning and LLM inference run on Cyfronet Athena A100 (80 GB).

**Reproducibility.** `RNG_SEED=42` is set in every script; classical experiments
are deterministic across runs. Code: `https://github.com/[TBD]`.

---

## 6. Results

### 6.1 Classical-ML benchmark

The four classical experiments establish baseline performance:

| # | Model | Macro-F1 | Acc | Macro-P | Macro-R |
|---|---|---|---|---|---|
| 1 | TF-IDF word (no preproc) + LogReg | **0.510** | 0.528 | 0.509 | 0.510 |
| 2 | TF-IDF word (full preproc) + LogReg | **0.524** | 0.542 | 0.524 | 0.525 |
| 3 | TF-IDF char 3-5 + LogReg | **0.646** | 0.667 | 0.649 | 0.644 |
| 4 | TF-IDF char 3-5 + Linear SVM | **0.643** | 0.667 | 0.656 | 0.637 |

**RQ1 — Does preprocessing help?** Marginally yes: aggressive preprocessing
adds +1.4 F1 points (0.510 → 0.524). The improvement is consistent but small.

**RQ2 — Do character n-grams help on morphologically rich Kyrgyz?**
**Strongly yes**: char-ngrams add +12.2 F1 points (0.524 → 0.646). This is
the largest single-decision effect we observe. We attribute this to two
factors. First, Kyrgyz is agglutinative — word-level n-grams treat inflected
forms of the same stem as distinct tokens, fragmenting an already-small
training signal. Character n-grams partially recover stems across inflections.
Second, and more importantly, **89.8% of comments in our corpus are written
using the Russian keyboard rather than Kyrgyz-specific Cyrillic** (Ң ң, Ө ө,
Ү ү). Word-level n-grams treat `жолго тушусун` (Russian-keyboard) and
`жолго түшүсүн` (Kyrgyz-keyboard) as completely different token sequences;
char-ngrams recognise them as nearly identical. The +12 F1 jump is
*quantitative validation* of the orthographic-prevalence observation.

**RQ3 — Classical ML ceiling.** Char-ngram TF-IDF with either LogReg or
Linear SVM reaches ~0.65 macro-F1. The two classifiers are within 0.4 F1
points; the feature representation matters far more than the classifier
choice at this scale.

**Per-class breakdown** (best classical model, Exp 3):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `non_hate` | 0.73 | 0.78 | 0.75 | 95 |
| `offensive` | 0.65 | 0.59 | 0.62 | 68 |
| `hate` | 0.58 | 0.57 | 0.57 | 53 |

The classifier is systematically worst at `hate` — F1 = 0.57 — likely because
`hate` is the smallest class (185 training examples) and the most
heterogeneous (five sub-types documented in §4: slur-based, ethnic-collective
insult, incitement-of-violence, eliminationist rhetoric, intersectional
stacked-slur). A bag-of-words model trained on this little data cannot capture
the semantic patterns the `hate` class requires.

### 6.2 Transformer benchmark

*[TBD: mBERT and XLM-R results to be added after Athena jobs complete.
Expected: 0.70–0.78 macro-F1 based on equivalent low-resource benchmarks
(Romim 2021 Bengali; Toraman 2022 Turkish). Particular interest in XLM-R
margin over char-ngrams, which would quantify the value of contextual
multilingual representations on Kyrgyz YouTube text.]*

### 6.3 LLM benchmark

*[TBD: Aya-Expanse-8B zero-shot and 5-shot results to be added. Interest in:
(a) does the LLM close the gap with fine-tuned XLM-R without any training
signal? (b) does 5-shot meaningfully add over zero-shot, or is the prompt
itself sufficient? (c) where does the LLM systematically fail — particularly
on the curse-formula register, intra-religious criticism, and stance-ambiguity
cases documented in §4.]*

### 6.4 Combined headline

*[TBD: full headline figure once all 8 systems are in. Anchor narrative:
classical char-ngrams reach 0.65; XLM-R (expected) ~0.75; LLM few-shot
(expected) somewhere between. The interesting question is the gap-pattern
across the sub-registers from §4.]*

---

## 7. Error Analysis

*[TBD pending best-model selection. Pre-registered error categories from §4
to track:*

- *Slur-bleached generic insult (anti-LGBT slur at non-LGBT target): expected*
  *easy for lexicon, hardest for LLM-stance-aware classifier*
- *Ethnic-collective insult without explicit slur: expected hardest for*
  *TF-IDF, intermediate for transformer, easy for LLM*
- *Curse-formula register comments: expected easy for transformer (token*
  *co-occurrence), hardest for TF-IDF*
- *Misogyny without lexical markers (descriptive sexist generalisation):*
  *systematically missed by lexicon, requires LLM to catch*
- *Intra-religious criticism (defending Islam from extremists): high stance-*
  *ambiguity, expected LLM-only*
- *Eliminationist rhetoric without explicit slur: blood-libel anti-Roma*
  *pattern — measurable gap between lexicon and transformer/LLM*
- *Compound stacked-slur intersectional hate: should be easy for any*
  *competent model — used as confidence sanity check*

*Each category will be quantified with per-category precision/recall once*
*all eight models have run.]*

---

## 8. Discussion

### 8.1 What worked

**Character n-grams over word n-grams** was the single most impactful design
decision in the classical-ML phase, and the post-hoc explanation — Russian-
keyboard orthographic prevalence — generalises to any Cyrillic-Turkic
low-resource language with similar keyboard-input patterns. This is a
finding researchers working on Kazakh, Uzbek, Tatar, and other related
languages should anticipate.

**Davidson + violence-incitement + three carve-outs** as a schema produces
operationally reproducible labels while remaining interpretable. The bare
incitement-extension would over-label criminal-justice opinion and
religious-curse register as `hate`; the bare Davidson would miss eliminationist
rhetoric. The three carve-outs (dehumanisation-without-protected-target,
optative-curse-register, criminal-behaviour-target) discriminate cleanly
between identity-based and behaviour-based incitement, matching modern
hate-speech taxonomies (HatEval 2019, OffensEval, Founta 2018).

### 8.2 What didn't (or what we deliberately accepted)

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
HatEval-style misogyny flag orthogonal to the hate/offensive axis.

**Schema gap: collective-punishment-via-curse.** The Turkic curse-formula
register includes generational and family-collective curses (`тукумуңар
курут болуп`, `үй-бүлөнүн кордугу балдарына тийсин`) that are content-genocidal
but grammatically optative. Our schema labels them `offensive` for operational
reproducibility; Dangerous Speech (Benesch) would label them `hate`. This is
a deliberate trade-off named in §3.4 carve-out (b).

### 8.3 Cross-cultural observations

Several rhetorical patterns in this corpus have no direct analogue in
US-English benchmark datasets:

- **The Turkic-Islamic curse-formula register** is essentially absent from
  HateXplain / Davidson / MetaHate. The 🤲 prayer-hands-emoji + Allah-invocation
  + `омийин` template ritualises the curse as prayer — a distinct speech
  act from secular profanity-driven invective.
- **`Манкурт` as ethnic-loyalty pejorative** (drawn from Aitmatov's
  literary tradition) has no English equivalent. It functions similarly to
  "Uncle Tom" / "race traitor" but with a memory-destruction metaphor that
  encodes specifically post-colonial / post-Soviet identity anxieties.
- **Blood-libel anti-Roma framing** appears in our corpus despite Roma being
  absent from US benchmark target categories. This is direct evidence that
  hate-speech NLP needs European/Central Asian data to cover this target.
- **The Tajik-conflict cluster** of comments is the canonical post-war-
  ethnic-tension corpus our work captures incidentally — analogous data
  would exist for Russian-Ukrainian, Armenian-Azerbaijani, Israeli-Palestinian,
  and other conflict pairs but is rarely systematically collected.
- **"LGBT as foreign threat to Kyrgyz identity"** stacks anti-LGBT + ethnic-
  loyalty + Western-conspiracy framing — same global pattern as Russian
  "traditional values" / Polish "LGBT-free zones" / US "groomer" rhetoric,
  with culturally specific Kyrgyz vocabulary.

### 8.4 Limitations

(a) **Domain narrowness.** All comments are from YouTube political content.
General-domain hate-speech behaviour may differ.

(b) **Single-annotator gold.** Inter-rater reliability is only available
against the LLM second-annotator.

(c) **Small dataset.** 1,079 labelled comments places `hate` at 185
training examples — likely below the threshold where fine-tuned transformers
reach their full potential.

(d) **Schema gaps documented but not closed.** Misogyny-without-lexical-
markers, collective-punishment-via-curse, and dehumanisation-without-protected
-target all fall into the `offensive` bucket despite content-severity that
modern Dangerous Speech / HatEval frameworks would label `hate`.

(e) **Lexicon coverage gaps.** Emerging phonetic-distortion slurs
(`джапджик`-style mutations of `тажик`) are not in our lexicon and likely
under-recruited.

(f) **Skip rate.** 10.2% of candidates were skipped. These are not random:
they cluster in garbled text, stance-ambiguity (e.g., femicide observation
vs femicide threat), and short-context comments. The dataset's class
distribution is conditional on the skip-rule.

---

## 9. Conclusion

We introduce **Kara**, the first publicly-described Kyrgyz hate-speech
detection dataset. The classical-ML benchmark establishes a strong baseline
at 0.646 macro-F1 using character n-grams + Logistic Regression. The
character-vs-word-ngram gap (+12 F1 points) quantitatively validates the
89.8% Russian-keyboard prevalence observation in Kyrgyz YouTube comments.
Beyond the benchmark, the paper contributes a refined Davidson-based
annotation schema with three documented carve-outs (dehumanisation-without-
protected-target, optative-curse-register, criminal-behaviour-target), and
characterises seven culturally-specific hate-speech sub-registers (Turkic-
Islamic curse formulas, eliminationist ethnic-loyalty rhetoric, blood-libel
anti-Roma framing, intersectional stacked-slur attacks, misogynistic-
targeting cluster, anti-LGBT/Western-conspiracy stacking, Tajik-conflict
ethnic targeting) absent from US-English benchmark corpora. The dataset and
annotation framework are designed for extensibility: a v2 should add a
misogyny flag orthogonal to the hate/offensive axis, target-level subtype
labels for cross-cultural comparison with HateXplain, and a second human
annotator. Transformer fine-tuning and LLM zero/few-shot evaluation results
will be added in the final version.

---

## References

*[Final reference list to be assembled from SOURCES.md — citation map already
prepared with 30+ entries across the seven themed groups: Kyrgyz NLP, hate
speech detection surveys, low-resource hate speech, Turkic/Central Asian
languages, transformer methods, LLM evaluation, and annotation methodology.]*

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
