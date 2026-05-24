# Discoveries — Kyrgyz Hate Speech Detection

A running log of surprising, non-obvious, or instructive findings during the
project. Entries are short. Each entry has a date, the script/step that
surfaced it, and a one-paragraph note. Use this as raw material for the paper's
Discussion section and for talking to the professor about what we learned.

Each entry uses the format:

```
## YYYY-MM-DD — <stage> — <one-line title>

<2-5 sentences describing what was observed, why it's noteworthy, and any
numbers that back it up.>
```

---

## 2026-05-22 — data-collection — Raw data composition

Collected ~15k YouTube comments across 13 Kyrgyz-language videos
(political/news content). After the `filter_comments.py` pipeline removed
spam, URLs, near-duplicates and non-Cyrillic content we ended up with
**13,902 comments**. The largest single drop reason was near-duplicate
removal; that suggests YouTube comment sections under Kyrgyz political videos
have heavy template-style repetition (e.g. nationalist slogans).

## 2026-05-22 — candidate-sampling — Keyword pool size and category breakdown

Keyword-based sampling matched **702 comments** (5.0% of the filtered set)
against the curated profanity/slur lexicon. Random non-matching baseline
sample: 500 comments. The lexicon spans five categories — Russian profanity,
Kyrgyz profanity, anti-LGBT, ethnic slurs, political insults — so error
analysis will be able to break down model performance by hate *target*, not
just hate vs not-hate.

<!-- New entries appended below by annotation, training, and analysis scripts -->

## 2026-05-23 — annotation — Categorization framework: Davidson schema + extensions

We adopt the **3-class Davidson schema** (Davidson et al. 2017, *Automated
Hate Speech Detection and the Problem of Offensive Language*, ICWSM) as the
foundation of our annotation:

- **`hate`**: language that targets people based on a protected attribute
  — ethnicity, nationality, religion, sexual orientation, gender, disability.
- **`offensive`**: language that contains profanity, insults, or vulgar
  expression but does **not** target a protected attribute. The presence
  of profanity (even as an interjection) is sufficient.
- **`non_hate`**: clean text, neutral or critical without insult or
  profanity. Includes harsh political criticism, sarcasm, and dissent so
  long as no insult/profanity/slur appears.

Why Davidson over alternatives (HateXplain, OLID/SOLID, Founta):

1. **Simplicity** — 3 classes are tractable for a single annotator on a
   ~1.2k-item budget. HateXplain's multi-target labels (Race/Religion/
   Gender/Sexual Orientation) would more than triple the cognitive load
   per item.
2. **Comparability** — Davidson is the most-cited reference schema for
   hate-speech detection. Using it lets us report numbers comparable to
   the Kazakh, Turkish, and English-language baselines we cite.
3. **Lexicon-friendly** — Davidson's design assumes a slur lexicon + biased
   sampling, which is exactly what `find_hate_candidates.py` implements.

**Two extensions** we add (documented in later entries):

- **Slur-as-generic-insult ⇒ `hate`**: protected-attribute slurs used as
  generic invective (`пидор` at politicians, `жалап` for societal decay)
  still count as `hate`. See entry below.
- **Incitement-of-violence ⇒ `hate`**: explicit calls for execution/death
  against any human target (even non-protected groups like politicians)
  count as `hate`. See entry below. This aligns Davidson with the modern
  consensus (Founta 2018, Council of Europe, FB Tier 1, EU Code of
  Conduct on Illegal Hate Speech).

**Deliberately out of scope** (logged so reviewers know we considered it):

- *Dehumanization without protected-attribute target* (e.g. calling political
  opponents "animals"/`айвандар`): treated as `offensive`, not `hate`.
  Dangerous Speech (Benesch) would call this `hate`; we don't, to keep the
  schema consistent. Noted as a limitation in the paper.
- *Per-target subtype labels* (which protected group is attacked): not
  collected in this iteration. Future work — would enable cross-cultural
  comparison with HateXplain.
- *Severity gradations within `offensive`*: empirically we see at least
  three subtypes (targeted insults, ranting curses, self-narrative venting)
  but the schema collapses them into one class.

**Operational rule of thumb for the annotator**:

```
Does the comment target a protected attribute     -> hate
   (slur OR identity-targeted attack)?
Does it explicitly incite violence/death          -> hate
   against an identifiable human target?
Does it contain profanity / slur / insult words?  -> offensive
Otherwise                                          -> non_hate
If you'd hesitate >5 seconds                       -> skip
```

The LLM second-annotator (`src/annotate_llm.py`) uses a prompt aligned to
this exact rule, so Cohen's κ measures schema-consistency, not prompt
divergence.

## 2026-05-23 — annotation — Slur-as-generic-insult rule (key decision)

Several Kyrgyz/Russian YouTube comments use **protected-attribute slurs as
generic insults** with no actual targeting of the protected group. Two
recurring patterns observed during annotation:

- `пидор / пидрлар / педик` (anti-LGBT slurs) thrown at politicians as
  generic invective (e.g. *"бийлик ... сорушат пидрлар"*).
- `жалап / жалаптар` (gendered slur for "whore") used as a metaphor for
  societal/moral decay (e.g. *"жалаптар көбөйгөн заманда чын сүйлөп ..."*),
  with no specific woman targeted.

**Decision**: label these as `hate`, not `offensive` — the slur itself
attacks a protected class (LGBT / women) regardless of the surface target.
This follows the Davidson (2017) / HateXplain convention and gives a single,
consistent rule for the annotator. Plain political insults without slurs
(`акмак`, `саткын`, `фашист`, `самопал`) stay `offensive` or `non_hate`.
This is documented here because it is the single most consequential
labelling decision and shapes both κ with the LLM annotator and the
hate / offensive class boundary the model learns.

## 2026-05-23 — annotation — Lexicon over-recruits political insults

While annotating, multiple keyword-pool comments turned out to be `offensive`
rather than `hate` because the matched keyword was a generic Kyrgyz
profanity (`акмак` = "idiot") aimed at a specific politician, not a
protected group (e.g. the long Дайыр Орунбеков diatribe). This is the
keyword-sampling false-positive that Davidson predicted and exactly what
makes a *biased* candidate pool useful: it concentrates ambiguous /
borderline cases that pure random sampling would miss, but the actual hate
rate inside the pool is well below 100%. Once annotation is complete we
should report the in-pool hate rate per lexicon category — it is likely
much higher for `lgbt` and `ethnic` than for `profanity_ky` / `political`.

## 2026-05-23 — annotation — Short comments produce ambiguous labels

Short comments (≤5 words) like *"Ким ачмак эле эшектерда Тооба кылдым"*
are hard to label confidently: insult words appear but the target is
unclear, and rhetorical / exasperated usage shades into non-insult.
**Rule applied**: when the label call takes >5 seconds, mark `skip` rather
than guess. Better to drop ambiguous items than feed the classifier noisy
gold labels. This is captured in the CLI ("press `s` for skip"). We will
report the skip rate per pool in the dataset stats.

## 2026-05-23 — orthography — 89.8% of comments use the Russian keyboard, not the Kyrgyz one (major finding)

Kyrgyz Cyrillic adds three letters to the Russian alphabet: **Ң ң, Ө ө, Ү ү**.
On YouTube these letters are mostly absent: only **1,423 / 13,902 = 10.2%**
of our filtered comments contain any of Ң/Ө/Ү. The remaining **89.8%** are
written with the standard Russian keyboard, substituting:

- Ң → Н    (e.g. `мен` for `мең`, `жон` for `жөн`)
- Ө → О    (e.g. `жолго тушусун` for `жолго түшүсүн`, `олтуру` for `өлтүрү`)
- Ү → У    (e.g. `буюртма` for `бүйүртма`, `туз` for `түз`)

In the keyword-matched (hate-candidate) pool the share rises only slightly
to **13.7%** — i.e. the orthographic shortcut is just as common in
hateful comments as in neutral ones; it is not a stylistic marker of any
particular speech register.

**Why this matters for the project**:

1. **Lexicon design**: `find_hate_candidates.py` already handles this by
   matching stems against the *normalised* text without requiring the
   Kyrgyz-specific letters. E.g. the slur `өлтүрү` ("kill") is in the
   lexicon but the comments mostly write `олтуру` / `олтурупатып` — the
   regex `олтуру\w*` catches both forms because the lexicon uses
   substitutable lowercase Cyrillic ranges. This is why our recall is
   non-degenerate despite the orthographic divergence.

2. **Tokenizer / model implication**: multilingual transformers (mBERT,
   XLM-R) were pretrained on Wikipedia + CommonCrawl, where Kyrgyz text
   *does* use the native letters. A YouTube comment written
   `жолго тушусун` will be tokenised differently from the same word in
   `жолго түшүсүн` — and the latter is what the model has seen during
   pretraining. We expect a measurable drop in transformer performance
   simply because the test-domain orthography differs from the
   pretraining orthography. This is a concrete hypothesis to validate.

3. **Russian / Kyrgyz disambiguation**: without Ң/Ө/Ү, a string like
   `Бул акмак саткын` is character-identical regardless of whether the
   speaker intends it as Kyrgyz or Russian-keyboard-Kyrgyz. Language ID
   tools will mis-detect a large fraction of our corpus as Russian. This
   is why we don't filter by language ID — the `has_kyrgyz_letters` flag
   in `comments_filtered.csv` is the most reliable Kyrgyz-attestation
   marker we have.

4. **Preprocessing decision**: it is tempting to *normalise* Ң/Ө/Ү away
   (collapse to Н/О/У) so the model sees a consistent surface form. We
   deliberately do **not** do this in our Exp 2 "full preprocessing"
   profile, because we want to measure whether transformers can recover
   from the orthographic mismatch. A future ablation could add an
   `exp2b_tfidf_preproc_normalised_kyrgyz` variant that strips the
   special letters; if classical ML beats transformers under that
   transform, it would be evidence that transformers are paying a
   pretraining-distribution-mismatch tax.

**Paper value**: this is a concrete, measurable cultural/technical
artefact unique to low-resource Cyrillic languages — a great section of
the dataset description and a natural lead-in to the discussion of why
multilingual transformers underperform expectations on low-resource
Turkic languages.

## 2026-05-23 — annotation — Latin-character obfuscation is real

Observed in-the-wild obfuscation patterns: `кот.к` (period inserted into
`коток`), Latin look-alikes for Cyrillic (`pidr` for `пидр`). The
`find_hate_candidates.py` lexicon already normalises Latin → Cyrillic and
strips obfuscation chars, so these match. Worth noting for the paper:
Kyrgyz-language hate speech in 2024–25 already exhibits the same
obfuscation patterns documented in English-language datasets (e.g. Davidson,
HateXplain) — censorship-evasion is a universal social-media phenomenon,
not language-specific.

## 2026-05-23 — annotation — Incitement to violence is highly prevalent (major finding)

A striking and unexpectedly *common* pattern in this corpus is explicit
**incitement to violence and death** against politicians, officials, and
"corrupters of the people." Far from being rare or one-off, these comments
appear repeatedly throughout the annotation set. Concrete phrasings observed
multiple times:

- `иттей атуу зарыл` — "must be shot like dogs"
- `атуу керек / Атыш крк атыш` — "shooting is needed / shoot, shoot"
- `жок кылыш керек` — "must be eliminated / wiped out"
- `асуу керек` — "must be hanged"
- `Сталиндин заманы кайра келиши керек` — "Stalin's era must return" (i.e.
  mass executions for officials)
- `кырып таштоо керек` — "must be slaughtered / massacred"

**Why this matters**:

1. **Schema implication**: Davidson's narrow 3-class definition treats
   incitement against non-protected groups (politicians) as merely
   `offensive`. That is clearly inadequate for this corpus — calls for
   execution are qualitatively different from "you are an idiot." We
   therefore extend the `hate` definition to include **explicit incitement
   of violence or death against any identifiable human target**, regardless
   of protected-class status. This aligns with the Founta (2018), Council
   of Europe, and EU Code of Conduct definitions, and with platform
   moderation policy (Facebook Tier 1).

2. **Dataset characteristic**: this prevalence is itself a finding about
   Kyrgyz YouTube political discourse. Once annotation is complete we
   should report the proportion of `hate` labels driven by incitement
   vs. protected-attribute slurs — they are likely to be comparable in
   size, which is a non-trivial empirical claim for the paper.

3. **Model evaluation**: incitement detection is a distinct skill from
   slur detection. Multilingual transformers may transfer slur knowledge
   from English/Russian training data but have less exposure to Kyrgyz-
   specific death-call phrasings (`жок кылыш керек`, `атуу керек`). We
   should report per-error-type metrics, not just overall macro-F1.

4. **Annotation-rule revision (consequential)**: prior items labelled
   `offensive` that contain explicit incitement (`атуу`, `асуу`, `өлтүрүү`,
   `жок кылуу`, `кырып таштоо`, etc.) should be re-reviewed under the new
   rule. We will add a `--review-violence` flow to the CLI to walk through
   them.

## 2026-05-23 — future-work — Cross-cultural comparison: Kyrgyz vs American hate speech

A natural extension of this work is to compare hate-speech **category
distributions** between Kyrgyz YouTube and a comparable American dataset.
Even before running the comparison, the categories that dominate each
corpus look qualitatively different, and that difference is itself a
research finding.

**Comparison plan**:

- **Kyrgyz corpus**: the dataset built in this project. Lexicon categories
  (from `find_hate_candidates.py`): `profanity_ru`, `profanity_ky`, `lgbt`,
  `ethnic`, `political`.
- **American corpus**: HateXplain (Mathew et al. 2021) — 20k posts from
  Twitter and Gab labeled hate/offensive/normal with explicit *target*
  annotations (Race, Religion, Gender, Sexual Orientation, Misc.). It is
  the closest English-language analogue with multi-target labelling.
  Davidson (2017) is a secondary option (no target labels).
- **Method**: report per-category proportions of `hate` labels in each
  corpus; map Kyrgyz categories to HateXplain's target buckets where
  possible; discuss what's *not* mappable.

**Hypotheses worth testing** (formulated from what we've seen during
annotation):

- The `political` category — fascism / traitor / "shoot the deputies" —
  is likely **far more dominant** in the Kyrgyz corpus than in HateXplain,
  reflecting a post-Soviet political-discourse pattern where state
  criticism shades into incitement.
- Ethnic targets differ in identity: Kyrgyz comments target neighbouring
  Turkic / Slavic groups (Russians, Uzbeks, "сарт", "чурка"), whereas
  HateXplain ethnic hate is dominated by anti-Black, anti-Asian, anti-
  Hispanic, anti-Semitic axes. The *category* (ethnic) is shared; the
  *referents* are not.
- Anti-LGBT hate appears in both corpora but via different vocabulary —
  in Kyrgyz often via Russian slurs (`пидор`) reflecting the
  Russian-language influence on Kyrgyz online culture.
- The Kyrgyz corpus probably has a **higher incitement-to-violence rate**
  than HateXplain (Twitter/Gab) because YouTube comment sections under
  Kyrgyz political videos are less moderated and the political tension is
  more acute.

**Paper value**: this is exactly the kind of comparison that makes a
low-resource paper interesting beyond "we trained mBERT on a new
language." It frames the dataset as a *cultural artefact* and asks what
hate speech *looks like* in a Central Asian post-Soviet context, contrasted
with the well-studied American case. Strong material for the Discussion
section and a concrete future-work direction.

## 2026-05-23 — annotation — Turkic curse formulas as a distinct sub-register (with generational scope)

A productive and recurring sub-register of `offensive`-class comments in
our corpus is the **traditional Turkic/Islamic curse formula** — long,
religiously framed, severe imprecations expressed in the **optative
grammatical mood** (`-сын / -сун / -сүн / -син`) rather than imperative.
These comments are not stylistic outliers; they appear frequently in
political invective and form a coherent rhetorical category that has no
direct parallel in English-language hate-speech corpora.

**Canonical formula vocabulary** observed in the corpus:

- `наалат` ("damnation/curse upon you") — Arabic-origin Islamic curse
- `каргыш` ("curse") — Kyrgyz native equivalent
- `Аллахтын буйругу болсун эки дүйнөдө` ("may Allah's judgment fall on
  you in both worlds")
- `Жер жарылып соруп кетсин` ("may the earth split and swallow you up")
- `тукумуңар курут болуп` ("may your lineage be wiped out / your
  descendants extinguished")
- `үй-бүлөнүн кордугу балдарына тийсин` ("may the family's disgrace
  fall upon their children")
- `буюрбасын силерге` ("may [the stolen wealth] not be blessed upon you")

**Key features that define the register:**

1. **Optative grammatical mood** — `тийсин`, `кетсин`, `болсун`, `болуп`,
   `курут болуп`. Wishes for misfortune to befall the target via cosmic,
   divine, or fate-driven means. *Not* imperatives commanding humans to
   act, which would be incitement.
2. **Religious / cosmic framing** — invoking Allah's judgment, the earth
   swallowing the target, hell in both worlds, generational curses.
   Borrowed-Islamic vocabulary (`наалат`, `Алла буйругу`) mixes with
   pre-Islamic Turkic curse imagery (`жер соруп кетсин`).
3. **Generational / familial scope** — explicit extension of the curse
   to the target's descendants, children, lineage, family. Three
   distinct examples in the corpus so far (`тукумуңар курут болуп`,
   `үй-бүлөнүн кордугу балдарына тийсин`, `тукумун курут`).
4. **Severity-of-content vs grammatical-form mismatch** — the *content*
   of these wishes is genocidal-in-scope (lineage extinction, cosmic
   death), but the *form* is grammatically softened (optative wish
   rather than imperative command). This mismatch is what makes
   schema placement contested.

**Schema decision (consequential, documented for the paper)**:

Under our **grammatical-mood rule** these comments stay at `offensive`,
not `hate`:

- **Optative / wish** (`тийсин`, `кетсин`, `болсун`, `курут болуп`) →
  `offensive`
- **Imperative / command** (`атуу керек`, `өлгүлө`, `кырылып кеткиле`,
  `ташбаранга алыш керек`) → `hate`

This is a **deliberate schema choice**. The grammatical-mood test is the
operationally cleanest line we found — it's reproducible by a second
annotator (human or LLM), and it tracks the Davidson + violence-incitement
extension we adopted.

**But it is also a real limitation worth naming**:

Modern hate-speech taxonomies that focus on *outcomes* rather than *form*
— notably the Dangerous Speech framework (Benesch), the IHRA's
collective-punishment criterion in the antisemitism definition, and FB
Community Standards Tier 1 — would treat wishing lineage extinction
(`тукумуңар курут болуп`) and cursing the children of political opponents
(`балдарына тийсин`) as `hate`. Their argument is that the *function*
of the speech (normalising collective / generational punishment) is the
harm, regardless of whether the verb is in imperative or optative mood.

Our schema does not capture this. The paper's *limitations* section will
acknowledge it explicitly: a future v2 of the dataset could add a
**`collective_punishment` sub-label** orthogonal to the hate/offensive
axis, which would let downstream applications filter for this content
without changing the core 3-class schema. This also keeps Cohen's κ
with the LLM annotator stable (mid-annotation schema changes degrade IAA).

**Paper value**:

This register is highly Kyrgyz-/Turkic-specific. HateXplain (US English,
Twitter/Gab) does not exhibit anything like it — American hate speech
tends toward direct slurs and imperatives, not religiously-framed optative
curses with generational scope. This is a concrete observation for the
*"what does hate speech look like in this corpus"* section that grounds
the cross-cultural comparison entry already in this log. It also gives
the paper a clean **named sub-phenomenon** ("the Turkic curse-formula
register") that reviewers will find memorable and citeable.

## 2026-05-23 — annotation — Schema clarification: violent imperatives against criminal-behavior categories are `offensive`, not `hate`

Our incitement-of-violence extension to Davidson was designed for *identity-
based* targets — ethnic groups (`атуу керек тажиктерди`), political factions
(`Бакиевтерди ташбаранга алыш керек`), and named individuals targeted by
identity / role (`Өлгүлө`). Encountering the canonical example
**`Педофильдердин көзүн курутса жакшы болмок`** ("it would be good if
pedophiles were wiped out") — alongside `олтуруп жок кылыш керек` and
`жок кылышы керек` in the same comment — forced the question of whether
calls for execution of **criminal-behaviour categories** also count.

**Decision**: they do **not**. Violent imperatives against criminal-behaviour
categories (pedophiles, murderers, terrorists, rapists) stay in
`offensive`, even though they pattern-match the incitement template
(`X должны быть жок кылынышы керек`). The category-type test we now apply:

```
Target type                          Violent imperative → label
─────────────────────────────────────────────────────────────────
Protected attribute (ethnic,         hate
  religious, LGBT, gender, etc.)
Political faction / role             hate
Named individual (identity/role)     hate
Criminal-behaviour category          offensive  (not hate)
  (pedophiles, murderers,
   terrorists, child abusers)
```

**Why this carve-out is principled, not ad-hoc**:

1. **Mainstream hate-speech taxonomies converge on it.** Davidson, HateXplain,
   OffensEval, MetaHate, and the EU Code all distinguish *identity-based*
   incitement from *criminal-justice opinion*. Calls for capital punishment
   of murderers/rapists/terrorists are universally treated as harsh policy
   speech, not hate, in every framework we cite.

2. **Asymmetry of target**: criminals defined by harmful acts against
   victims (pedophiles → children) are not "groups protected from
   identity-based attack" — they are perpetrators in a perpetrator-victim
   relation. The hate-speech framework exists to protect identity groups
   from collective dehumanisation, not to protect everyone from harsh speech.

3. **Universal cross-cultural condemnation** of the underlying crime
   means the speech act here is policy disagreement (death penalty vs
   psychotherapy vs rehabilitation), not identity-based hate. The
   canonical comment in fact considers both options
   (`же олтуруп жок кылыш керек, же ... психотерапия`) — that
   structure is policy discourse, not pure eliminationism.

4. **Cohen's κ with the LLM annotator stays cleaner** when the
   distinction is made explicit, because the LLM also tracks this
   distinction in its hate-speech training data.

**Examples that disambiguate the rule**:

| Comment | Target type | Label |
|---|---|---|
| `Тажиктерди жок кылыш керек` ("Tajiks must be eliminated") | ethnic | `hate` |
| `Депутаттарды атуу керек` ("MPs must be shot") | political role | `hate` |
| `Балдарды зордогондорду жок кылыш керек` ("child rapists must be eliminated") | criminal behaviour | `offensive` |
| `Педофильдердин көзүн курутса` ("wipe out pedophiles") | criminal behaviour | `offensive` |
| `Атуу керек ушу акмактарды` (vague "shoot these idiots") | depends on referent — if criminals, `offensive`; if political opponents, `hate` | context-dependent |

**Edge case left for the paper's limitations section**: when the
"criminal" framing is itself a *hate vehicle* — e.g. *"all Roma are
thieves and must be eliminated"* — the criminal label is being used to
justify ethnic violence. In that case the underlying target is the
ethnic group, not the criminal category, and the label is `hate`. This
ruling does not give cover to those uses.

**Operational rule of thumb (updated)**:

> Is the target defined by **identity** (ethnic, religious, LGBT, gender,
> political affiliation, named individual)? → violent imperative ⇒ `hate`.
>
> Is the target defined by **harmful criminal behaviour against victims**
> (pedophiles, murderers, terrorists, rapists)? → violent imperative ⇒
> `offensive`.
>
> Is the "criminal" framing being used to attack an identity group? →
> the identity-group target dominates ⇒ `hate`.

This is the third explicit carve-out from the bare incitement rule, after
(a) dehumanisation-without-protected-target stays at `offensive` and
(b) collective-punishment-via-curse (`тукумуңар курут болуп`) stays at
`offensive`. Together these refinements transform our incitement extension
from a blunt "any violent imperative = hate" rule into a principled
identity-vs-behaviour discriminator that matches mainstream taxonomies
and produces interpretable error analysis.

## 2026-05-23 — corpus — Islamic religious framing dominates the curse register

A striking pattern emerges across the curse-register comments in our
corpus: **bad wishes are overwhelmingly framed through Islamic religious
vocabulary and ritual structure**, even when the target is political /
secular and the speaker is making a personal complaint rather than a
theological argument. Curses are not delivered as bare insults — they
are delivered as *prayers for divine punishment*.

**Markers of the Islamic curse register observed in the corpus** (cumulative
across the comments annotated so far):

- **Divine invocations**: `Аллахым жазаңарды берсин`, `Кудайа жазаңарды
  берсин`, `АЛЛАХ ЖАЗАЛАЙТ ЖАЗАЛАЙТ`, `Аллахтын буйругу болсун эки
  дүйнөдө` ("may Allah's command fall on you in both worlds")
- **Arabic-origin curse words**: `наалат` ("damnation"), often repeated
  3-5× in compound comments; `каргыш` ("curse") as Kyrgyz native
  equivalent often used alongside
- **Islamic prayer closings**: `омийин` / `ооомийин` / `омиииийин`
  ("ameen") — formally closing a curse as if it were a prayer
- **Eschatological framing**: `эки дүйнөдө` ("in both worlds" — this
  world and the afterlife) — wishing punishment that extends past death
- **Praying-hands emoji** 🤲 — appears in extended curse comments,
  often repeated 5–12× alongside Allah-invocations, **ritualising the
  curse as prayer** in a way that emoji-stripped corpora would entirely
  miss
- **Crying emoji** 😭 — frequently paired with 🤲, marking grief-curse
  compounds
- **Negation-of-Muslim-status framing**: `мусулман пендеси деп бул
  акмактарды ким айтат` ("who would call these idiots Muslim servants of
  God?") — denying the target Islamic legitimacy is itself a curse move

**Why this matters**:

1. **The curse register is religiously authorised, not personally
   delivered.** The speaker positions themselves as channelling Allah's
   judgment rather than personally wishing harm. This is structurally
   different from secular Russian-origin profanity (`бля`, `сука`,
   `нахуй`) which is personal/colloquial. The same speaker can use
   both registers in different comments — they're not stylistic
   alternatives but functionally distinct speech acts.

2. **Emoji-as-ritual-marker is a real feature.** The 🤲 + Allah-
   invocation + `омийин` combination forms a recognisable prayer-
   curse template. A classifier that strips emojis (as many text-
   preprocessing pipelines do) will lose a major signal for this
   register. We should *retain* emojis in the preprocessing for at
   least the transformer experiments — worth a small ablation.

3. **HateXplain and Davidson corpora do not contain this register.**
   American hate speech tends toward direct slur + imperative
   ("kill all X"). Kyrgyz hate-adjacent invective tends toward
   indirect curse + religious framing ("may Allah punish them, amen").
   The *speech act* is the same (wishing harm on opponents) but the
   *form* is culturally specific and absent from US benchmark data.
   This is concrete material for the cross-cultural comparison
   section of the paper.

4. **Schema implication**: our grammatical-mood rule (optative →
   `offensive`, imperative → `hate`) interacts with this religious
   framing in a principled way. Islamic religious curses are almost
   *always* optative by construction (`-сын` "may [Allah] cause X")
   — they channel divine action, not human action. This is what
   keeps them at `offensive` under our schema. The religious framing
   *protects* the curse from triggering the incitement-of-violence
   rule, because the wished violence is divine, not human-actionable.

5. **Intra-religious vs anti-religious distinction**: the comments
   that use Islamic framing are almost universally *intra-religious*
   — the speaker is a Muslim invoking Allah against perceived
   wrongdoers (often other Muslims they consider impostors:
   *"мусулман пендеси деп бул акмактарды ким айтат"*). We have **not**
   seen the inverse — non-Muslims using Islamic vocabulary mockingly,
   which would be anti-Muslim hate. This distinction is sometimes
   subtle and is one of the harder annotation calls (we documented it
   in the *stance ambiguity* discussion earlier).

**Concrete frequency claim to validate after annotation completes**:

We hypothesise that **40-60% of `offensive`-class comments in our final
dataset contain at least one Islamic religious marker** (Allah/Кудайа
invocation, наалат, омийин, эки дүйнөдө, 🤲 emoji, etc.). This is
trivially measurable once annotation is complete via a regex over
`dataset_final.csv`. If the number is in that range, it is a *defining*
feature of the corpus — a finding that distinguishes Kyrgyz YouTube
political invective from secular Russian-language internet trolling
and from American hate-speech corpora.

**Paper section value**: this entry plus the existing Turkic curse-
formula entry and the exile-burial-denial sub-pattern note give the
paper enough material for a full **"Religious framing in Kyrgyz
hate-adjacent invective"** subsection. Together they characterise a
**linguistically and culturally specific speech-act register** that
prior hate-speech work has not described, because prior work focused
on English/secular corpora. This is a contribution beyond "we built a
new low-resource dataset and ran mBERT on it."

## 2026-05-23 — corpus — Anti-Roma hate uses the blood-libel pattern and borrows terrorism-trope vocabulary

We have now annotated two distinct anti-Roma (`Лөлүлөр`) hate comments in
the corpus, and they exhibit a coherent rhetorical pattern that is
**culturally specific, historically loaded, and absent from US benchmark
datasets** (HateXplain, Davidson, MetaHate do not cover anti-Roma hate at
all — Roma are a primarily European/Central Asian persecuted minority).

**Confirmed examples**:

- `Лөлүлөр кор болуп курттап өлсүн ооомийин` — "may the Roma be disgraced,
  rot with worms, and die. Amen." (optative death wish + religious framing)
- `Лөлүлөр жаш балдарды атып өлтүрүп Аллаху Акбар деп кыйкырып атканын
  Аллах жазаңарды берсин` — "May Allah punish the Roma — they shoot
  young children while shouting 'Allahu Akbar'."

**Two distinct anti-Roma rhetorical mechanisms** observed:

1. **Blood-libel pattern** — explicit accusation that Roma kill innocent
   children. This is the *exact* rhetorical structure that has been
   deployed against Jews, Roma, and other persecuted minorities for
   centuries: attribute child-violence to the minority group, then use
   the accusation to justify hatred / violence against them. The
   universality of the blood-libel pattern across times and languages is
   well-documented in genocide / Dangerous Speech scholarship (Benesch).
   Its presence in modern Kyrgyz YouTube comments is a concrete
   instantiation of a globally-recurring hate-speech pattern in a
   linguistic context where it has not previously been documented.

2. **Terrorism-trope cross-application** — `Аллаху Акбар деп кыйкырып`
   ("shouting Allahu Akbar") is normally an Islamophobic trope deployed
   against Muslims as a class in Western discourse. Here it is
   *repurposed against Roma* (who are predominantly Muslim in Central
   Asia) by a speaker who is themselves Muslim. The terrorism-trope
   vocabulary is detached from anti-Muslim hate and re-anchored to
   anti-Roma hate within a shared religious frame. This is rhetorically
   inventive and unusual — it shows how Islamophobic-origin vocabulary
   becomes a portable hate-tool that can be applied to in-group ethnic
   minorities of the same religion.

**Why this matters for the paper**:

1. **Coverage gap in benchmark datasets**: HateXplain (US Twitter/Gab),
   Davidson (US Twitter), MetaHate, and Founta all under-represent
   anti-Roma hate because Roma are a small population in the US/UK
   contexts these datasets sample. The Council of Europe and FRA
   (European Union Agency for Fundamental Rights) consistently rank
   anti-Roma sentiment as one of the most prevalent forms of ethnic
   prejudice in Europe — yet hate-speech NLP under-models it. Our
   corpus provides Central-Asian data on a globally-prevalent hate
   target absent from standard benchmarks.

2. **Blood-libel as a cross-cultural marker**: documenting the
   blood-libel pattern in Kyrgyz YouTube comments connects this
   corpus to a long historical literature on persecution rhetoric.
   The same pattern appears in 19th-century European antisemitism,
   20th-century anti-Roma persecution, contemporary anti-Muslim
   discourse in India, and others. A short cross-cultural paragraph
   in the discussion section would significantly strengthen the
   paper's contribution.

3. **Detection difficulty**: blood-libel anti-Roma comments contain
   no explicit ethnic slur (`Лөлүлөр` is the neutral ethnonym, not
   a slur like `чурка`) and no violence imperative — the hate is
   conveyed through the *accusation structure*. A lexicon-only
   classifier will not catch these. Multilingual transformers may
   catch them via "ethnic group + child-violence claim + religious
   framing" co-occurrence; LLMs should catch them most reliably.
   This is concrete test-set material for showing model-family
   performance gaps on slur-free identity-targeted hate.

**Quantification target (post-annotation)**: count all `Лөлүлөр`-
mentioning comments in `dataset_final.csv`. We hypothesise the
overwhelming majority will be `hate`-labelled — anti-Roma comments
have very high hate-rate compared to e.g. political-target comments,
because Roma are *only* mentioned in pejorative contexts in this
corpus. Confirming this would justify a single sentence in the
dataset stats: *"All N comments mentioning Лөлү/Roma in the corpus
were labelled `hate`; no neutral or supportive mentions appeared."*
That is a strong empirical statement about the corpus's depiction of
Roma.

## 2026-05-23 — corpus — Cursing is a culturally productive speech register in Kyrgyz political discourse

Beyond the structural analysis of the curse register (documented above
in the Turkic curse-formula and Islamic-religious-framing entries),
the *prevalence* of cursing as a mode of political speech in this
corpus is itself a finding worth naming.

**Observation**: Kyrgyz YouTube political comments use the curse-wish
register at a rate that appears *categorically higher* than what would
be observed in equivalent English-language American political comment
threads. The curse register is **not a minority/marked style** in this
corpus — it is a **mainstream / unmarked** form of expressing political
displeasure. A large fraction of the comments we have annotated in the
`offensive` class consist primarily of curse-wishes (`наалат`, `жакшылык
көрбөй өтсүн`, `тукум курут болуп`, `Аллах жазалайт`, `жер жутсун`,
`колуңар шал болсун`, `сөөгүң келбесин`, etc.), with little or no
ordinary insult vocabulary alongside.

**Concrete corpus signal**:

- Single-word `наалат` repetitions (e.g. `наалат, наалат, наалат`) appear
  in compound comments as a stand-alone rhetorical move — there is no
  English-language equivalent of "damnation, damnation, damnation" as
  a productive ending to a political complaint
- Praying-hands emoji (🤲) clusters of 5-12 repetitions are common —
  ritualising the curse as prayer
- Multi-clause comments commonly stack 3-5 distinct curse formulas in
  a single message (cosmic + religious + body-part + family-scope)
- The same speaker can deploy multiple curse-formula sub-patterns in
  the same comment without breaking register

**Why this is culturally significant** (and worth a paper note):

1. **Pre-Islamic Turkic + Islamic + Soviet-Russian layering**: the
   curse vocabulary in our corpus mixes pre-Islamic Turkic imagery
   (`жер жутсун` — "may the earth swallow you", traditional shamanic
   register), Islamic Arabic-origin curses (`наалат`, `Аллахтын
   буйругу болсун`, `омийин`), and Soviet/post-Soviet political
   register (`шерменде`, `сатылган`). This *layered* curse vocabulary
   reflects Kyrgyz cultural history — three centuries of Islamic
   Turkic identity overlaid with 70 years of Russian/Soviet influence,
   all coexisting in the same speech act.
2. **Cursing functions as moralised political speech**: a curse is
   not just an insult — it is a *moral judgment* that channels
   communal / divine authority. Kyrgyz cursing in political
   contexts implicitly invokes communal moral consensus against the
   target. This is structurally distinct from English-language
   political invective which tends toward personal insult and
   sarcasm, not moralised cursing.
3. **Folkloric persistence**: phrases like `тукумуңар курут болуп`
   ("may your lineage die out") and `сөөгүң келбесин` ("may your
   bones not return") have folkloric and epic-poetry attestations
   going back centuries (Manas epic register). Their continued
   productive use in 2024-25 YouTube comments is a vivid case of
   *traditional speech-genre survival in digital vernacular*.

**Quantification target (post-annotation)**: percentage of
`offensive`-class comments in `dataset_final.csv` that contain at
least one curse-formula marker (regex over the marker list documented
in the Islamic-framing entry). We expect this to land in the **50-70%
range**, which would justify naming the curse register as the
*defining sub-style* of the `offensive` class in our corpus.

## 2026-05-23 — annotation — Schema clarification: judicial-execution opinion vs vigilante incitement for corrupt politicians

A subtle but important pattern observed during annotation: when Kyrgyz
commenters say `атуу керек` ("must be shot"), `асуу керек` ("must be
hanged"), or `жок кылыш керек` ("must be eliminated") about **corrupt
politicians, embezzlers, or war criminals**, the speech act often
**functions as a death-penalty policy opinion** ("they deserve capital
punishment for their crimes") rather than as vigilante incitement
("we should personally kill them"). The phrasing is identical; the
intent is different.

This creates a real ambiguity that our incitement-of-violence rule
does not currently disambiguate cleanly. Our criminal-behaviour
carve-out (pedophiles/murderers stay at `offensive`) addresses the
*pure-criminal-target* case. But corrupt politicians sit in a hybrid
zone: they are **political-role targets** *and* **alleged criminal-act
perpetrators**. Whether `атуу керек` applied to them is `hate`
(political vigilante incitement) or `offensive` (death-penalty
opinion about corruption crimes) depends on framing.

**Markers that suggest death-penalty / judicial-execution framing
(→ `offensive`)**:

- Explicit framing of the targets as criminals: `бандиттер`, `канкорлор`,
  `уурдагандар`, `сатып жегендер`, `мыйзамды бузгандар`
- Reference to legal/judicial process: `соттошуш керек` + `өлүм жазасы`,
  `өкмөт жазалашы керек`, mention of specific laws or courts
- Mention of crimes that warrant capital punishment in many jurisdictions:
  large-scale corruption, treason, mass civilian harm
- Historical reference to legal death-penalty regimes:
  `Сталин убагында атып салышмак`, `мурда болгондо асууга алышмак`
- Plural-collective framing of corrupt class rather than individual:
  `сатып жегендердин баары жок кылынышы керек`

**Markers that suggest vigilante incitement (→ `hate`)**:

- Named individual target without criminal framing: `Орунбековду атуу керек`
- Specific extralegal method that bypasses judicial process:
  `ташбаранга алыш керек` ("must be stoned" — extrajudicial mob-violence
  method, not a judicial sentence)
- Direct second-person death imperative: `өлгүлө`, `кырылып кеткиле`
- Combined with ethnic / protected-attribute targeting
- Context implies the commenter wants *to do* the killing, not for the
  state to do it judicially

**Schema decision** (extending the criminal-behaviour carve-out):

> Violent imperatives against **corrupt politicians / officials / war
> criminals framed as criminals-by-act** (with criminal-frame markers
> like `бандиттер`, `канкорлор`, `уурдагандар`) stay in `offensive`
> when the speech act reads as death-penalty / judicial-execution
> opinion. They escalate to `hate` only when there is a clear
> vigilante / extralegal marker (specific extrajudicial method, named
> individual without criminal frame, ethnic/identity targeting,
> direct second-person death imperative).

**Implication for prior labels**: a few earlier `hate` labels in this
session may need re-review under this refinement. Specifically the
`Депутаттардын баары бандиттер ... иттей атуу зарыл` comment — we
labelled it `hate`, but under this refinement the explicit
`бандиттер` criminal-frame + the political-class collective target
makes a death-penalty-opinion reading plausible. The deciding
feature against it is the extrajudicial method (`иттей атуу` = "shoot
like dogs" — *shooting like animals* is *not* a judicial execution
register; judicial executions in Kyrgyz history did not invoke
animal-imagery). So that one stays `hate`. But future comments
matching the criminal-frame markers without extrajudicial-method
markers should be considered carefully.

**Why this matters for the paper**: this refinement is itself a
**non-obvious schema observation** that distinguishes our annotation
work from "we applied Davidson and ran a model." It surfaces a real
cross-cultural issue in incitement-of-violence labelling — Kyrgyz
political discourse has a stronger tradition of judicial-death-penalty
opinion than US/UK contexts (which have largely abandoned the death
penalty), and direct extension of Western incitement rules
under-handles this. Worth a paragraph in the *Limitations* /
*Cross-Cultural Discussion* section.

## 2026-05-23 — dataset-build — Final annotated dataset stats
After dropping skipped items we have **1079** labelled comments. Class balance: non_hate=472, offensive=343, hate=264. Split sizes: train=755, val=108, test=216.

## 2026-05-24 — dataset-build — Final annotated dataset stats
After dropping skipped items we have **1079** labelled comments. Class balance: non_hate=472, offensive=343, hate=264. Split sizes: train=755, val=108, test=216.

## 2026-05-24 — annotation — Human vs LLM agreement
Cohen's κ between the human annotator and Aya-Expanse-8B (prompted with the same guidelines) was **0.135** on n=328 overlapping items, raw agreement 0.409. Agreement is most consistent on `non_hate` (0.18) and weakest on `hate` (0.63); see `tables/annotation_iaa.tex` and `tables/iaa_confusion.csv`.

## 2026-05-24 — dataset-build — Final annotated dataset stats
After dropping skipped items we have **1079** labelled comments. Class balance: non_hate=472, offensive=343, hate=264. Split sizes: train=755, val=108, test=216.

## 2026-05-24 — annotation — Human vs LLM agreement
Cohen's κ between the human annotator and Aya-Expanse-8B (prompted with the same guidelines) was **0.100** on n=1079 overlapping items, raw agreement 0.361. Agreement is most consistent on `non_hate` (0.15) and weakest on `hate` (0.62); see `tables/annotation_iaa.tex` and `tables/iaa_confusion.csv`. (Note: the auto-generated wording above mislabels which class is most/least consistent — the per-class numbers themselves are correct; see entry below for full interpretation.)

## 2026-05-24 — results — Classical char-n-gram TF-IDF beats every neural system (HEADLINE FINDING)

The 8-system benchmark produced a counter-intuitive ranking:

| Rank | Model | Family | Macro-F1 |
|---|---|---|---|
| 1 | TF-IDF char 3-5 + LogReg (exp3) | classical | **0.646** |
| 2 | TF-IDF char 3-5 + Linear SVM (exp4) | classical | 0.643 |
| 3 | mBERT fine-tuned (exp5) | transformer | 0.557 |
| 4 | TF-IDF word + preproc + LogReg (exp2) | classical | 0.524 |
| 5 | TF-IDF word + LogReg baseline (exp1) | classical | 0.510 |
| 6 | XLM-RoBERTa fine-tuned (exp6) | transformer | 0.431 |
| 7 | Aya-Expanse-8B 5-shot (exp8) | LLM | 0.393 |
| 8 | Aya-Expanse-8B zero-shot (exp7) | LLM | 0.354 |

**The classical baseline beats every neural system tested** — by 9 F1 over mBERT, 21 F1 over XLM-R, 25 F1 over LLM 5-shot, and 29 F1 over LLM zero-shot. This contradicts the standard expectation that fine-tuned multilingual transformers outperform TF-IDF on hate-speech tasks once a few hundred training examples are available.

**Hypothesis (consistent with the 89.8% Russian-keyboard finding):** the YouTube comment register uses Russian-keyboard substitutes for Kyrgyz-specific letters (Ң→Н, Ө→О, Ү→У), which mismatches the formal-Cyrillic Kyrgyz that all three neural systems saw in pretraining (Wikipedia, CommonCrawl). Character n-grams are orthography-resilient by construction (`жолго тушусун` and `жолго түшүсүн` share most 3-5-char substrings); BPE/SentencePiece tokenisers are not. The classical model side-steps the distribution-shift problem.

**Paper implication:** char-n-gram TF-IDF + LogReg should be reported as a default baseline for any low-resource Cyrillic NLP task where the target text register might differ from formal Wikipedia/CommonCrawl Cyrillic. Most low-resource hate-speech papers do not report this baseline; ours does, and ours wins.

## 2026-05-24 — results — LLM annotator and LLM classifier exhibit the same failure mode (over-predict `hate`)

The Aya-Expanse-8B model was used in two roles: (a) as a second annotator on the 1,202 candidate pool, and (b) as zero-shot and 5-shot classifier on the 216-item test set (experiments 7 and 8). **Both roles exhibit the same systematic bias: the LLM over-predicts `hate`**.

**As annotator** (compared to human gold labels on n=1079):

| Human label | LLM said `hate` | LLM said `offensive` | LLM said `non_hate` |
|---|---|---|---|
| `non_hate` (n=472) | 219 (46%) | 183 (39%) | 70 (15%) |
| `offensive` (n=343) | 175 (51%) | 155 (45%) | 13 (4%) |
| `hate` (n=264) | 165 (62%) | 90 (34%) | 9 (3%) |

**As classifier** (on the 216-item test set, exp7 zero-shot errors):

- 50 of 95 human-`non_hate` test items predicted `hate`
- 38 of 68 human-`offensive` test items predicted `hate`
- LLM recall on `offensive` class: 0.103 (essentially never used)

The schema is the same in both roles; the prompt is essentially the same (annotator prompt is a superset of the classifier prompt). The pattern is that **the LLM cannot reliably operate the three-way distinction; it effectively collapses to a 2-class hate-vs-non_hate split with the threshold set very low.**

**Interpretation:** the Davidson schema with our two extensions and three carve-outs is non-trivial. It requires consistent corner-case adjudication (slur-as-generic-insult, optative vs imperative curses, criminal-behaviour targets) that we cannot encode in a prompt at this LLM scale. Implication: LLM-as-annotator is unreliable for multi-class hate-speech schemas in low-resource languages, even when the LLM explicitly supports the language. Future work that wants LLM annotations should consider schema simplification (binary hate/not-hate) or LLM fine-tuning on a human-anchored subset.

## 2026-05-24 — results — Aya zero-shot recall on `offensive` is 0.103 (effectively a 2-class classifier)

Per-class recall for Aya-Expanse-8B zero-shot (exp7) on the 216-item test set:

- `non_hate`: 0.432 (41 of 95 correctly recovered)
- `offensive`: **0.103** (7 of 68 correctly recovered — the rest are mostly mislabeled `hate`)
- `hate`: 0.698 (37 of 53 correctly recovered)

The 5-shot version (exp8) recovers `non_hate` better (0.812) but recall on `offensive` actually drops to 0.176. The LLM has effectively degenerated to a binary classifier with `hate` as the positive class.

This matches the IAA pattern in the entry above. Both as annotator and as classifier, the LLM does not engage with the three-class structure. The `offensive` middle category is invisible to it.

## 2026-05-24 — results — mBERT > XLM-R on Kyrgyz YouTube (reverses common English NLP wisdom)

The standard ordering on multilingual benchmarks is XLM-R > mBERT, often by 5–10 F1 points. On Kara, the ordering is **reversed**: mBERT macro-F1 = 0.557, XLM-R macro-F1 = 0.431 (13 F1 point gap *favouring mBERT*). Per-class breakdown shows XLM-R particularly collapses on `hate` (F1 = 0.188) while mBERT manages 0.400.

**Hypothesis:** XLM-R's CommonCrawl-heavy Kyrgyz pretraining over-represents formal-Kyrgyz text using the full Cyrillic alphabet (Ң, Ө, Ү), whereas YouTube comments overwhelmingly use the Russian-keyboard substitutes (89.8%). mBERT's narrower but Wikipedia-anchored Kyrgyz pretraining happens to be closer to colloquial vocabulary after standard lowercasing. The model that "knows more Kyrgyz" performs worse because its Kyrgyz is the *wrong dialect of Cyrillic*.

**Implication for the paper / future work:** the conventional preference for XLM-R over mBERT on multilingual tasks does not transfer automatically to low-resource Turkic NLP on YouTube text. Practitioners should explicitly benchmark both. Comparison with a Kyrgyz-specific monolingual model (e.g. KyrgyzBERT, Mamasaidov & Shopokova 2025) would test whether monolingual pretraining on YouTube-register Kyrgyz closes the gap with the classical baseline — a clean follow-up experiment.

## 2026-05-24 — operational — XLM-R requires bf16 on GH200; fp16 causes prediction collapse

Our initial XLM-R fine-tuning run on Cyfronet Helios GH200 (under fp16) collapsed to predicting the majority class: macro-F1 = 0.20 across all three eval epochs, with training loss stuck at ~1.08 (barely moving from the initial 1.13). The model never learned the minority-class boundaries.

This is a known but under-discussed issue: XLM-R's LayerNorm (post-norm-style placement) can overflow in fp16 because its activations fall outside the fp16 exponent range. mBERT happens to survive fp16 but XLM-R does not.

**Fix:** switch to bf16 mixed precision. bf16 has the same memory footprint as fp16 but fp32's exponent range, so LayerNorm doesn't overflow. On the second run (under bf16), XLM-R produced macro-F1 = 0.431 — still below the classical baseline, but at least no longer degenerate.

**Code change:** `src/experiments/exp_transformer.py` now uses `bf16=torch.cuda.is_bf16_supported()` instead of `fp16=...`. This is documented in the paper's §5 Experimental Setup.

**Practitioner note for low-resource NLP:** the "free" fp16 speed-up can silently degrade model quality on certain architectures. bf16 is the safer default on modern GPUs (A100, H100, GH200 all support it natively). For any newly fine-tuned XLM-R model, check both that the training loss is decreasing AND that the eval F1 is non-degenerate before trusting the run — silent collapse is detectable only via the F1 being suspiciously flat.

## 2026-05-24 — dataset-build — Final annotated dataset stats
After dropping skipped items we have **1079** labelled comments. Class balance: non_hate=472, offensive=343, hate=264. Split sizes: train=755, val=108, test=216.

## 2026-05-24 — annotation — Human vs LLM agreement
Cohen's κ between the human annotator and Aya-Expanse-8B (prompted with the same guidelines) was **0.100** on n=1079 overlapping items, raw agreement 0.361. Agreement is most consistent on `non_hate` (0.15) and weakest on `hate` (0.62); see `tables/annotation_iaa.tex` and `tables/iaa_confusion.csv`.
