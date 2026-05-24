# Expanded Kyrgyz Hate-Speech / Offensive-Language Lexicon

This is a post-annotation expansion of the seed lexicon used by
`find_hate_candidates.py` to recruit the original 702 keyword-matched
candidates. The vocabulary below was compiled by the annotator while
labelling the 1,202-item candidate pool — it captures terms that
recurred but were either (a) not in the seed lexicon at all, or (b) had
inflectional forms the seed lexicon missed.

**Purpose.** Reusable resource for any future Kyrgyz hate-speech /
offensive-language work. Drop-in expansion for the lexicon-matching
stage of a Davidson-style biased+random sampler.

**Important caveats:**

- Categories overlap. Many tokens function as both gendered slurs and
  generic profanity depending on usage context (see the
  "slur-as-generic-insult" rule in `paper/PAPER.md` §3.4 and the
  schema-clarification entries in `DISCOVERIES.md`).
- These are slurs and vulgar terms documented for **research purposes
  only**. Their inclusion here is a methodological contribution to
  low-resource Kyrgyz NLP, not an endorsement.
- Tokens preserve the orthographic variation observed in the corpus
  (89.8% Russian-keyboard substitutes for Ң, Ө, Ү). Both forms are
  attested in YouTube comments.
- Russian-language slurs are heavily represented because Kyrgyz online
  discourse is bilingual; both languages contribute to the
  hate-speech vocabulary.
- [?] marks terms where the gloss is uncertain; the form is attested in
  the corpus but the meaning may be regional or context-dependent.

---

## 1. Violence-incitement verbs and phrases

These trigger the **incitement-of-violence rule** in our schema (→ `hate`
when target is identifiable, with the criminal-behaviour and
optative-mood carve-outs documented in §3.4 of the paper).

### Imperatives — direct call for harm

| Token | Gloss |
|---|---|
| `атыш керек` / `атуу керек` / `атышкерег` | shooting is needed |
| `атып салыш керек` | should be shot dead |
| `өлтүрүш керек` / `олтуруш керек` | should be killed |
| `жок кылыш керек` | should be eliminated |
| `өрттөө керек` / `орттор керек` / `орттоп салыш керек` / `ортто` | should be burned |
| `кырыш керек` | should be massacred |
| `ташбаран` | stoning |
| `асыш керек` / `илиш керек` | should be hanged |
| `сабаш керек` | should be beaten |
| `союп салса` | if [they were] slaughtered |
| `жардырып алага тебиш керек` | should be blown up |
| `тепкилебейсиңерби` | why don't [you all] kick [them] |
| `КЫЙНАП` | torturing |
| `когуш` | hunt / chase down |

### Imperatives addressed to the target (direct-address violence)

| Token | Gloss |
|---|---|
| `кырылып кеткиле` | perish, [all of you] |
| `өлгүлө` / `олгуло` | die, [all of you] |
| `котуңду айрыйбыз` | we'll tear your ass apart |
| `көтүн айрыш керек` | their ass should be torn apart |
| `жулуп` | tear/pull out |

### Optative-mood death wishes (stay `offensive` per carve-out (b) unless target is a protected attribute)

| Token | Gloss |
|---|---|
| `кырылып олушсун` | may [they] perish |
| `мойну үзүлсүн` / `мойну узулсун` | may [their] neck snap |
| `өлүм сага` / `олум сага` | death to you |
| `чирип өл` / `чирип ол` | rot and die |
| `жер жутсун` (existing) | may the earth swallow [them] |
| `сасыган өлүксүн` | may [they] stink-die [?] |

---

## 2. Generic insults / pejorative nouns

These typically trigger `offensive` when applied to individuals or
unnamed groups. Several can escalate to `hate` when stacked with a
protected-attribute target.

### Animal-based dehumanisation (stays `offensive` per carve-out (a))

| Token | Gloss |
|---|---|
| `айван` / `айбан` / `айвандар` / `айбандар` | animal(s) / beast(s) |
| `мал` | livestock / cattle |
| `чочко` | pig |
| `корова` (Russian) | cow (often gendered against women) |
| `шакал` | jackal |
| `ит` / `иттер` / `иттерди` | dog(s) |
| `крыса` (Russian) | rat |
| `канчыктар` (Russian) | bitches (female dogs, gendered) |
| `сучка` (Russian) | bitch (gendered, diminutive) |
| `тулку` | fox (gendered, "sly woman") |
| `желмогуз` | folkloric monster (gendered, female) |
| `жезкемпир` | folkloric "brass-witch" (gendered, older women) |

### "Brainless" / stupidity insults

| Token | Gloss |
|---|---|
| `акмак` / `акмактар` / `акмакчылык` | idiot / idiots / idiocy |
| `келесоолор` | fools / lazybones |
| `тупойлор` (Russian -лор) | dumb ones |
| `былжырак` | scatterbrained |
| `жинди` | crazy / insane |
| `мээси жок` | brainless ("no brain") |
| `көк мээ` / `кокмээлер` / `көкмээ` | airhead ("blue brain") |
| `далбандар` [?] | dopes [?] |
| `далбаёпсуң` / `далбойоп` (Russian) | dumb-ass (you are) |
| `далбандар` [?] | dumb-asses [?] |
| `дабдырлар` [?] | scatterbrains [?] |
| `тантыбачы` [?] | nonsense-talker [?] |
| `мырк` / `мырктар` | rube / yokel (may carry ethnic-class undertones) |
| `доопороздор` / `доопараз` [?] | (regional pejorative, ~"fool") [?] |
| `мыкачы` / `мыкаачы` | tormentor / murderer |
| `канкорлор` | "blood-shedders" / bloodthirsty murderers |
| `чмо` / `чмолор` / `чумо` (Russian prison slang) | scumbag(s) |
| `лох` (Russian) | sucker / chump |
| `гандон` (Russian) | "condom" — vulgar insult |
| `хуесос` (Russian) | cocksucker |
| `Сволучтар` (Russian) | bastards |
| `стерва` (Russian) | bitch (gendered) |
| `идиоттор` (Russian-influenced) | idiots |
| `катын` | pejorative for "woman" |

### Body / hygiene insults

| Token | Gloss |
|---|---|
| `манка` | snot / mucus (calling someone "snot") |
| `чимкирик` | mucus / boogers |
| `сасыган` | stinking |
| `арам` | unclean / forbidden (Islamic-context insult) |
| `чычкак` [?] | diarrhea / scummy [?] |

### Treacherous / disloyal

| Token | Gloss |
|---|---|
| `саткын` | traitor / sold-out |
| `чыккынчылар` | turncoats / defectors |
| `манкурт` / `манкурттар` (existing) | memory-destroyed slave (ethnic-loyalty pejorative, Aitmatov) |
| `жетим` | orphan (insult to dignity in Kyrgyz culture) |

---

## 3. Russian / Slavic profanity (extends the `profanity_ru` lexicon)

The Kyrgyz online discourse is heavily bilingual; Russian profanity is
ubiquitous. These trigger `offensive` even as exclamations (Davidson
rule).

### Core vulgar interjections

| Token | Gloss |
|---|---|
| `бля` / `блять` / `биля` (existing) | "fuck" (interjection) |
| `сука` / `суки` / `сучка` (partial existing) | bitch / bitches / little-bitch |
| `пздц` / `пиздец` (existing) | "fuck" / disaster |
| `похуй` | "I don't give a fuck" |
| `нахуй` (existing) | "the fuck" |

### `хуй`-family (vulgar penis-related)

| Token | Gloss |
|---|---|
| `хуй` / `хуйня` / `ХУЙНА` (existing) | dick / dick-shit |
| `хуйбаш` | dickhead |
| `хуесос` | cocksucker |

### `пизд`-family (vulgar vagina-related)

| Token | Gloss |
|---|---|
| `пизда` / `опкосун` | cunt |
| `пиздабол` | bullshitter |

### `еб`-family (vulgar fuck-related)

| Token | Gloss |
|---|---|
| `ебан` / `ебандар` / `ёбаный` (existing) | fucker(s) |
| `заебал` | "[you] pissed me off" |
| `сайпал` [?] / `далбойоп` (vulgar Russian-Kyrgyz mix) | get fucked [?] |
| `Энен дурайын` / `энендурайын` | "fuck your mother" |
| `оозуна сиктим` | "[I] fucked your mouth" |
| `котчугун` | "(your) butthole" |
| `сигейн` / `сигип` / `сигил` | obscene Kyrgyz "fuck" cognates |
| `скейн` / `сгн` / `ам` / `ске` | obscene short forms |

### Other vulgar

| Token | Gloss |
|---|---|
| `сволучтар` (Russian) | bastards |

---

## 4. Kyrgyz vulgar profanity (extends `profanity_ky`)

### Core vulgar (penis / butt / vulva)

| Token | Gloss |
|---|---|
| `коток` / `котоктор` / `котокбаш` / `көтөк` / `көтак` / `көтәк` (existing) | dick / dickhead |
| `кот` / `көт` | ass |
| `котчугун` / `көтчүгүн` | "(your) butthole" |
| `ам` | vulva |
| `ташак` / `ташагын` | testicles / "(your) testicles" |

### Kyrgyz obscene exclamations

| Token | Gloss |
|---|---|
| `сигип` / `сигейн` | obscene Kyrgyz "fuck" |
| `Африст` [?] | pejorative [?] |
| `туйруш герек` / `тыйыш керек` [?] | should be muzzled / silenced [?] |

---

## 5. Anti-LGBT slurs (extends `lgbt`)

| Token | Gloss |
|---|---|
| `пидор` / `пидорас` / `пидр` (existing) | f*ggot / f*ggot-ass |
| `пидрлар` | f*ggots |
| `пидараз` / `пидарас` | f*ggot-ass |
| `педик` (existing) | f*g |
| `гомик` / `гомосек` (existing) | "homo" |
| `голубой` (existing) | "blue" (slur for gay men) |
| `гей` | gay (descriptive; pejorative when used as slur) |
| `транс` | trans (descriptive; pejorative when used as slur) |
| `лесбиянкалар` | lesbians (pejorative in slur context) |
| `кызтекелер` | "girl-billy-goats" — pejorative for non-feminine women |
| `гейропа` | "gay-Europe" (anti-European, anti-LGBT compound slur) |

---

## 6. Gendered / misogynistic slurs (women-targeted)

| Token | Gloss |
|---|---|
| `жалап` / `жалаптар` / `жаляп` (existing) | whore(s) |
| `бузуку` | corrupted / loose woman |
| `сучка` (Russian, gendered diminutive of `сука`) | bitch (female-specific) |
| `канчыктар` (Russian) | bitches (female-specific) |
| `тулку` | fox (gendered, manipulative woman) |
| `желмогуз` | folkloric monster (typically female) |
| `жезкемпир` | "brass-witch" / hag (older women) |
| `корова` (Russian) | cow (gendered body-shaming) |
| `стерва` (Russian) | bitch / shrew (female-specific) |
| `катын` | woman (often pejorative) |
| `ээнбаштар` | freeloaders / hangers-on (often gendered female) |
| `тантыбачы` [?] | nonsense-talker [?] (often female-coded) |
| `кызтекелер` | "girl-billy-goats" — non-feminine women |
| `Лох` (general; also gendered) | sucker / fool |

---

## 7. Ethnic slurs (extends `ethnic`)

| Token | Gloss |
|---|---|
| `чурка` / `чурки` (existing) | derogatory term for Central Asians (used by Russians; also reclaimed-pejoratively in self-attack) |
| `сарт` (existing) | historic pejorative for Uzbeks / sedentary Turkic |
| `хач` / `хачи` / `хачик` (existing) | derogatory for Armenians / Caucasians |
| `жид` / `жидяр` / `жидовк` (existing) | anti-Jewish slur |
| `хохол` (existing) | derogatory for Ukrainians |
| `кыргызня` / `киргизня` (existing) | derogatory for Kyrgyz |
| `узкоглаз` (existing) | "narrow-eyed" — anti-Asian |
| `пиндос` (existing) | derogatory for Americans |
| `манкурт` (existing) | ethnic-loyalty pejorative |
| `Лөлү` / `Лөлүлөр` | Roma (often used pejoratively in Central Asia) |
| `Африст` [?] | African [?] (likely anti-Black) |
| `мырк` / `мырктар` | rube (may carry ethnic-class undertones) |

---

## 8. Religious / Islamic slurs and curses

### Curses (Arabic-origin, Islamic register)

| Token | Gloss |
|---|---|
| `наалат` | damnation |
| `каргыш` | curse |
| `кор болуп` | may [they] be disgraced |
| `жакшылык көрбөсүн` / `жакшылык корбосун` | may [they] not see good |
| `Аллахтын буйругу болсун` | may Allah's will fall on [them] |
| `жер жутсун` | may the earth swallow [them] |
| `омийин` / `оомийин` / `омиииийин` | "amen" (closes a curse-as-prayer) |
| `тукум курут болуп` | may [their] lineage die out |
| `Аллах жазалайт` | Allah will punish |

### Religion-as-insult

| Token | Gloss |
|---|---|
| `шайтан` / `шайтандар` | devil(s) / satan |
| `арам` | haram / unclean (used as insult) |
| `ит` / `иттер` | dog(s) (unclean animal in Islamic context) |

---

## 9. Cross-cutting compound attacks (often stack 2+ categories)

Many comments in our corpus stack multiple slur categories — e.g.
`Уйгур зек ЛГБТ` (ethnic + LGBT) or `жалап-манкурт` (gendered + ethnic-
loyalty). The intersectional-hate pattern is documented as a paper
finding in `paper/PAPER.md` §4.4 and is one of the highest-confidence
`hate`-class signals.

---

## How to use this lexicon programmatically

The seed lexicon in `find_hate_candidates.py` is a Python dict
(`KEYWORDS_BY_CATEGORY`). To extend it with the tokens above:

```python
KEYWORDS_BY_CATEGORY["profanity_ru"].extend([
    "сучка", "пздц", "хуйбаш", "хуесос", "пиздабол",
    "сволоч", "стерва", "канчык", "далбойоп", "заебал",
    "опкосун", "ёбаный", "пизда", "похуй",
])

KEYWORDS_BY_CATEGORY["profanity_ky"].extend([
    "айван", "айбан", "мал", "чочко", "шакал", "канкор",
    "мыкачы", "сасыган", "арам", "манка", "чимкирик",
    "ташак", "былжырак", "жинди", "көкмээ", "келесоо",
    "жетим", "катын", "тулку", "желмогуз", "ээнбаш",
    "тантыба",
])

KEYWORDS_BY_CATEGORY["lgbt"].extend([
    "пидрлар", "пидараз", "пидарас",
    "кызтеке", "гейропа",
    "лесбиян", "транс",
])

KEYWORDS_BY_CATEGORY["ethnic"].extend([
    "лөлү", "африст", "мырк",
])

# New categories that emerged from annotation:

KEYWORDS_BY_CATEGORY["violence_imperative"] = [
    "атыш керек", "атуу керек", "атып салыш керек",
    "өлтүрүш керек", "олтуруш керек",
    "жок кылыш керек", "өрттөө керек",
    "кырыш керек", "ташбаран", "асыш керек", "илиш керек",
    "сабаш керек", "союп салса", "тепкилеп",
    "көтүн айрыш керек", "көтүңду айрыйбыз",
    "мойну үзүлсүн", "өлгүлө", "кырылып кеткиле",
    "кырылып олушсун",
]

KEYWORDS_BY_CATEGORY["curse_optative"] = [
    "наалат", "каргыш", "омийин", "оомийин",
    "Аллахтын буйругу", "Аллах жазалайт",
    "тукум курут болуп", "жер жутсун",
    "жакшылык көрбөсүн", "кор болуп",
    "чирип өл", "өлүм сага",
]
```

Re-run `python3 -m find_hate_candidates` to recruit a fresh candidate
pool with the expanded lexicon. Expect significantly higher recall on
violence-incitement and curse-formula comments (categories the original
lexicon under-recruited).
