"""
Keyword-based hate-speech candidate finder.

Methodology follows Davidson et al. (2017), Founta et al. (2018), TurkHSD, and
the Kazakh hate speech paper: a curated slur/profanity lexicon is used to
extract a CANDIDATE POOL of comments that are *likely* to contain hate speech.
Each candidate is later confirmed/rejected by human annotation. The keyword
filter is intentionally noisy (false positives are expected — that is the point
of biased sampling).

Reads:  data/comments_filtered.csv
Writes:
  data/candidates_hate.csv     — comments matching >=1 keyword (for annotation)
  data/candidates_random.csv   — random sample from NON-matching comments (baseline)
  data/keyword_stats.csv       — hit count per keyword and per category
"""

import csv
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict

IN_FILE = "data/comments_filtered.csv"
OUT_HATE = "data/candidates_hate.csv"
OUT_RANDOM = "data/candidates_random.csv"
OUT_STATS = "data/keyword_stats.csv"

RANDOM_SAMPLE_SIZE = 500
RNG_SEED = 42

# ── Lexicon ──────────────────────────────────────────────────────────────────
# Stems only. Patterns allow inflectional endings.

KEYWORDS_BY_CATEGORY = {
    "profanity_ru": [
        "сука", "сучка",
        "блядь", "бля", "биля",
        "мудак", "мудила",
        "дебил",
        "урод",
        "тварь",
        "чмо", "чумо", "чмошник",
        "гнида",
        "пиздец", "пздц", "пизда",
        "хуй", "хуйня", "хуйло",
        "ебать", "ёбан", "ебан", "ёбаный",
    ],
    "profanity_ky": [
        "коток", "котоктор",
        "котокбаш",
        "көтөк", "көтак", "көтәк",
        "эшек",
        "акмак", "акмок",
        "тентек",
        "жалап", "жаляп",
        "олтуру", "өлтүрү", "өлсүн",
    ],
    "lgbt": [
        "пидор", "пидорас", "пидр",
        "педик",
        "гомик", "гомосек",
        "сгейн", "скейн",
        "голубой",
    ],
    "ethnic": [
        "чурка", "чуркестан",
        "хач", "хачи", "хачик",
        "жид", "жидяр", "жидовк",
        "хохол",
        "кыргызня", "киргизня",
        "сарт",
        "узкоглаз",
        "пиндос",
        "манкурт",
    ],
    "political": [
        "предатель",
        "сатылган",
        "продажн",
        "либераст",
        "ватник", "ваты",
    ],
}

# Compound-only patterns: ambiguous short stems that count as hate ONLY in compound form.
# E.g., "кот" alone matches "cat"; "котбаш" / "котокбаш" is the slur.
COMPOUND_PATTERNS = [
    (r"ам\s*жалап", "profanity_ky"),
    (r"ам\s*баш", "profanity_ky"),
    (r"кот\s*баш", "profanity_ky"),
    (r"коту\s*баш", "profanity_ky"),
]

# Latin look-alikes → Cyrillic (common obfuscation)
LATIN_TO_CYRILLIC = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "k": "к", "h": "н", "b": "в",
    "t": "т", "m": "м",
    "A": "а", "E": "е", "O": "о", "P": "р", "C": "с",
    "X": "х", "Y": "у", "K": "к", "H": "н", "B": "в",
    "T": "т", "M": "м",
    "0": "о", "3": "е", "4": "ч",
})

OBFUSCATION_CHARS = re.compile(r"[*_\-.,!?\"']+")
WORD_CHAR = r"[а-яёңөү]"


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(LATIN_TO_CYRILLIC)
    text = OBFUSCATION_CHARS.sub("", text)
    return text


def build_patterns():
    """Compile one regex per keyword: word-start anchor + stem + optional inflectional suffix."""
    compiled = []
    for cat, words in KEYWORDS_BY_CATEGORY.items():
        for w in words:
            pat = re.compile(
                rf"(?<!{WORD_CHAR}){re.escape(w)}{WORD_CHAR}*",
                re.IGNORECASE | re.UNICODE,
            )
            compiled.append((cat, w, pat))
    compound = [(re.compile(p, re.IGNORECASE | re.UNICODE), cat)
                for p, cat in COMPOUND_PATTERNS]
    return compiled, compound


def match_comment(text: str, patterns, compound_patterns):
    """Return (matched_keywords, matched_categories)."""
    normalized = normalize_for_match(text)
    hit_words, hit_cats = [], set()

    for cat, w, pat in patterns:
        if pat.search(normalized):
            hit_words.append(w)
            hit_cats.add(cat)

    for pat, cat in compound_patterns:
        m = pat.search(normalized)
        if m:
            hit_words.append(m.group(0))
            hit_cats.add(cat)

    return hit_words, sorted(hit_cats)


def main():
    if not os.path.exists(IN_FILE):
        raise SystemExit(f"Missing input: {IN_FILE}. Run filter_comments.py first.")

    patterns, compound_patterns = build_patterns()
    random.seed(RNG_SEED)

    candidates, non_matching = [], []
    keyword_hits = Counter()
    category_hits = Counter()

    with open(IN_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matched, cats = match_comment(row["text"], patterns, compound_patterns)
            if matched:
                row_out = dict(row)
                row_out["matched_keywords"] = "|".join(sorted(set(matched)))
                row_out["matched_categories"] = "|".join(cats)
                row_out["label_prior"] = "potentially_hate"
                candidates.append(row_out)
                for kw in set(matched):
                    keyword_hits[kw] += 1
                for c in cats:
                    category_hits[c] += 1
            else:
                non_matching.append(row)

    # ── Random baseline sample ──
    sample_n = min(RANDOM_SAMPLE_SIZE, len(non_matching))
    random_sample = random.sample(non_matching, sample_n)
    for r in random_sample:
        r["matched_keywords"] = ""
        r["matched_categories"] = ""
        r["label_prior"] = "likely_non_hate"

    # ── Write outputs ──
    out_fields = list(candidates[0].keys()) if candidates else \
                 list(random_sample[0].keys()) if random_sample else []

    os.makedirs("data", exist_ok=True)

    with open(OUT_HATE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(candidates)

    with open(OUT_RANDOM, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(random_sample)

    with open(OUT_STATS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "keyword", "hits"])
        for cat, words in KEYWORDS_BY_CATEGORY.items():
            for kw in words:
                w.writerow([cat, kw, keyword_hits.get(kw, 0)])

    # ── Console summary ──
    print(f"Read:           {len(candidates) + len(non_matching)} comments")
    print(f"Candidates:     {len(candidates)} matched ≥1 keyword")
    print(f"Non-matching:   {len(non_matching)}  (random sample: {sample_n})")
    print()
    print("Hits by category:")
    for cat, n in category_hits.most_common():
        print(f"  {cat:<16} {n:>5}")
    print()
    print("Top 15 keywords:")
    for kw, n in keyword_hits.most_common(15):
        print(f"  {kw:<14} {n:>4}")
    print()
    print(f"Files written:")
    print(f"  {OUT_HATE}")
    print(f"  {OUT_RANDOM}")
    print(f"  {OUT_STATS}")
    print()
    print(f"Annotation budget: {len(candidates) + sample_n} comments")


if __name__ == "__main__":
    main()
