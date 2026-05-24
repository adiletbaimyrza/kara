"""Research-grade filtering pipeline for Kyrgyz YouTube comments."""

import csv
import glob
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

RAW_DIR = "data/raw"
OUT_FILE = "data/comments_filtered.csv"
STATS_FILE = "data/filter_stats.csv"
REJECTED_FILE = "data/comments_rejected.csv"
REJECTED_SAMPLE_PER_REASON = 30
RNG_SEED = 42

KYRGYZ_SPECIFIC = set("ңөүҢӨҮ")
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
TELEGRAM_RE = re.compile(r"(t\.me/|telegram\.me/|wa\.me/)\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)
REPEAT_CHAR_RE = re.compile(r"([^\W\d_])\1{4,}", re.IGNORECASE | re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC + decode HTML-ish artifacts + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    text = WS_RE.sub(" ", text).strip()
    return text


def is_cyrillic_letter(c: str) -> bool:
    return "Ѐ" <= c <= "ӿ" and c.isalpha()


def letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isalpha()) / len(chars)


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if is_cyrillic_letter(c)) / len(letters)


def word_count(text: str) -> int:
    return len(text.split())


def has_excessive_repeats(text: str) -> bool:
    return bool(REPEAT_CHAR_RE.search(text))


def is_token_repetition(text: str) -> bool:
    tokens = [t.lower() for t in text.split() if t.strip()]
    if len(tokens) < 3:
        return False
    most_common = Counter(tokens).most_common(1)[0][1]
    return most_common / len(tokens) >= 0.7


def is_url_or_link(text: str) -> bool:
    return bool(URL_RE.search(text) or TELEGRAM_RE.search(text))


def is_phone_spam(text: str) -> bool:
    return bool(PHONE_RE.search(text))


def is_mention_spam(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    mentions = MENTION_RE.findall(text)
    return len(mentions) >= 3 or len(mentions) / len(tokens) > 0.4


def is_hashtag_spam(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    hashtags = HASHTAG_RE.findall(text)
    return len(hashtags) >= 3 or len(hashtags) / len(tokens) > 0.4


def has_kyrgyz_letters(text: str) -> bool:
    return any(c in KYRGYZ_SPECIFIC for c in text)


def canonical_form(text: str) -> str:
    """Used as dedup key. Catches: 'Жакшы!', 'жакшыыыы', 'Жакшы 👍👍'."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(c for c in text if c.isalpha())
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


# ── Filter pipeline ───────────────────────────────────────────────────────────

FILTER_STAGES = [
    ("empty",            lambda t: not t),
    ("too_short",        lambda t: word_count(t) < 3),
    ("too_long",         lambda t: word_count(t) > 100),
    ("low_letter_ratio", lambda t: letter_ratio(t) < 0.40),
    ("non_cyrillic",     lambda t: cyrillic_ratio(t) < 0.60),
    ("char_repeats",     has_excessive_repeats),
    ("token_repetition", is_token_repetition),
    ("url_link",         is_url_or_link),
    ("phone",            is_phone_spam),
    ("mention_spam",     is_mention_spam),
    ("hashtag_spam",     is_hashtag_spam),
]


def filter_comment(text: str):
    """Returns (kept, drop_reason). drop_reason is None when kept."""
    for name, predicate in FILTER_STAGES:
        try:
            if predicate(text):
                return False, name
        except Exception:
            return False, name
    return True, None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stage_counts = defaultdict(lambda: defaultdict(int))  # video_id -> stage -> count
    rejected_samples = defaultdict(list)                   # stage -> [(video, text), ...]
    kept_per_video = defaultdict(int)
    raw_per_video = defaultdict(int)

    canonical_seen = {}  # canonical -> first comment id (for tracking dedup origin)
    all_kept = []

    files = sorted(glob.glob(f"{RAW_DIR}/*.json"))
    if not files:
        print(f"No JSON files in {RAW_DIR}/")
        return

    for path in files:
        video_id = os.path.basename(path).replace(".info.json", "")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        for c in data.get("comments", []):
            raw_per_video[video_id] += 1
            text = normalize(c.get("text", ""))

            kept, reason = filter_comment(text)
            if not kept:
                stage_counts[video_id][reason] += 1
                if len(rejected_samples[reason]) < REJECTED_SAMPLE_PER_REASON:
                    rejected_samples[reason].append((video_id, text[:200]))
                continue

            canon = canonical_form(text)
            if not canon or canon in canonical_seen:
                stage_counts[video_id]["duplicate"] += 1
                if len(rejected_samples["duplicate"]) < REJECTED_SAMPLE_PER_REASON:
                    rejected_samples["duplicate"].append((video_id, text[:200]))
                continue
            canonical_seen[canon] = c.get("id", "")

            all_kept.append({
                "id": c.get("id", ""),
                "video_id": video_id,
                "text": text,
                "likes": c.get("like_count", 0) or 0,
                "date": c.get("timestamp", "") or "",
                "n_words": word_count(text),
                "has_kyrgyz_letters": int(has_kyrgyz_letters(text)),
                "cyrillic_ratio": round(cyrillic_ratio(text), 3),
                "label": "",
            })
            kept_per_video[video_id] += 1

    # ── Write outputs ──
    os.makedirs("data", exist_ok=True)

    fields = ["id", "video_id", "text", "likes", "date",
              "n_words", "has_kyrgyz_letters", "cyrillic_ratio", "label"]
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_kept)

    all_reasons = [s for s, _ in FILTER_STAGES] + ["duplicate"]
    with open(STATS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "raw", "kept"] + all_reasons)
        for vid in sorted(raw_per_video):
            row = [vid, raw_per_video[vid], kept_per_video[vid]]
            row += [stage_counts[vid].get(r, 0) for r in all_reasons]
            w.writerow(row)
        total = ["TOTAL", sum(raw_per_video.values()), sum(kept_per_video.values())]
        total += [sum(stage_counts[v].get(r, 0) for v in raw_per_video) for r in all_reasons]
        w.writerow(total)

    with open(REJECTED_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reason", "video_id", "text"])
        for reason in all_reasons:
            for vid, text in rejected_samples[reason]:
                w.writerow([reason, vid, text])

    # ── Console summary ──
    print(f"\n{'Video':<14} {'Raw':>6} {'Kept':>6} {'Drop%':>6}")
    print("-" * 40)
    for vid in sorted(raw_per_video):
        r, k = raw_per_video[vid], kept_per_video[vid]
        pct = (1 - k / r) * 100 if r else 0
        print(f"{vid:<14} {r:>6} {k:>6} {pct:>5.1f}%")

    total_raw = sum(raw_per_video.values())
    total_kept = sum(kept_per_video.values())
    print("-" * 40)
    print(f"{'TOTAL':<14} {total_raw:>6} {total_kept:>6} {(1-total_kept/total_raw)*100:>5.1f}%")

    print(f"\nDrops by reason:")
    totals = Counter()
    for vid in stage_counts:
        for reason, n in stage_counts[vid].items():
            totals[reason] += n
    for reason in all_reasons:
        if totals[reason]:
            print(f"  {reason:<22} {totals[reason]:>6}")

    kept_with_kyr = sum(1 for c in all_kept if c["has_kyrgyz_letters"])
    print(f"\nLanguage breakdown of kept comments:")
    print(f"  contains Kyrgyz-specific letters (Ң/Ө/Ү): {kept_with_kyr} ({kept_with_kyr/len(all_kept)*100:.1f}%)")
    print(f"  Russian/code-mixed (no Kyrgyz-specific):  {len(all_kept) - kept_with_kyr} ({(1-kept_with_kyr/len(all_kept))*100:.1f}%)")

    print(f"\nFiles written:")
    print(f"  {OUT_FILE}")
    print(f"  {STATS_FILE}")
    print(f"  {REJECTED_FILE}")


if __name__ == "__main__":
    main()
