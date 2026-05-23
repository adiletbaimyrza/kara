"""Terminal annotation tool for the Kyrgyz hate-speech dataset.

Loads ``candidates_hate.csv`` + ``candidates_random.csv``, shuffles them with a
fixed seed, and asks the user to label each comment with one keystroke. After
every keystroke we flush to ``data/annotated/annotations_human.csv`` so a
crash never loses more than one label.

Run:
    python -m src.annotate_cli                    # start / resume annotation
    python -m src.annotate_cli --review hate      # cycle through your hate labels
    python -m src.annotate_cli --stats            # show progress + counts

Keys:
    h = hate           o = offensive          n = non_hate
    s = skip           u = undo last           q = quit (saves & exits)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import termios
import time
import tty
from collections import Counter
from pathlib import Path

import pandas as pd

from .paths import (
    ANN_HUMAN,
    CANDIDATES_HATE,
    CANDIDATES_RANDOM,
    LABELS,
    RNG_SEED,
    ensure_dirs,
)

KEY2LABEL = {"h": "hate", "o": "offensive", "n": "non_hate"}
ALL_LABELS_PLUS = LABELS + ["skip"]

ANN_FIELDS = [
    "id",
    "video_id",
    "text",
    "label",
    "label_prior",
    "matched_keywords",
    "matched_categories",
    "annot_time_sec",
    "annot_ts",
]


# ── terminal helpers ─────────────────────────────────────────────────────────

def _getch() -> str:
    """Read a single keystroke. Falls back to line input if raw mode fails.

    Some terminals (Warp, tmux/screen with odd configs, IDE terminals, certain
    ssh PTYs) don't honour tty.setraw and silently leave stdin in cooked mode.
    In that case sys.stdin.read(1) would block until Enter. We detect that by
    requiring stdin to actually be a TTY first, and we also offer a
    ``KARA_SIMPLE_INPUT=1`` env override that uses input() unconditionally.
    """
    import os as _os
    if _os.environ.get("KARA_SIMPLE_INPUT") == "1" or not sys.stdin.isatty():
        s = input("> ").strip().lower()
        return s[0] if s else ""

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        s = input("> ").strip().lower()
        return s[0] if s else ""

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _clear() -> None:
    print("\033[2J\033[H", end="")


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


def _color(s: str, c: str) -> str:
    return {"r": f"\033[31m{s}\033[0m",
            "g": f"\033[32m{s}\033[0m",
            "y": f"\033[33m{s}\033[0m",
            "b": f"\033[34m{s}\033[0m",
            "m": f"\033[35m{s}\033[0m"}.get(c, s)


# ── load / save ──────────────────────────────────────────────────────────────

def _load_pool() -> list[dict]:
    """Concatenate hate + random pools, deduplicate, shuffle deterministically."""
    rows = []
    for path in (CANDIDATES_HATE, CANDIDATES_RANDOM):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
        rows.extend(pd.read_csv(path).fillna("").to_dict(orient="records"))

    # dedup by id
    seen = {}
    for r in rows:
        seen.setdefault(r["id"], r)
    rows = list(seen.values())

    rng = random.Random(RNG_SEED)
    rng.shuffle(rows)
    return rows


def _load_existing() -> dict[str, dict]:
    if not ANN_HUMAN.exists():
        return {}
    df = pd.read_csv(ANN_HUMAN).fillna("")
    return {row["id"]: row.to_dict() for _, row in df.iterrows()}


def _write_all(records: dict[str, dict]) -> None:
    ensure_dirs()
    ANN_HUMAN.parent.mkdir(parents=True, exist_ok=True)
    with ANN_HUMAN.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ANN_FIELDS)
        w.writeheader()
        for rec in records.values():
            w.writerow({k: rec.get(k, "") for k in ANN_FIELDS})


# ── stats ────────────────────────────────────────────────────────────────────

def _print_stats(records: dict[str, dict], total: int) -> None:
    counts = Counter(r["label"] for r in records.values() if r.get("label"))
    print(_bold("Annotation progress"))
    print("-" * 40)
    done = sum(counts.values())
    print(f"Annotated: {done} / {total}  ({done / total * 100:.1f}%)")
    for lab in ALL_LABELS_PLUS:
        n = counts.get(lab, 0)
        pct = n / done * 100 if done else 0
        print(f"  {lab:<12} {n:>5}  ({pct:>5.1f}%)")
    # time/pace
    times = [float(r["annot_time_sec"]) for r in records.values()
             if r.get("annot_time_sec")]
    if times:
        med = sorted(times)[len(times) // 2]
        print(f"\nMedian time/sample: {med:.1f}s")
        print(f"Total annotation time: {sum(times) / 60:.1f} min")


# ── main loop ────────────────────────────────────────────────────────────────

def annotate() -> None:
    ensure_dirs()
    pool = _load_pool()
    existing = _load_existing()

    pending = [r for r in pool if r["id"] not in existing]
    if not pending:
        print(_color("All candidates already annotated.", "g"))
        _print_stats(existing, len(pool))
        return

    print(f"Total pool: {len(pool)} | Already done: {len(existing)} | "
          f"Pending: {len(pending)}")
    print(_dim("Press any key to start, q to quit."))
    if _getch() in ("q", "Q"):
        return

    history: list[str] = []  # ids written this session, for undo
    t_session = time.time()

    for i, row in enumerate(pending):
        _clear()
        done = len(existing)
        total = len(pool)
        print(_bold(f"[{done + 1}/{total}]  ({(done + 1) / total * 100:.1f}%)"))
        print(_dim(f"id={row['id']}  video={row['video_id']}  "
                   f"prior={row.get('label_prior', '')}"))
        if row.get("matched_keywords"):
            print(_color(f"matched: {row['matched_keywords']}", "y"))
        print()
        print(_bold(row["text"]))
        print()
        print(_dim("h=hate  o=offensive  n=non_hate  s=skip  u=undo  q=quit"))

        t0 = time.time()
        while True:
            ch = _getch().lower()
            if ch == "q":
                _write_all(existing)
                print()
                _print_stats(existing, len(pool))
                print(_color("Saved. Exit.", "g"))
                return
            if ch == "u":
                if history:
                    last = history.pop()
                    existing.pop(last, None)
                    _write_all(existing)
                    print(_color(f"undid {last}", "y"))
                    time.sleep(0.4)
                # re-prompt this same item (don't advance)
                continue
            if ch == "s":
                label = "skip"
                break
            if ch in KEY2LABEL:
                label = KEY2LABEL[ch]
                break
            # unknown key: show visible feedback so the user knows we're alive
            print(_color(f"  ? unknown key ({ch!r}) — use h/o/n/s/u/q", "y"))

        elapsed = time.time() - t0
        rec = {
            "id": row["id"],
            "video_id": row["video_id"],
            "text": row["text"],
            "label": label,
            "label_prior": row.get("label_prior", ""),
            "matched_keywords": row.get("matched_keywords", ""),
            "matched_categories": row.get("matched_categories", ""),
            "annot_time_sec": round(elapsed, 2),
            "annot_ts": int(time.time()),
        }
        existing[row["id"]] = rec
        history.append(row["id"])
        _write_all(existing)

    _clear()
    print(_color("All done!", "g"))
    _print_stats(existing, len(pool))
    print(f"\nThis session: {(time.time() - t_session) / 60:.1f} min")


def review(label: str) -> None:
    """Re-display all items you labelled with ``label`` so you can sanity-check."""
    if label not in ALL_LABELS_PLUS:
        raise SystemExit(f"Unknown label: {label}. Use one of {ALL_LABELS_PLUS}")
    records = _load_existing()
    matches = [r for r in records.values() if r["label"] == label]
    print(f"{len(matches)} items labelled '{label}'\n")
    for r in matches:
        print(_bold(f"[{r['id']}]"), _color(f"({r['label']})", "y"))
        print(r["text"])
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate Kyrgyz hate-speech candidates.")
    ap.add_argument("--stats", action="store_true", help="show progress and exit")
    ap.add_argument("--review", metavar="LABEL", help="display all items with the given label")
    args = ap.parse_args()

    if args.stats:
        pool = _load_pool()
        _print_stats(_load_existing(), len(pool))
        return
    if args.review:
        review(args.review)
        return
    annotate()


if __name__ == "__main__":
    main()
