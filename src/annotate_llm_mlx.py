"""LLM-as-second-annotator using mlx-lm (Apple Silicon optimised).

Functionally identical to ``src/annotate_llm.py`` (HuggingFace transformers /
CUDA) but uses Apple's MLX framework with a 4-bit-quantized Aya-Expanse-8B.
Roughly 5-10x faster on M-series Macs than the transformers/MPS path.

Writes to the same ``data/annotated/annotations_llm.csv`` as the CUDA version
so downstream steps (`src/build_dataset.py` for IAA) work identically.

Run:
    python -m src.annotate_llm_mlx                      # all candidates
    python -m src.annotate_llm_mlx --limit 50           # smoke test
    python -m src.annotate_llm_mlx --model mlx-community/aya-expanse-8b-4bit
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time

import pandas as pd

from .paths import (
    ANN_LLM,
    CANDIDATES_HATE,
    CANDIDATES_RANDOM,
    LABELS,
    RNG_SEED,
    ensure_dirs,
)

DEFAULT_MODEL = "mlx-community/aya-expanse-8b-4bit"

LLM_FIELDS = ["id", "video_id", "text", "label", "raw_response",
              "matched_keywords", "label_prior", "model"]

# ── prompts ──────────────────────────────────────────────────────────────────
# Same prompt content as annotate_llm.py for IAA comparability.

SYSTEM_PROMPT = """You are a careful linguistic annotator for Kyrgyz-language social media.
Your task is to classify a comment into exactly one of three labels:

- hate: The comment attacks or dehumanises people because of their ethnicity,
        nationality, religion, sexual orientation, gender, or another
        protected attribute. Slurs targeting such groups count as hate.
        Also: explicit calls for violence/death against any identifiable
        human target (named individuals, political groups, or protected
        classes) count as hate. Exception: calls for legal punishment
        (jail, court) of corrupt politicians or criminals are NOT hate.
- offensive: The comment contains profanity, insults, or vulgar language
        but does NOT target a protected group (e.g. cursing at a politician
        as an individual, generic profanity, personal insults). Also:
        Turkic-Islamic curse formulas in optative mood (`наалат`,
        `Аллах жазалайт`, `жер жутсун`) against non-protected targets
        count as offensive, not hate.
- non_hate: The comment is neutral, critical without insult, supportive,
        or off-topic banter. Political opinions (even harsh) without
        slurs or insults are non_hate. Calls for legal action (jailing,
        ICC) without insult are non_hate. Situation-critique without
        personal attack is non_hate.

Reply with EXACTLY one word: hate, offensive, or non_hate."""

FEW_SHOTS = [
    ("Бул чурка эмне үчүн биздин өлкөгө келди?",
     "hate"),                                  # ethnic slur
    ("Жашасын Кыргызстан, мыкты команда!",
     "non_hate"),
    ("Бул президент чыныгы акмак, эмне кылып жатат?",
     "offensive"),                             # generic insult, no protected group
    ("Пидорастар коомубузга жат, кууп чыгыш керек.",
     "hate"),                                  # anti-LGBT
    ("Мындай саясатты колдобойм, дагы көп жакшырта алышат эле.",
     "non_hate"),
]


def build_chat(text: str) -> list[dict]:
    """Chat messages: system + few-shot pairs + final user turn."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_text, shot_label in FEW_SHOTS:
        msgs.append({"role": "user",
                     "content": f"Comment: {shot_text}\nLabel:"})
        msgs.append({"role": "assistant", "content": shot_label})
    msgs.append({"role": "user",
                 "content": f"Comment: {text}\nLabel:"})
    return msgs


_LABEL_RE = re.compile(r"\b(hate|offensive|non_hate|nonhate|non-hate)\b",
                       re.IGNORECASE)


def parse_label(raw: str) -> str:
    """Map the model's free-text reply to one of LABELS; default non_hate."""
    if not raw:
        return "non_hate"
    m = _LABEL_RE.search(raw)
    if not m:
        return "non_hate"
    tok = m.group(1).lower().replace("-", "_").replace("nonhate", "non_hate")
    return tok if tok in LABELS else "non_hate"


# ── data loading (mirrors annotate_cli + annotate_llm) ──────────────────────

def _load_pool() -> list[dict]:
    rows = []
    for path in (CANDIDATES_HATE, CANDIDATES_RANDOM):
        rows.extend(pd.read_csv(path).fillna("").to_dict(orient="records"))
    seen = {}
    for r in rows:
        seen.setdefault(r["id"], r)
    rows = list(seen.values())
    rng = random.Random(RNG_SEED)
    rng.shuffle(rows)
    return rows


def _load_existing() -> dict[str, dict]:
    if not ANN_LLM.exists():
        return {}
    df = pd.read_csv(ANN_LLM).fillna("")
    return {row["id"]: row.to_dict() for _, row in df.iterrows()}


def _write_all(records: dict[str, dict]) -> None:
    ensure_dirs()
    with ANN_LLM.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LLM_FIELDS)
        w.writeheader()
        for rec in records.values():
            w.writerow({k: rec.get(k, "") for k in LLM_FIELDS})


# ── MLX model wrapper ───────────────────────────────────────────────────────

class MLXAnnotator:
    def __init__(self, model_name: str):
        from mlx_lm import load
        print(f"loading mlx model: {model_name} ...")
        t0 = time.time()
        self.model, self.tokenizer = load(model_name)
        print(f"  loaded in {time.time() - t0:.1f}s")
        self.model_name = model_name

    def label_one(self, text: str) -> tuple[str, str]:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            build_chat(text),
            tokenize=False,
            add_generation_prompt=True,
        )

        # Greedy: temperature 0 via deterministic sampling. mlx-lm's generate
        # accepts a sampler kwarg in newer versions; fall back if unavailable.
        try:
            raw = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=8,
                verbose=False,
            )
        except TypeError:
            # older mlx-lm signature
            raw = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=8,
            )

        # Some mlx-lm versions echo the prompt; strip it if present.
        if raw.startswith(prompt):
            raw = raw[len(prompt):]
        raw = raw.strip()
        return parse_label(raw), raw


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate only N items (smoke test)")
    args = ap.parse_args()

    ensure_dirs()
    pool = _load_pool()
    if args.limit:
        pool = pool[: args.limit]

    existing = _load_existing()
    pending = [r for r in pool if r["id"] not in existing]
    print(f"Pool: {len(pool)} | already done: {len(existing)} | "
          f"pending: {len(pending)}")
    if not pending:
        return

    annot = MLXAnnotator(args.model)

    t0 = time.time()
    for i, row in enumerate(pending, 1):
        try:
            label, raw = annot.label_one(row["text"])
        except Exception as e:                                # pragma: no cover
            print(f"  ERROR on id={row['id']}: {e}")
            label, raw = "non_hate", f"ERROR:{e}"

        existing[row["id"]] = {
            "id": row["id"],
            "video_id": row["video_id"],
            "text": row["text"],
            "label": label,
            "raw_response": raw,
            "matched_keywords": row.get("matched_keywords", ""),
            "label_prior": row.get("label_prior", ""),
            "model": args.model,
        }

        if i % 10 == 0 or i == len(pending):
            _write_all(existing)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(pending) - i) / rate if rate > 0 else 0.0
            print(f"  [{i}/{len(pending)}]  {rate:.2f} items/s  "
                  f"ETA {eta/60:.1f} min")

    _write_all(existing)
    total_min = (time.time() - t0) / 60
    print(f"\ndone in {total_min:.1f} min ({len(pending)} items)")


if __name__ == "__main__":
    main()
