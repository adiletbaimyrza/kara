"""LLM-as-second-annotator.

Uses **Aya-Expanse-8B** (``CohereForAI/aya-expanse-8b``) — a multilingual
instruction-tuned LLM that explicitly supports Kyrgyz. We prompt the model with
the same annotation guidelines a human gets, plus a small number of in-context
examples, and parse a single-token label per comment.

Reads the union of ``candidates_hate.csv`` + ``candidates_random.csv`` (same
deterministic shuffle as the human annotator, so the IDs align 1-to-1), runs
inference, and writes ``data/annotated/annotations_llm.csv``.

Run:
    python -m src.annotate_llm                      # all candidates
    python -m src.annotate_llm --limit 50           # smoke test
    python -m src.annotate_llm --model CohereForAI/aya-expanse-8b
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

import pandas as pd

from .paths import (
    ANN_LLM,
    CANDIDATES_HATE,
    CANDIDATES_RANDOM,
    LABELS,
    RNG_SEED,
    ensure_dirs,
)

DEFAULT_MODEL = "CohereForAI/aya-expanse-8b"

LLM_FIELDS = ["id", "video_id", "text", "label", "raw_response",
              "matched_keywords", "label_prior", "model"]

# ── prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a careful linguistic annotator for Kyrgyz-language social media.
Your task is to classify a comment into exactly one of three labels:

- hate: The comment attacks or dehumanises people because of their ethnicity,
        nationality, religion, sexual orientation, gender, or another
        protected attribute. Slurs targeting such groups count as hate.
- offensive: The comment contains profanity, insults, or vulgar language
        but does NOT target a protected group (e.g. cursing at a politician
        as an individual, generic profanity, personal insults).
- non_hate: The comment is neutral, critical without insult, supportive,
        or off-topic banter.

Reply with EXACTLY one word: hate, offensive, or non_hate."""

# Few-shot examples (manually crafted; small and representative).
FEW_SHOTS = [
    ("Бул чурка эмне үчүн биздин өлкөгө келди?",
     "hate"),  # ethnic slur
    ("Жашасын Кыргызстан, мыкты команда!",
     "non_hate"),
    ("Бул президент чыныгы акмак, эмне кылып жатат?",
     "offensive"),  # generic insult, no protected group
    ("Пидорастар коомубузга жат, кууп чыгыш керек.",
     "hate"),  # anti-LGBT
    ("Мындай саясатты колдобойм, дагы көп жакшырта алышат эле.",
     "non_hate"),
]


def build_prompt(text: str) -> list[dict]:
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


# ── data loading (must mirror annotate_cli._load_pool) ──────────────────────

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


# ── model wrapper ───────────────────────────────────────────────────────────

class LLMAnnotator:
    def __init__(self, model_name: str, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"loading model {model_name} (dtype={dtype}) ...")
        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        self.model_name = model_name

    def label_one(self, text: str) -> tuple[str, str]:
        import torch
        msgs = build_prompt(text)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        return parse_label(raw), raw


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate only N items (smoke test)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
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

    annot = LLMAnnotator(args.model, args.dtype)

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
        if i % 20 == 0 or i == len(pending):
            _write_all(existing)
            rate = i / (time.time() - t0)
            eta = (len(pending) - i) / rate
            print(f"  [{i}/{len(pending)}] {rate:.2f} items/s  ETA {eta:.0f}s")

    _write_all(existing)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
