"""Experiment 8 — Aya-Expanse-8B few-shot (5-shot per class).

Same model as Exp 7 with k=5 in-context examples sampled from the training
split (deterministic seed). Tests whether in-context examples close the gap
to fine-tuned transformers.
"""

from __future__ import annotations

import argparse

from ._llm_common import (
    DEFAULT_MODEL,
    HFCausalLM,
    build_messages,
    parse_label,
    pick_few_shots,
)
from ..paths import RNG_SEED
from .base import Experiment


class ExpLLMFewShot(Experiment):
    name = "exp8_llm_fewshot"
    family = "llm"
    description = "Aya-Expanse-8B 5-shot per class (15 examples in-context)"

    def __init__(self, model_name: str = DEFAULT_MODEL, k_per_class: int = 5):
        super().__init__()
        self.model_name = model_name
        self.k_per_class = k_per_class

    def run(self) -> dict:
        train, _val, test = self.load_splits()
        shots = pick_few_shots(train, self.k_per_class, seed=RNG_SEED)
        print(f"few-shot examples: {len(shots)} "
              f"(k={self.k_per_class} per class)")

        lm = HFCausalLM(self.model_name)

        preds, raws = [], []
        for i, text in enumerate(test["text"].tolist()):
            try:
                raw = lm.generate(build_messages(text, few_shots=shots))
            except Exception as e:                                  # pragma: no cover
                raw = f"ERROR:{e}"
            preds.append(parse_label(raw))
            raws.append(raw)
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(test)}]")

        return self.evaluate(
            test["label"].tolist(),
            preds,
            ids=test["id"],
            texts=test["text"],
            probs=None,
            extra={"model_name": self.model_name,
                   "k_per_class": self.k_per_class,
                   "raw_samples": raws[:20]},
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ExpLLMFewShot(args.model, args.k).run_logged()


if __name__ == "__main__":
    main()
