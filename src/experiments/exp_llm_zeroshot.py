"""Experiment 7 — Aya-Expanse-8B zero-shot."""

from __future__ import annotations

import argparse

import numpy as np

from ._llm_common import DEFAULT_MODEL, HFCausalLM, build_messages, parse_label
from .base import Experiment


class ExpLLMZeroShot(Experiment):
    name = "exp7_llm_zeroshot"
    family = "llm"
    description = "Aya-Expanse-8B zero-shot"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        self.model_name = model_name

    def run(self) -> dict:
        _train, _val, test = self.load_splits()
        lm = HFCausalLM(self.model_name)

        preds, raws = [], []
        for i, text in enumerate(test["text"].tolist()):
            try:
                raw = lm.generate(build_messages(text, few_shots=None))
            except Exception as e:                                  # pragma: no cover
                raw = f"ERROR:{e}"
            preds.append(parse_label(raw))
            raws.append(raw)
            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(test)}]")

        # We don't have probabilities from generative LM.
        return self.evaluate(
            test["label"].tolist(),
            preds,
            ids=test["id"],
            texts=test["text"],
            probs=None,
            extra={"model_name": self.model_name, "raw_samples": raws[:20]},
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    ExpLLMZeroShot(args.model).run_logged()


if __name__ == "__main__":
    main()
