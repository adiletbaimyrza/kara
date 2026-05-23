"""Driver — runs every experiment listed in ``EXPERIMENTS`` and writes
``results/summary.csv`` (one row per experiment).

Usage:
    python -m src.run_all                       # everything
    python -m src.run_all --only exp1,exp2      # only some
    python -m src.run_all --skip llm            # everything except LLM
    python -m src.run_all --force               # overwrite existing results

The driver is idempotent: an experiment whose ``metrics.json`` already exists
is skipped unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Type

from .experiments.base import Experiment
from .experiments.exp_tfidf_baseline import ExpTfidfBaseline
from .experiments.exp_tfidf_char_ngram import ExpTfidfCharNgram
from .experiments.exp_tfidf_preproc import ExpTfidfPreproc
from .experiments.exp_tfidf_svm import ExpTfidfSvm
from .paths import RESULTS

# Transformers and LLM experiments are imported lazily — they require heavy
# deps (``transformers``, ``torch``) that are only present on Helios. We still
# want classical experiments to run on a laptop without those installed.


def _maybe_import_transformer():
    try:
        from .experiments.exp_transformer import ExpMBert, ExpXLMR
        return [ExpMBert, ExpXLMR]
    except ImportError as e:
        print(f"  (skipping transformer experiments — missing dep: {e})")
        return []


def _maybe_import_llm():
    try:
        from .experiments.exp_llm_fewshot import ExpLLMFewShot
        from .experiments.exp_llm_zeroshot import ExpLLMZeroShot
        return [ExpLLMZeroShot, ExpLLMFewShot]
    except ImportError as e:
        print(f"  (skipping LLM experiments — missing dep: {e})")
        return []


CLASSICAL: list[Type[Experiment]] = [
    ExpTfidfBaseline,
    ExpTfidfPreproc,
    ExpTfidfCharNgram,
    ExpTfidfSvm,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated experiment names "
                                   "(prefix match, e.g. exp1,exp5)")
    ap.add_argument("--skip", help="comma-separated families to skip: "
                                   "classical,transformer,llm")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if metrics.json exists")
    args = ap.parse_args()

    only = set((args.only or "").split(",")) if args.only else None
    skip_families = set((args.skip or "").split(",")) if args.skip else set()

    families: list[Type[Experiment]] = list(CLASSICAL)
    if "transformer" not in skip_families:
        families.extend(_maybe_import_transformer())
    if "llm" not in skip_families:
        families.extend(_maybe_import_llm())
    if "classical" in skip_families:
        families = [c for c in families if c not in CLASSICAL]

    print(f"Experiments queued: {[c.name for c in families]}\n")

    summary: list[tuple[str, str, float | None]] = []
    for cls in families:
        name = cls.name
        if only and not any(name.startswith(p) for p in only):
            print(f"-- skip {name} (not in --only)")
            continue

        metrics_path = RESULTS / name / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"-- skip {name} (metrics.json exists; use --force to overwrite)")
            continue

        print(f"\n=========================================")
        print(f"RUN  {name}")
        print(f"=========================================")
        t0 = time.time()
        try:
            metrics = cls().run_logged()
            summary.append((name, "ok", metrics.get("macro_f1")))
            print(f"-> macro_f1 = {metrics['macro_f1']:.4f}  "
                  f"({time.time() - t0:.1f}s)")
        except Exception as exc:                                    # pragma: no cover
            traceback.print_exc()
            summary.append((name, f"ERR:{exc}", None))

    print("\n=========================================")
    print("ALL DONE")
    print("=========================================")
    for name, status, f1 in summary:
        f1s = f"{f1:.4f}" if f1 is not None else "  -   "
        print(f"  {name:<28} {status:<8} macro_f1={f1s}")


if __name__ == "__main__":
    main()
