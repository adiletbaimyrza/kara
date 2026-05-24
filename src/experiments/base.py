"""Common Experiment interface."""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..paths import LABELS, RESULTS, SUMMARY, TEST, TRAIN, VAL, ensure_dirs


class Experiment:
    """Subclass and override ``name`` and ``run``."""

    name: str = "base"
    family: str = "misc"   # "classical" / "transformer" / "llm"
    description: str = ""

    def __init__(self):
        ensure_dirs()
        self.outdir: Path = RESULTS / self.name
        self.outdir.mkdir(parents=True, exist_ok=True)

    # -- data ---------------------------------------------------------------

    @staticmethod
    def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            pd.read_csv(TRAIN),
            pd.read_csv(VAL),
            pd.read_csv(TEST),
        )

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        y_true: Iterable[str],
        y_pred: Iterable[str],
        ids: Iterable[str],
        texts: Iterable[str],
        probs: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> dict:
        y_true = list(y_true)
        y_pred = list(y_pred)

        metrics = {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "n_test": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
            "per_class": {},
        }
        for label in LABELS:
            metrics["per_class"][label] = {
                "precision": float(precision_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
                "f1":        float(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)),
                "support":   int(sum(1 for y in y_true if y == label)),
            }
        if extra:
            metrics.update(extra)

        # predictions
        pred_df = pd.DataFrame({
            "id": list(ids),
            "text": list(texts),
            "gold": y_true,
            "pred": y_pred,
        })
        if probs is not None:
            probs = np.asarray(probs)
            for i, label in enumerate(LABELS):
                pred_df[f"p_{label}"] = probs[:, i]
        pred_df.to_csv(self.outdir / "predictions.csv", index=False)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(self.outdir / "confusion_matrix.csv")

        (self.outdir / "classification_report.txt").write_text(
            classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=4)
        )
        (self.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        self._append_summary(metrics)
        return metrics

    @staticmethod
    def _append_summary(m: dict) -> None:
        row = {
            "name": m["name"],
            "family": m["family"],
            "description": m["description"],
            "n_test": m["n_test"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "weighted_f1": m["weighted_f1"],
        }
        for label in LABELS:
            row[f"f1_{label}"] = m["per_class"][label]["f1"]

        if SUMMARY.exists():
            df = pd.read_csv(SUMMARY)
            df = df[df["name"] != m["name"]]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(SUMMARY, index=False)

    # -- orchestration ------------------------------------------------------

    def run(self) -> dict:
        raise NotImplementedError

    def run_logged(self, force: bool = False) -> dict:
        """Run with stdout captured to ``log.txt``. Skips if metrics.json exists.

        Set ``force=True`` (or env ``KARA_FORCE=1``) to re-run anyway.
        """
        import json
        import os

        force = force or os.environ.get("KARA_FORCE") == "1"
        metrics_path = self.outdir / "metrics.json"
        if metrics_path.exists() and not force:
            existing = json.loads(metrics_path.read_text())
            print(f"=== {self.name} ===")
            print(f"SKIPPED: {metrics_path} already exists "
                  f"(macro_f1={existing.get('macro_f1', 'n/a'):.4f}). "
                  f"Set KARA_FORCE=1 to re-run.")
            return existing

        log_path = self.outdir / "log.txt"
        t0 = time.time()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(f"=== {self.name} ===")
            print(f"started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            metrics = self.run()
            print(f"finished in {time.time() - t0:.1f}s")
            print(f"macro_f1 = {metrics['macro_f1']:.4f}")
        log_path.write_text(buf.getvalue())
        print(buf.getvalue())
        return metrics
