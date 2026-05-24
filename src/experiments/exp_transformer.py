"""Experiments 5 & 6 — fine-tuned multilingual transformers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import LABEL2ID, ID2LABEL, LABELS, RESULTS, RNG_SEED
from ..preprocess import normalise_minimal
from .base import Experiment


def _build_dataset(df: pd.DataFrame, tokenizer, max_len: int):
    from datasets import Dataset
    df = df.copy()
    df["text"] = [normalise_minimal(t) for t in df["text"]]
    df["label_id"] = df["label"].map(LABEL2ID)
    ds = Dataset.from_pandas(df[["id", "text", "label_id"]], preserve_index=False)
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True,
                            max_length=max_len, padding=False),
        batched=True,
    )
    ds = ds.rename_column("label_id", "labels")
    ds = ds.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _compute_metrics_factory():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    def _fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, labels=list(range(len(LABELS))),
                                 average="macro", zero_division=0),
            "macro_precision": precision_score(labels, preds,
                                               labels=list(range(len(LABELS))),
                                               average="macro", zero_division=0),
            "macro_recall": recall_score(labels, preds,
                                         labels=list(range(len(LABELS))),
                                         average="macro", zero_division=0),
        }
    return _fn


class TransformerExperiment(Experiment):
    family = "transformer"
    model_name: str = ""
    max_len: int = 192
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5

    def run(self) -> dict:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        print(f"model: {self.model_name}")
        print(f"device available: cuda={torch.cuda.is_available()} "
              f"mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")

        train_df, val_df, test_df = self.load_splits()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        train_ds = _build_dataset(train_df, tokenizer, self.max_len)
        val_ds   = _build_dataset(val_df, tokenizer, self.max_len)
        test_ds  = _build_dataset(test_df, tokenizer, self.max_len)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        ckpt_dir = self.outdir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Use bf16 on CUDA when available — same memory as fp16 but with
        # fp32's exponent range, avoiding XLM-R LayerNorm overflow (a known
        # issue where XLM-R collapses to predicting one class under fp16).
        # mBERT survives fp16 but bf16 is also fine for it.
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=20,
            report_to=[],
            seed=RNG_SEED,
            bf16=use_bf16,
            fp16=use_fp16,
        )

        # `tokenizer=` was renamed to `processing_class=` in transformers >=4.46
        # and removed entirely in >=4.57. Use the new name unconditionally.
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_compute_metrics_factory(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Save training history
        history = []
        for entry in trainer.state.log_history:
            history.append(entry)
        (self.outdir / "train_history.json").write_text(
            json.dumps(history, indent=2)
        )

        # Test-set predictions
        preds = trainer.predict(test_ds)
        logits = preds.predictions
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = [ID2LABEL[i] for i in np.argmax(logits, axis=-1)]
        y_test = [ID2LABEL[i] for i in test_df["label"].map(LABEL2ID).tolist()]

        # Remove heavy checkpoints to keep results dir light.
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        return self.evaluate(
            y_test, y_pred, ids=test_df["id"], texts=test_df["text"], probs=probs,
            extra={"model_name": self.model_name,
                   "epochs": self.epochs,
                   "batch_size": self.batch_size,
                   "learning_rate": self.lr},
        )


class ExpMBert(TransformerExperiment):
    name = "exp5_mbert"
    model_name = "bert-base-multilingual-cased"
    description = "mBERT fine-tuned, 5 epochs, bs 16, lr 2e-5"


class ExpXLMR(TransformerExperiment):
    name = "exp6_xlmr"
    model_name = "xlm-roberta-base"
    description = "XLM-RoBERTa-base fine-tuned, 5 epochs, bs 16, lr 2e-5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mbert", "xlmr", "both"], default="both")
    args = ap.parse_args()
    if args.model in ("mbert", "both"):
        ExpMBert().run_logged()
    if args.model in ("xlmr", "both"):
        ExpXLMR().run_logged()


if __name__ == "__main__":
    main()
