"""Merge human + LLM annotations, compute IAA, write splits.

Gold label = the human label. LLM label is used only for Inter-Annotator
Agreement (Cohen's κ, % agreement, per-class agreement, and a confusion matrix
saved as ``figures/iaa_confusion.png``).

After dropping items the human labelled "skip", we produce a stratified
70/10/20 train/val/test split with a fixed seed and write to
``data/splits/``.

A summary is appended to ``DISCOVERIES.md``.

Run:
    python -m src.build_dataset
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .paths import (
    ANN_HUMAN,
    ANN_LLM,
    DATASET_FINAL,
    DISCOVERIES,
    LABELS,
    RNG_SEED,
    SPLITS,
    TABLES,
    TEST,
    TRAIN,
    VAL,
    ensure_dirs,
)


def _load_human() -> pd.DataFrame:
    if not ANN_HUMAN.exists():
        raise SystemExit(
            f"Missing {ANN_HUMAN}. Run `python -m src.annotate_cli` first."
        )
    df = pd.read_csv(ANN_HUMAN)
    df = df[df["label"].isin(LABELS)].copy()        # drop "skip" and empty
    print(f"Human annotations (excluding skip): {len(df)}")
    return df


def _load_llm() -> pd.DataFrame | None:
    if not ANN_LLM.exists():
        print(f"No LLM annotations at {ANN_LLM}; IAA will be skipped.")
        return None
    df = pd.read_csv(ANN_LLM)
    df = df[df["label"].isin(LABELS)].copy()
    print(f"LLM annotations: {len(df)}")
    return df


def _iaa(human: pd.DataFrame, llm: pd.DataFrame) -> dict:
    merged = human[["id", "label"]].rename(columns={"label": "human"}).merge(
        llm[["id", "label"]].rename(columns={"label": "llm"}),
        on="id", how="inner",
    )
    if len(merged) == 0:
        return {"n_overlap": 0}

    kappa = cohen_kappa_score(merged["human"], merged["llm"], labels=LABELS)
    agree_total = (merged["human"] == merged["llm"]).mean()

    per_class = {}
    for lab in LABELS:
        sub = merged[merged["human"] == lab]
        per_class[lab] = {
            "n_human": int(len(sub)),
            "agreement": float((sub["llm"] == lab).mean()) if len(sub) else 0.0,
        }

    cm = confusion_matrix(merged["human"], merged["llm"], labels=LABELS)
    return {
        "n_overlap": int(len(merged)),
        "cohen_kappa": float(kappa),
        "raw_agreement": float(agree_total),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "labels": LABELS,
    }


def _write_iaa_table(iaa: dict) -> None:
    out = TABLES / "annotation_iaa.tex"
    md = TABLES / "annotation_iaa.md"
    TABLES.mkdir(parents=True, exist_ok=True)

    if iaa.get("n_overlap", 0) == 0:
        out.write_text("% no LLM annotations available\n")
        md.write_text("_No LLM annotations available._\n")
        return

    rows = [("\\# overlap items", str(iaa["n_overlap"])),
            ("Raw agreement", f"{iaa['raw_agreement']:.3f}"),
            ("Cohen's $\\kappa$", f"{iaa['cohen_kappa']:.3f}")]
    for lab in LABELS:
        rows.append((f"Agreement on `{lab}`",
                     f"{iaa['per_class'][lab]['agreement']:.3f} "
                     f"(n={iaa['per_class'][lab]['n_human']})"))

    tex = ["\\begin{tabular}{lr}", "\\toprule",
           "Metric & Value \\\\", "\\midrule"]
    for k, v in rows:
        tex.append(f"{k} & {v} \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    out.write_text("\n".join(tex))

    md_rows = []
    for k, v in rows:
        k_clean = k.replace("$\\kappa$", "κ").replace("\\#", "#")
        md_rows.append(f"| {k_clean} | {v} |")
    md.write_text(
        "| Metric | Value |\n|---|---|\n" + "\n".join(md_rows) + "\n"
    )


def _write_iaa_confusion(iaa: dict) -> None:
    if iaa.get("n_overlap", 0) == 0:
        return
    cm_df = pd.DataFrame(iaa["confusion"], index=LABELS, columns=LABELS)
    cm_df.to_csv(TABLES / "iaa_confusion.csv")


def _split_and_save(df: pd.DataFrame) -> dict:
    """Stratified 70/10/20 split."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RNG_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=2/3, stratify=temp_df["label"], random_state=RNG_SEED,
    )

    SPLITS.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN, index=False)
    val_df.to_csv(VAL, index=False)
    test_df.to_csv(TEST, index=False)

    return {
        "train": dict(Counter(train_df["label"])),
        "val":   dict(Counter(val_df["label"])),
        "test":  dict(Counter(test_df["label"])),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }


def _write_dataset_stats(stats: dict, df: pd.DataFrame) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Class & Train & Val & Test \\\\", "\\midrule"]
    for lab in LABELS:
        lines.append(
            f"{lab} & {stats['train'].get(lab, 0)} & "
            f"{stats['val'].get(lab, 0)} & "
            f"{stats['test'].get(lab, 0)} \\\\"
        )
    lines.extend([
        f"\\midrule",
        f"Total & {stats['n_train']} & {stats['n_val']} & {stats['n_test']} \\\\",
        "\\bottomrule", "\\end{tabular}",
    ])
    (TABLES / "dataset_stats.tex").write_text("\n".join(lines))

    md = [
        "| Class | Train | Val | Test |",
        "|---|---|---|---|",
    ]
    for lab in LABELS:
        md.append(
            f"| {lab} | {stats['train'].get(lab, 0)} "
            f"| {stats['val'].get(lab, 0)} "
            f"| {stats['test'].get(lab, 0)} |"
        )
    md.append(
        f"| **Total** | {stats['n_train']} | {stats['n_val']} | {stats['n_test']} |"
    )
    md.append("")
    md.append(f"- Average text length: {df['text'].str.split().str.len().mean():.1f} words")
    md.append(f"- Distinct tokens (whitespace): {len(set(' '.join(df['text']).split())):,}")
    (TABLES / "dataset_stats.md").write_text("\n".join(md))


def _append_discoveries(stats: dict, iaa: dict, n_total: int) -> None:
    import datetime
    date = datetime.date.today().isoformat()
    chunk = [f"\n## {date} — dataset-build — Final annotated dataset stats\n"]
    chunk.append(f"After dropping skipped items we have **{n_total}** "
                 f"labelled comments. Class balance: ")
    counts = Counter()
    for split in ("train", "val", "test"):
        for lab, n in stats[split].items():
            counts[lab] += n
    counts_str = ", ".join(f"{lab}={counts[lab]}" for lab in LABELS)
    chunk.append(f"{counts_str}. Split sizes: "
                 f"train={stats['n_train']}, val={stats['n_val']}, "
                 f"test={stats['n_test']}.\n")

    if iaa.get("n_overlap"):
        chunk.append(f"\n## {date} — annotation — Human vs LLM agreement\n")
        chunk.append(
            f"Cohen's κ between the human annotator and Aya-Expanse-8B "
            f"(prompted with the same guidelines) was "
            f"**{iaa['cohen_kappa']:.3f}** on n={iaa['n_overlap']} overlapping "
            f"items, raw agreement {iaa['raw_agreement']:.3f}. "
            f"Agreement is most consistent on `non_hate` "
            f"({iaa['per_class']['non_hate']['agreement']:.2f}) and weakest on "
            f"`hate` ({iaa['per_class']['hate']['agreement']:.2f}); see "
            f"`tables/annotation_iaa.tex` and `tables/iaa_confusion.csv`.\n"
        )

    with DISCOVERIES.open("a", encoding="utf-8") as f:
        f.write("".join(chunk))


def main() -> None:
    ensure_dirs()

    human = _load_human()
    llm = _load_llm()

    iaa = _iaa(human, llm) if llm is not None else {"n_overlap": 0}
    _write_iaa_table(iaa)
    _write_iaa_confusion(iaa)

    final = human[["id", "video_id", "text", "label"]].copy()
    final.to_csv(DATASET_FINAL, index=False)

    stats = _split_and_save(final)
    _write_dataset_stats(stats, final)

    print("\nFinal dataset:")
    print(f"  total = {len(final)}")
    print(f"  train = {stats['n_train']}  {stats['train']}")
    print(f"  val   = {stats['n_val']}  {stats['val']}")
    print(f"  test  = {stats['n_test']}  {stats['test']}")
    if iaa.get("n_overlap"):
        print(f"\nIAA (human vs LLM): κ={iaa['cohen_kappa']:.3f}, "
              f"raw agreement={iaa['raw_agreement']:.3f}, "
              f"n={iaa['n_overlap']}")

    _append_discoveries(stats, iaa, len(final))


if __name__ == "__main__":
    main()
