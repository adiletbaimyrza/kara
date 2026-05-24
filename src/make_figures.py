"""Build all figures and tables from ``results/`` + ``data/splits/``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .paths import (
    CANDIDATES_HATE,
    CANDIDATES_RANDOM,
    COMMENTS_FILTERED,
    DATASET_FINAL,
    FIGURES,
    LABELS,
    RESULTS,
    SUMMARY,
    TABLES,
    TEST,
    TRAIN,
    VAL,
    ensure_dirs,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def _save(fig_name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{fig_name}.png", dpi=150)
    plt.savefig(FIGURES / f"{fig_name}.pdf")
    plt.close()


# Flowcharts (drawn with matplotlib patches — no graphviz dependency)

def _flowbox(ax, x, y, *args, color="#cee3f6"):
    # Accept either (w, h, text) or (text,) — the latter uses default box size.
    if len(args) == 1:
        w, h, text = 1.7, 1.0, args[0]
    elif len(args) == 3:
        w, h, text = args
    else:
        raise TypeError("_flowbox expects (ax, x, y, text) or (ax, x, y, w, h, text)")
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=color, edgecolor="#333", linewidth=1.2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))


def fig_pipeline_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4)
    ax.axis("off")

    # node positions
    nodes = [
        (1.0, 2,  "YouTube\n13 videos\n~15k comments"),
        (3.0, 2,  "Filter\n(11 stages)\n→ 13,902"),
        (5.5, 3,  "Keyword pool\n(profanity/\nslur lexicon)\n702"),
        (5.5, 1,  "Random pool\n500"),
        (8.0, 2,  "Annotation\nself + LLM\n(3-class)"),
        (10.0, 2, "Train/Val/Test\n70/10/20"),
    ]
    for x, y, txt in nodes:
        _flowbox(ax, x, y, 1.7, 1.0, txt)
    _arrow(ax, 1.85, 2, 2.15, 2)
    _arrow(ax, 3.85, 2.1, 4.65, 2.85)
    _arrow(ax, 3.85, 1.9, 4.65, 1.15)
    _arrow(ax, 6.35, 2.85, 7.15, 2.1)
    _arrow(ax, 6.35, 1.15, 7.15, 1.9)
    _arrow(ax, 8.85, 2, 9.15, 2)
    ax.set_title("Dataset construction pipeline")
    _save("pipeline_flowchart")


def fig_annotation_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")
    _flowbox(ax, 1.0, 2,   "Candidates\n(1,202)")
    _flowbox(ax, 3.5, 3.2, "Human annotator\n(CLI tool)", color="#cdedce")
    _flowbox(ax, 3.5, 0.8, "LLM annotator\n(Aya-Expanse-8B)", color="#f7d7c4")
    _flowbox(ax, 6.0, 2,   "Cohen's κ\n+ confusion")
    _flowbox(ax, 8.5, 2,   "Gold dataset\n(human labels)")
    _arrow(ax, 1.85, 2.1, 2.65, 3.0)
    _arrow(ax, 1.85, 1.9, 2.65, 1.0)
    _arrow(ax, 4.35, 3.0, 5.15, 2.1)
    _arrow(ax, 4.35, 1.0, 5.15, 1.9)
    _arrow(ax, 6.85, 2, 7.65, 2)
    ax.set_title("Annotation pipeline")
    _save("annotation_flowchart")


# Dataset figures

def fig_class_distribution() -> None:
    if not TRAIN.exists():
        return
    splits = {"train": pd.read_csv(TRAIN),
              "val":   pd.read_csv(VAL),
              "test":  pd.read_csv(TEST)}
    counts = pd.DataFrame({s: df["label"].value_counts() for s, df in splits.items()})
    counts = counts.reindex(LABELS).fillna(0).astype(int)
    counts.plot(kind="bar", figsize=(7, 4), color=["#7eb0ff", "#ffae65", "#9adf9a"])
    plt.title("Class distribution per split")
    plt.xticks(rotation=0)
    plt.ylabel("count")
    _save("class_distribution")


def fig_text_length_dist() -> None:
    if not DATASET_FINAL.exists():
        return
    df = pd.read_csv(DATASET_FINAL)
    df["n_words"] = df["text"].str.split().str.len()
    plt.figure(figsize=(7, 4))
    for lab, color in zip(LABELS, ["#7eb0ff", "#ffae65", "#cc5e63"]):
        sub = df[df["label"] == lab]["n_words"]
        plt.hist(sub, bins=range(0, max(df["n_words"].max(), 50) + 5, 2),
                 alpha=0.55, label=lab, color=color)
    plt.title("Text length by class")
    plt.xlabel("words per comment")
    plt.ylabel("count")
    plt.legend()
    _save("text_length_dist")


def fig_keyword_category_hits() -> None:
    if not CANDIDATES_HATE.exists():
        return
    df = pd.read_csv(CANDIDATES_HATE)
    cats = (df["matched_categories"].fillna("").str.split("|")
              .explode().str.strip())
    cats = cats[cats != ""].value_counts()
    plt.figure(figsize=(7, 4))
    cats.plot(kind="barh", color="#9c89b8")
    plt.title("Candidates by lexicon category")
    plt.xlabel("comments")
    _save("keyword_category_hits")


def fig_iaa_confusion() -> None:
    cm_path = TABLES / "iaa_confusion.csv"
    if not cm_path.exists():
        return
    cm = pd.read_csv(cm_path, index_col=0)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm.values, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(LABELS)), LABELS, rotation=20)
    plt.yticks(range(len(LABELS)), LABELS)
    plt.xlabel("LLM label")
    plt.ylabel("human label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                     color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
    plt.title("Human vs LLM annotator")
    _save("iaa_confusion")


# Results figures

def _summary() -> pd.DataFrame | None:
    if not SUMMARY.exists():
        return None
    df = pd.read_csv(SUMMARY)
    return df.sort_values("macro_f1", ascending=True)


def fig_results_f1_bar() -> None:
    df = _summary()
    if df is None:
        return
    colors = {"classical": "#7eb0ff", "transformer": "#ffae65", "llm": "#c44e52"}
    df["color"] = df["family"].map(colors).fillna("#999")
    plt.figure(figsize=(8, 5))
    plt.barh(df["name"], df["macro_f1"], color=df["color"])
    for i, v in enumerate(df["macro_f1"]):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.xlabel("macro-F1")
    plt.title("Model comparison — macro-F1 on test set")
    plt.xlim(0, max(1.0, df["macro_f1"].max() + 0.08))
    legend = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
    plt.legend(handles=legend, loc="lower right")
    _save("results_f1_bar")


def fig_confusion_matrices() -> None:
    df = _summary()
    if df is None:
        return
    df = df.sort_values("macro_f1", ascending=False)
    n = len(df)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for i, (_, row) in enumerate(df.iterrows()):
        cm_path = RESULTS / row["name"] / "confusion_matrix.csv"
        if not cm_path.exists():
            continue
        cm = pd.read_csv(cm_path, index_col=0)
        ax = axes[i]
        ax.axis("on")
        ax.imshow(cm.values, cmap="Blues")
        ax.set_title(f"{row['name']}\nF1={row['macro_f1']:.3f}", fontsize=8)
        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=30, fontsize=7)
        ax.set_yticks(range(len(LABELS)))
        ax.set_yticklabels(LABELS, fontsize=7)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, int(cm.values[r, c]), ha="center", va="center",
                        fontsize=7,
                        color="white" if cm.values[r, c] > cm.values.max() / 2 else "black")
    _save("confusion_matrices")


def fig_per_class_metrics() -> None:
    df = _summary()
    if df is None:
        return
    cols = [f"f1_{lab}" for lab in LABELS]
    df = df.set_index("name")[cols]
    df.columns = LABELS
    df.plot(kind="bar", figsize=(10, 4),
            color=["#7eb0ff", "#ffae65", "#cc5e63"])
    plt.title("Per-class F1")
    plt.ylabel("F1")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="class")
    _save("per_class_metrics")


def fig_training_curves() -> None:
    curves = {}
    for exp in ("exp5_mbert", "exp6_xlmr"):
        path = RESULTS / exp / "train_history.json"
        if path.exists():
            curves[exp] = json.loads(path.read_text())
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for exp, hist in curves.items():
        train_loss = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
        val_f1 =    [(h["epoch"], h.get("eval_macro_f1")) for h in hist if "eval_macro_f1" in h]
        if train_loss:
            xs, ys = zip(*train_loss)
            axes[0].plot(xs, ys, label=exp)
        if val_f1:
            xs, ys = zip(*val_f1)
            axes[1].plot(xs, ys, label=exp, marker="o")
    axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].set_title("Val macro-F1"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1"); axes[1].legend()
    _save("training_curves")


# Tables

def _write_results_main() -> None:
    df = _summary()
    if df is None:
        return
    df = df.sort_values(["family", "macro_f1"], ascending=[True, False])
    cols = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    tex = ["\\begin{tabular}{llrrrr}", "\\toprule",
           "Family & Model & Acc & Macro-F1 & Macro-P & Macro-R \\\\",
           "\\midrule"]
    md = ["| Family | Model | Acc | Macro-F1 | Macro-P | Macro-R |",
          "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        vals = [f"{r[c]:.3f}" for c in cols]
        tex.append(f"{r['family']} & {r['name']} & " + " & ".join(vals) + " \\\\")
        md.append(f"| {r['family']} | {r['name']} | " + " | ".join(vals) + " |")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "results_main.tex").write_text("\n".join(tex))
    (TABLES / "results_main.md").write_text("\n".join(md))


def _write_ablation() -> None:
    df = _summary()
    if df is None:
        return
    names = ["exp1_tfidf_baseline", "exp2_tfidf_preproc",
             "exp3_tfidf_char_ngram", "exp4_tfidf_svm"]
    sub = df[df["name"].isin(names)].set_index("name").reindex(names)
    if sub.empty:
        return
    tex = ["\\begin{tabular}{lrr}", "\\toprule",
           "System & Macro-F1 & $\\Delta$ vs baseline \\\\",
           "\\midrule"]
    md = ["| System | Macro-F1 | Δ vs baseline |", "|---|---|---|"]
    base = sub.loc[names[0], "macro_f1"]
    for n in names:
        if n not in sub.index:
            continue
        f1 = sub.loc[n, "macro_f1"]
        delta = f1 - base
        tex.append(f"{n} & {f1:.3f} & {delta:+.3f} \\\\")
        md.append(f"| {n} | {f1:.3f} | {delta:+.3f} |")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "ablation_preprocessing.tex").write_text("\n".join(tex))
    (TABLES / "ablation_preprocessing.md").write_text("\n".join(md))


def _write_llm_vs_finetune() -> None:
    df = _summary()
    if df is None:
        return
    names = ["exp5_mbert", "exp6_xlmr", "exp7_llm_zeroshot", "exp8_llm_fewshot"]
    sub = df[df["name"].isin(names)]
    if sub.empty:
        return
    tex = ["\\begin{tabular}{llrr}", "\\toprule",
           "Family & Model & Macro-F1 & Accuracy \\\\",
           "\\midrule"]
    md = ["| Family | Model | Macro-F1 | Accuracy |", "|---|---|---|---|"]
    for _, r in sub.sort_values("macro_f1", ascending=False).iterrows():
        tex.append(f"{r['family']} & {r['name']} & {r['macro_f1']:.3f} & {r['accuracy']:.3f} \\\\")
        md.append(f"| {r['family']} | {r['name']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} |")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "llm_vs_finetune.tex").write_text("\n".join(tex))
    (TABLES / "llm_vs_finetune.md").write_text("\n".join(md))


# Main

def main() -> None:
    ensure_dirs()

    print("[1/3] flowcharts + dataset figures")
    fig_pipeline_flowchart()
    fig_annotation_flowchart()
    fig_class_distribution()
    fig_text_length_dist()
    fig_keyword_category_hits()
    fig_iaa_confusion()

    print("[2/3] results figures + tables")
    fig_results_f1_bar()
    fig_confusion_matrices()
    fig_per_class_metrics()
    fig_training_curves()
    _write_results_main()
    _write_ablation()
    _write_llm_vs_finetune()

    print("[3/3] error analysis")
    try:
        from .error_analysis import main as ea_main
        ea_main()
    except SystemExit as e:
        print(f"  error_analysis skipped: {e}")

    print("\nFigures:")
    for p in sorted(FIGURES.glob("*.png")):
        print(f"  {p}")
    print("Tables:")
    for p in sorted(TABLES.glob("*")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
