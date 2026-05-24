"""Best-model error categorisation."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .paths import FIGURES, LABELS, RESULTS, SUMMARY, TABLES, ensure_dirs

SLUR_WORDS = re.compile(
    r"\b(чурка|хач|жид|пидор|педик|гомик|сарт|узкоглаз|кыргызня|"
    r"хохол|пиндос|манкурт)\w*", re.IGNORECASE
)
LATIN_RE = re.compile(r"[A-Za-z]")
QUOTE_RE = re.compile(r"['\"«»‘’“”]")
TARGET_WORDS = re.compile(
    r"\b(все|алар|их|их\s+\w+|нация|нация\s+\w+|народ|орус|кыргыз|узбек|"
    r"казах|таджик|еврей|гей|лесбиян|мусулман)\w*", re.IGNORECASE
)


def categorise(row) -> str:
    text = str(row["text"])
    n_words = len(text.split())
    has_slur = bool(SLUR_WORDS.search(text))
    has_latin = bool(LATIN_RE.search(text))
    has_quote = bool(QUOTE_RE.search(text)) or " деп " in text.lower()
    has_target = bool(TARGET_WORDS.search(text))

    if n_words <= 4:
        return "low_information"
    if has_slur and has_quote:
        return "slur_quoted"
    if has_slur and not has_target:
        return "ambiguous_target"
    if has_slur and row["gold"] != "hate" and row["pred"] == "hate":
        return "keyword_fp"
    if has_latin and any(0x0400 <= ord(c) <= 0x04ff for c in text):
        return "code_switching"
    return "other"


def best_experiment() -> tuple[str, Path]:
    df = pd.read_csv(SUMMARY)
    df = df.sort_values("macro_f1", ascending=False)
    name = df.iloc[0]["name"]
    pred_path = RESULTS / name / "predictions.csv"
    return name, pred_path


def main() -> None:
    ensure_dirs()
    if not SUMMARY.exists():
        raise SystemExit(f"No {SUMMARY}. Run experiments first.")

    best_name, pred_path = best_experiment()
    print(f"best model: {best_name}")
    df = pd.read_csv(pred_path)
    errors = df[df["gold"] != df["pred"]].copy()
    errors["category"] = errors.apply(categorise, axis=1)
    print(f"errors: {len(errors)} / {len(df)} "
          f"({len(errors) / len(df) * 100:.1f}%)")

    # ── bar chart ──
    counts = errors["category"].value_counts()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar", color="#c44e52")
    plt.title(f"Error categories — {best_name}")
    plt.ylabel("count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "error_categories.png", dpi=150)
    plt.savefig(FIGURES / "error_categories.pdf")
    plt.close()

    # ── table: representative errors (up to 3 per category) ──
    # NB: include_groups=False drops the `category` column; do the grouping
    # manually instead so it stays in the output.
    sampled_parts = []
    for cat, g in errors.groupby("category"):
        sampled_parts.append(g.head(3))
    sampled = pd.concat(sampled_parts, ignore_index=True).head(12)

    rows = []
    for _, r in sampled.iterrows():
        text = str(r["text"]).replace("&", r"\&").replace("%", r"\%")
        if len(text) > 120:
            text = text[:117] + "..."
        rows.append((r["category"], r["gold"], r["pred"], text))

    lines = ["\\begin{tabular}{lllp{0.45\\textwidth}}", "\\toprule",
             "Category & Gold & Pred & Comment \\\\", "\\midrule"]
    for cat, gold, pred, text in rows:
        lines.append(f"{cat} & {gold} & {pred} & {text} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (TABLES / "error_examples.tex").write_text("\n".join(lines))

    md = ["| Category | Gold | Pred | Comment |", "|---|---|---|---|"]
    for cat, gold, pred, text in rows:
        text_md = text.replace("|", "\\|")
        md.append(f"| {cat} | {gold} | {pred} | {text_md} |")
    (TABLES / "error_examples.md").write_text("\n".join(md))

    print(f"  figures/error_categories.png  +  tables/error_examples.tex,.md")


if __name__ == "__main__":
    main()
