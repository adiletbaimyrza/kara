"""Centralised filesystem paths so every script agrees."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
COMMENTS_FILTERED = DATA / "comments_filtered.csv"
CANDIDATES_HATE = DATA / "candidates_hate.csv"
CANDIDATES_RANDOM = DATA / "candidates_random.csv"

ANNOTATED = DATA / "annotated"
ANN_HUMAN = ANNOTATED / "annotations_human.csv"
ANN_LLM = ANNOTATED / "annotations_llm.csv"
DATASET_FINAL = ANNOTATED / "dataset_final.csv"

SPLITS = DATA / "splits"
TRAIN = SPLITS / "train.csv"
VAL = SPLITS / "val.csv"
TEST = SPLITS / "test.csv"

RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary.csv"

FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

DISCOVERIES = ROOT / "DISCOVERIES.md"

LABELS = ["non_hate", "offensive", "hate"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}

RNG_SEED = 42


def ensure_dirs():
    for d in (ANNOTATED, SPLITS, RESULTS, FIGURES, TABLES):
        d.mkdir(parents=True, exist_ok=True)
