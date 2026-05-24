"""Shared text preprocessing for Kyrgyz hate-speech experiments."""

from __future__ import annotations

import re
import unicodedata

WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
TG_RE = re.compile(r"(t\.me/|telegram\.me/|wa\.me/)\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+", re.UNICODE)
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

# Latin look-alikes commonly used to obfuscate Cyrillic on YouTube.
LATIN_TO_CYRILLIC = str.maketrans({
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
    "x": "х", "y": "у", "k": "к", "h": "н", "b": "в",
    "t": "т", "m": "м",
    "A": "а", "E": "е", "O": "о", "P": "р", "C": "с",
    "X": "х", "Y": "у", "K": "к", "H": "н", "B": "в",
    "T": "т", "M": "м",
    "0": "о", "3": "е", "4": "ч",
})

# Light Kyrgyz/Russian stopword list — opt-in via stopwords=True.
STOPWORDS = {
    "жана", "бирок", "анткени", "себеби", "үчүн", "менен", "болуп", "болот",
    "болгон", "эмес", "эми", "ошондуктан", "башка", "буга", "анда", "ал",
    "ага", "сен", "мен", "биз", "силер", "алар",
    "и", "или", "но", "что", "это", "не", "на", "в", "с", "по", "за",
    "от", "до", "из", "у", "о", "об", "так", "же", "бы", "был", "была",
}


def normalise_minimal(text: str) -> str:
    """NFC + collapse whitespace. Safe default."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    return WS_RE.sub(" ", text).strip()


def normalise_full(text: str, *, stopwords: bool = False) -> str:
    """Aggressive normalisation for TF-IDF Exp 2."""
    text = normalise_minimal(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = TG_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    text = text.translate(LATIN_TO_CYRILLIC)
    text = PUNCT_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = WS_RE.sub(" ", text).strip()
    if stopwords:
        text = " ".join(t for t in text.split() if t not in STOPWORDS)
    return text


def normalise(text: str, profile: str = "minimal", **kwargs) -> str:
    if profile == "minimal":
        return normalise_minimal(text)
    if profile == "full":
        return normalise_full(text, **kwargs)
    raise ValueError(f"Unknown profile: {profile}")
