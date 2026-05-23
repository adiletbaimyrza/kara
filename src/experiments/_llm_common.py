"""Shared LLM eval utilities for Exp 7 (zero-shot) and Exp 8 (few-shot)."""

from __future__ import annotations

import re

from ..paths import LABELS

DEFAULT_MODEL = "CohereForAI/aya-expanse-8b"

SYSTEM_PROMPT_ZS = """You classify Kyrgyz-language social media comments into exactly one of three labels:

- hate: attacks/dehumanises people for ethnicity, nationality, religion,
        sexual orientation, gender, or another protected group.
- offensive: profanity or insults that do NOT target a protected group.
- non_hate: neutral, supportive, or merely critical without insult.

Reply with EXACTLY one word: hate, offensive, or non_hate."""

SYSTEM_PROMPT_FS = SYSTEM_PROMPT_ZS + "\n\nHere are example labellings:"

_LABEL_RE = re.compile(r"\b(hate|offensive|non_hate|nonhate|non-hate)\b",
                       re.IGNORECASE)


def parse_label(raw: str) -> str:
    if not raw:
        return "non_hate"
    m = _LABEL_RE.search(raw)
    if not m:
        return "non_hate"
    tok = m.group(1).lower().replace("-", "_").replace("nonhate", "non_hate")
    return tok if tok in LABELS else "non_hate"


def build_messages(text: str, few_shots: list[tuple[str, str]] | None = None):
    msgs = [{"role": "system",
             "content": SYSTEM_PROMPT_FS if few_shots else SYSTEM_PROMPT_ZS}]
    for shot_text, shot_label in (few_shots or []):
        msgs.append({"role": "user",
                     "content": f"Comment: {shot_text}\nLabel:"})
        msgs.append({"role": "assistant", "content": shot_label})
    msgs.append({"role": "user", "content": f"Comment: {text}\nLabel:"})
    return msgs


def pick_few_shots(train_df, k_per_class: int, seed: int = 42):
    """Sample k examples per class from train, shuffled deterministically."""
    import random
    rng = random.Random(seed)
    shots = []
    for lab in LABELS:
        sub = train_df[train_df["label"] == lab]["text"].tolist()
        rng.shuffle(sub)
        for t in sub[:k_per_class]:
            shots.append((t, lab))
    rng.shuffle(shots)
    return shots


class HFCausalLM:
    """Thin wrapper around HF causal LM with deterministic single-token-ish decoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"loading {model_name} (dtype={dtype}) ...")
        torch_dtype = {"bfloat16": torch.bfloat16,
                       "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto",
        )
        self.model.eval()
        self.model_name = model_name

    def generate(self, messages, max_new_tokens: int = 8) -> str:
        import torch
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
