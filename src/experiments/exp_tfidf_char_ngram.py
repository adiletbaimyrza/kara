"""Experiment 3 — TF-IDF character n-grams + LogReg.

Kyrgyz is morphologically rich (agglutinative suffixation). Character n-grams
should partially recover stems across inflected forms, which word n-grams miss
on a small dataset. RQ2: do char-ngrams help?
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..paths import RNG_SEED
from ..preprocess import normalise_full
from .base import Experiment


class ExpTfidfCharNgram(Experiment):
    name = "exp3_tfidf_char_ngram"
    family = "classical"
    description = "TF-IDF char 3-5grams + LogReg, full preprocessing"

    def run(self) -> dict:
        train, _val, test = self.load_splits()

        X_train = [normalise_full(t) for t in train["text"]]
        X_test = [normalise_full(t) for t in test["text"]]
        y_train = train["label"].tolist()
        y_test = test["label"].tolist()

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5), min_df=2, max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RNG_SEED, n_jobs=-1,
            )),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        return self.evaluate(
            y_test, y_pred, ids=test["id"], texts=test["text"], probs=probs,
        )


def main() -> None:
    ExpTfidfCharNgram().run_logged()


if __name__ == "__main__":
    main()
