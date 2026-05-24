"""Experiment 4 — TF-IDF char 3-5grams + Linear SVM."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..paths import RNG_SEED
from ..preprocess import normalise_full
from .base import Experiment


class ExpTfidfSvm(Experiment):
    name = "exp4_tfidf_svm"
    family = "classical"
    description = "TF-IDF char 3-5grams + Linear SVM (calibrated)"

    def run(self) -> dict:
        train, _val, test = self.load_splits()

        X_train = [normalise_full(t) for t in train["text"]]
        X_test = [normalise_full(t) for t in test["text"]]
        y_train = train["label"].tolist()
        y_test = test["label"].tolist()

        # Calibrate so we get probabilities consistent with other experiments.
        base_svm = LinearSVC(class_weight="balanced", C=1.0,
                             random_state=RNG_SEED, max_iter=5000)
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5), min_df=2, max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)

        return self.evaluate(
            y_test, y_pred, ids=test["id"], texts=test["text"], probs=probs,
        )


def main() -> None:
    ExpTfidfSvm().run_logged()


if __name__ == "__main__":
    main()
