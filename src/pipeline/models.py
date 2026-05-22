from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class RegistryEntry:
    build: Callable[[int], Any]
    families: frozenset[str]  # subset of {"linear", "probabilistic"}


MODEL_REGISTRY: dict[str, RegistryEntry] = {
    "logistic_regression": RegistryEntry(
        build=lambda seed: LogisticRegression(random_state=seed, max_iter=1000, n_jobs=1),
        families=frozenset({"linear", "probabilistic"}),
    ),
    "linear_svm": RegistryEntry(
        build=lambda seed: LinearSVC(random_state=seed),
        families=frozenset({"linear"}),
    ),
    "naive_bayes": RegistryEntry(
        build=lambda seed: MultinomialNB(),
        families=frozenset({"probabilistic"}),
    ),
    "ridge_classifier": RegistryEntry(
        build=lambda seed: RidgeClassifier(random_state=seed),
        families=frozenset({"linear"}),
    ),
    "sgd_classifier": RegistryEntry(
        build=lambda seed: SGDClassifier(random_state=seed, loss="log_loss"),
        families=frozenset({"linear", "probabilistic"}),
    ),
}
