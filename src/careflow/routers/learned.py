"""Learned pre-generation routers over a fixed set of paths.

A router sees only the context (before any reply is generated) and picks a path. Training targets are the
per-context head-to-head scores vs the reference path (0 / 0.5 / 1, both presentation orders averaged).
Train and test use *different dialogues* (ESConv valid -> test), so the winner's-curse issue of the oracle
(selection and evaluation on the same judgments) does not arise.

Routers:
- UtilityRouter: one ridge regressor per path predicts its score; choose argmax(score - lam * cost).
  Sweeping lam traces a quality/cost frontier.
- ClusterRouter: k-means on context embeddings; each cluster routes to the path with the best mean training
  score minus lam * cost. This is the geometric idea of GCUL (Fan et al., arXiv:2609.29327) — ambiguity lives in
  compact regions of representation space — applied to routing instead of rejection.
- random_policy: route a share of contexts uniformly at random to each path (budget-matched baseline).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge


class UtilityRouter:
    def __init__(self, paths: Sequence[str], alpha: float = 10.0) -> None:
        self.paths = list(paths)
        self.models = {p: Ridge(alpha=alpha) for p in self.paths}

    def fit(self, x: np.ndarray, scores: dict[str, np.ndarray]) -> UtilityRouter:
        for p in self.paths:
            self.models[p].fit(x, scores[p])
        return self

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self.models[p].predict(x) for p in self.paths], axis=1)

    def route(self, x: np.ndarray, cost: dict[str, float], lam: float) -> list[str]:
        util = self.predict_scores(x) - lam * np.array([cost[p] for p in self.paths])
        return [self.paths[i] for i in util.argmax(axis=1)]


class ClusterRouter:
    def __init__(self, paths: Sequence[str], k: int, seed: int) -> None:
        self.paths, self.k, self.seed = list(paths), k, seed

    def fit(self, x: np.ndarray, scores: dict[str, np.ndarray]) -> ClusterRouter:
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(x)
        lab = self.km.labels_
        glob = np.array([scores[p].mean() for p in self.paths])
        self.means = np.zeros((self.k, len(self.paths)))
        for c in range(self.k):
            m = lab == c
            # shrink small clusters towards the global mean (5 pseudo-observations)
            self.means[c] = (np.array([scores[p][m].sum() for p in self.paths]) + 5 * glob) / (m.sum() + 5)
        return self

    def route(self, x: np.ndarray, cost: dict[str, float], lam: float) -> list[str]:
        util = self.means[self.km.predict(x)] - lam * np.array([cost[p] for p in self.paths])
        return [self.paths[i] for i in util.argmax(axis=1)]


def evaluate(choice: Sequence[str], scores: dict[str, np.ndarray], cost: dict[str, float]) -> tuple[float, float]:
    """Mean score of the chosen path per context and mean cost."""
    q = np.array([scores[p][i] for i, p in enumerate(choice)])
    return float(q.mean()), float(np.mean([cost[p] for p in choice]))


def random_policy(n: int, shares: dict[str, float], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    paths = list(shares)
    return list(rng.choice(paths, size=n, p=np.array([shares[p] for p in paths]) / sum(shares.values())))
