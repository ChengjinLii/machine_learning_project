from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class RBFNetworkClassifier:
    """A compact RBF network for multiclass classification."""

    def __init__(
        self,
        n_centers: int = 18,
        alpha: float = 1e-3,
        random_state: int = 42,
        n_init: int = 20,
    ) -> None:
        self.n_centers = n_centers
        self.alpha = alpha
        self.random_state = random_state
        self.n_init = n_init
        self.classes_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self.gamma_: float | None = None
        self.weights_: np.ndarray | None = None
        self.kmeans_: KMeans | None = None

    def _compute_design_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.centers_ is None or self.gamma_ is None:
            raise RuntimeError("Model has not been fitted.")
        sq_dist = cdist(X, self.centers_, metric="sqeuclidean")
        phi = np.exp(-self.gamma_ * sq_dist)
        bias = np.ones((phi.shape[0], 1), dtype=phi.dtype)
        return np.hstack([bias, phi])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFNetworkClassifier":
        self.classes_ = np.unique(y)
        self.kmeans_ = KMeans(
            n_clusters=self.n_centers,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.kmeans_.fit(X)
        self.centers_ = self.kmeans_.cluster_centers_

        center_distances = cdist(self.centers_, self.centers_, metric="euclidean")
        np.fill_diagonal(center_distances, np.nan)
        nearest_distances = np.nanmin(center_distances, axis=1)
        sigma = float(np.nanmean(nearest_distances))
        sigma = max(sigma, 1e-6)
        self.gamma_ = 1.0 / (2.0 * sigma * sigma)

        phi = self._compute_design_matrix(X)
        y_index = np.searchsorted(self.classes_, y)
        one_hot = np.eye(len(self.classes_), dtype=float)[y_index]

        regularizer = self.alpha * np.eye(phi.shape[1], dtype=float)
        self.weights_ = np.linalg.solve(phi.T @ phi + regularizer, phi.T @ one_hot)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        phi = self._compute_design_matrix(X)
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted.")
        return phi @ self.weights_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted.")
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
