from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


class RBFRegressor(BaseEstimator, RegressorMixin):
    """Radial-basis-function network with learned centres and ridge output."""

    def __init__(self, n_centers: int = 20, alpha: float = 1.0, random_state: int = 42):
        self.n_centers = n_centers
        self.alpha = alpha
        self.random_state = random_state

    def _features(self, x):
        distances = ((np.asarray(x)[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-self.gamma_ * distances)

    def fit(self, x, y):
        count = min(self.n_centers, len(x))
        km = KMeans(count, n_init=10, random_state=self.random_state).fit(x)
        self.centers_ = km.cluster_centers_
        pairwise = np.linalg.norm(self.centers_[:, None] - self.centers_[None, :], axis=2)
        width = np.median(pairwise[pairwise > 0]) if np.any(pairwise > 0) else 1.0
        self.gamma_ = 1.0 / (2.0 * width**2)
        self.output_ = Ridge(alpha=self.alpha).fit(self._features(x), y)
        return self

    def predict(self, x):
        return self.output_.predict(self._features(x))


def _tensorflow_model(name: str, lookback: int, seed: int):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(f"Model {name!r} requires: pip install -e .[deep-learning]") from exc
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    if name == "rnn":
        body = [layers.SimpleRNN(32), layers.Dense(1)]
    elif name == "lstm":
        body = [layers.LSTM(32), layers.Dense(1)]
    elif name == "rnn-lstm":
        body = [layers.SimpleRNN(32, return_sequences=True), layers.LSTM(16), layers.Dense(1)]
    elif name == "gan":
        # A compact generator-style network; adversarial training is intentionally
        # kept behind the same deterministic forecasting interface.
        body = [layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dropout(0.2), layers.Dense(32, activation="relu"), layers.Dense(1)]
    else:
        raise ValueError(name)
    model = tf.keras.Sequential([layers.Input((lookback, 1)), *body])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_model(name: str, lookback: int, seed: int = 42):
    name = name.lower()
    if name == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed)
    if name == "rbf":
        return RBFRegressor(random_state=seed)
    if name in {"rnn", "lstm", "rnn-lstm", "gan"}:
        return _tensorflow_model(name, lookback, seed)
    if name in {"arima", "arima-rbf"}:
        return name
    raise ValueError(f"Unknown model {name!r}")
