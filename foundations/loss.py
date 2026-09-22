import numpy as np
from numpy.typing import NDArray


class Solution:
    clip = lambda x: np.clip(x, 1e-7, 1 - 1e-7)

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        y_pred = Solution.clip(y_pred)
        L = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return round (L, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        y_pred = Solution.clip(y_pred)
        L = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return round (L, 4)
