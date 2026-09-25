import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        #forward
        z1 = np.dot(x, np.transpose(W1)) + b1
        a1 = np.maximum(0.0, z1) 
        z2 = np.dot(a1, np.transpose(W2)) + b2
        L = np.mean(np.square(z2 - y_true))

        #backward
        db2 = 2 * (z2 - y_true) # n = 1
        dW2 = np.outer(db2, a1)
        db1 = np.dot(db2, W2) * (z1 > 0)
        dW1 = np.outer(db1, x)

        result = {
            'loss': np.round(L, 4),
            'dW1': np.round(dW1, 4),
            'db1': np.round(db1, 4),
            'dW2': np.round(dW2, 4),
            'db2': np.round(db2, 4)
        }
        return result



