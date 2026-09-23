import numpy as np


def init_params():
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2 / 4), size=(4, 8))
    b1 = np.zeros(8)
    W2 = rng.normal(0.0, np.sqrt(2 / (8 + 3)), size=(8, 3))
    b2 = np.zeros(3)
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ W2 + b2

    cache = (X, Z1, A1, Z2)
    return Z2, cache


def cross_entropy_loss(logits, y):
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp

    n = logits.shape[0]
    loss = -log_probs[np.arange(n), y].mean()
    return loss, log_probs


def backward(y, W2, cache, log_probs, divide_by_n=True):
    X, Z1, A1, Z2 = cache
    n = X.shape[0]

    probs = np.exp(log_probs)
    dZ2 = probs.copy()
    dZ2[np.arange(n), y] -= 1.0
    if divide_by_n:
        dZ2 /= n

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return dW1, db1, dW2, db2
