"""Trains (1) linear 784->10 and (2) MLP 784-16-16-10, pure numpy. Run once: python train.py"""
import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = (X / 255).astype(np.float32); y = y.astype(int)
Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
rng = np.random.default_rng(0)

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def forward(Ws, bs, x, relu):
    a = [x]
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = a[-1] @ W.T + b
        a.append(np.maximum(z, 0) if relu and k < len(Ws) - 1 else z)
    return a

def fit(sizes, relu, epochs, lr):
    Ws = [rng.normal(0, np.sqrt(2 / i), (o, i)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(o, np.float32) for o in sizes[1:]]
    for ep in range(epochs):
        p = rng.permutation(60000)
        for s in range(0, 60000, 128):
            idx = p[s:s + 128]
            a = forward(Ws, bs, Xtr[idx], relu)
            d = softmax(a[-1]); d[np.arange(len(idx)), ytr[idx]] -= 1; d /= len(idx)
            for k in reversed(range(len(Ws))):
                gW, gb = d.T @ a[k], d.sum(0)
                if k > 0: d = (d @ Ws[k]) * (a[k] > 0)
                Ws[k] -= lr * gW; bs[k] -= lr * gb
        acc = (forward(Ws, bs, Xte, relu)[-1].argmax(1) == yte).mean()
        print(f"{sizes} epoch {ep + 1}: test acc {acc:.3f}")
    return Ws, bs

(LW,), (Lb,) = fit([784, 10], False, 10, 0.1)
Ws, bs = fit([784, 16, 16, 10], True, 20, 0.1)
np.savez("model.npz", L_W=LW, L_b=Lb, W1=Ws[0], b1=bs[0], W2=Ws[1], b2=bs[1], W3=Ws[2], b3=bs[2])
