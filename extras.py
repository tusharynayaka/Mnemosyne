"""Run once after train.py: adds least squares (pseudo-inverse), PCA/SVD and rank to model.npz"""
import numpy as np
from sklearn.datasets import fetch_openml

m = dict(np.load("model.npz"))
X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = (X / 255).astype(np.float32); y = y.astype(int)
Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]

# least squares: minimise |[X 1] B - Y|^2, solved with the pseudo-inverse
Xb = np.hstack([Xtr, np.ones((60000, 1), np.float32)])
B = np.linalg.lstsq(Xb, np.eye(10, dtype=np.float32)[ytr], rcond=None)[0]
m["LS_W"], m["LS_b"] = B[:-1].T, B[-1]

# PCA = SVD of the centred data matrix; rows of Vt are eigenvectors of the covariance matrix
mu = Xtr.mean(0)
_, S, Vt = np.linalg.svd(Xtr[:10000] - mu, full_matrices=False)
idx = np.random.default_rng(0).choice(10000, 1500, replace=False)
m.update(mu=mu, PC=Vt[:6], pts=(Xtr[:10000][idx] - mu) @ Vt[:2].T, lab=ytr[:10000][idx],
         var2=(S[:2] ** 2).sum() / (S ** 2).sum())

# rank of the trained linear map and its pseudo-inverse (used to build null-space vectors)
m["rank"] = np.linalg.matrix_rank(m["L_W"]); m["pinv"] = np.linalg.pinv(m["L_W"])
m["acc"] = np.array([((Xte @ m["L_W"].T + m["L_b"]).argmax(1) == yte).mean(),
                     ((Xte @ B[:-1] + B[-1]).argmax(1) == yte).mean()])
np.savez("model.npz", **m)
print("accuracy trained / least squares:", m["acc"], " rank:", m["rank"], " var in 2 PCs:", m["var2"])
