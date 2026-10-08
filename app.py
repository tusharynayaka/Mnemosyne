import numpy as np
from flask import Flask, request, jsonify

m = np.load("model.npz")
app = Flask(__name__, static_folder="static", static_url_path="/static")

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

def center(img):  # MNIST digits are centred by centre of mass
    s = img.sum()
    if s == 0: return img
    ys, xs = np.indices(img.shape)
    cy, cx = (ys * img).sum() / s, (xs * img).sum() / s
    return np.roll(img, (round(13.5 - cy), round(13.5 - cx)), (0, 1))

@app.route("/")
def home(): return app.send_static_file("index.html")

@app.route("/model")
def model():
    return jsonify(
        L_W=m["L_W"].tolist(), L_b=m["L_b"].tolist(),
        W1=m["W1"].tolist(), b1=m["b1"].tolist(),
        W2=m["W2"].tolist(), b2=m["b2"].tolist(),
        W3=m["W3"].tolist(), b3=m["b3"].tolist(),
        LS_W=m["LS_W"].tolist(), LS_b=m["LS_b"].tolist(),
        mu=m["mu"].tolist(), pinv=m["pinv"].tolist(),
        PC=m["PC"].tolist(), pts=m["pts"].tolist(), lab=m["lab"].tolist(),
        rank=int(m["rank"]), acc=m["acc"].tolist(), var2=float(m["var2"])
    )

@app.route("/predict", methods=["POST"])
def predict():
    x = center(np.array(request.json["px"], np.float32).reshape(28, 28)).ravel()
    lin = softmax(m["L_W"] @ x + m["L_b"])            # y = softmax(Wx + b)
    a1 = np.maximum(m["W1"] @ x + m["b1"], 0)          # ReLU(W1 x + b1)
    a2 = np.maximum(m["W2"] @ a1 + m["b2"], 0)         # ReLU(W2 a1 + b2)
    out = softmax(m["W3"] @ a2 + m["b3"])              # softmax(W3 a2 + b3)
    n = lambda a: (a / (a.max() or 1)).tolist()
    ls = m["LS_W"] @ x + m["LS_b"]                      # least-squares model
    pc = (x - m["mu"]) @ m["PC"][:2].T                   # coordinates along top 2 eigenvectors
    z = np.random.default_rng(0).normal(size=784)        # null space: W n = 0
    nv = z - m["pinv"] @ (m["L_W"] @ z); nv = (0.4 * nv / np.abs(nv).max()).astype(np.float32)
    nd = float(np.abs(m["L_W"] @ (x + nv) - m["L_W"] @ x).max())
    return jsonify(lin=lin.tolist(), h1=n(a1), h2=n(a2), out=out.tolist(),
                   ls=ls.tolist(), pc=pc.tolist(), n=nv.tolist(), nd=nd)

if __name__ == "__main__":
    app.run(port=5000)
