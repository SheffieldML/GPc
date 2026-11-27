#!/usr/bin/env python3
"""CIP-003E Phase 7: MNIST deep GP, Gaussian one-hot baseline versus multi-class likelihoods.

Reduced setting (PCA inputs, a training subset) so that it runs in minutes. It is not the
full-MNIST experiment and says nothing about MLP or conv-net baselines.

    PYTHONPATH=python python/.venv/bin/python python/examples/deepgp_mnist.py DATA_DIR \
        [--train 5000] [--pca 30] [--iters 600] [--batch 500] [--inducing 100]

DATA_DIR holds the four ``*-idx*-ubyte.gz`` files (train/t10k images and labels).
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gpc

K = 10


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    ndim = buf[3]
    shape = tuple(int.from_bytes(buf[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def metrics(prob: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    prob = np.clip(prob, 1e-12, 1.0)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return (
        float(np.mean(np.argmax(prob, axis=1) == y)),
        float(-np.mean(np.log(prob[np.arange(len(y)), y]))),
    )


def gaussian_proba(mu, var, rng, draws: int = 500) -> np.ndarray:
    """P(class k) = P(f_k largest) under independent N(mu_k, var_k), by Monte Carlo."""
    counts = np.zeros(mu.shape)
    for _ in range(draws):
        f = mu + np.sqrt(var) * rng.standard_normal(mu.shape)
        counts[np.arange(len(mu)), np.argmax(f, axis=1)] += 1.0
    return counts / draws


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--pca", type=int, default=30)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--inducing", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--only", default="", help="comma-separated config names to run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.data_dir)
    rng = np.random.default_rng(a.seed)

    Xtr_all = read_idx(d / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(float) / 255.0
    ytr_all = read_idx(d / "train-labels-idx1-ubyte.gz").astype(int)
    Xte = read_idx(d / "t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(float)[: a.test] / 255.0
    yte = read_idx(d / "t10k-labels-idx1-ubyte.gz").astype(int)[: a.test]
    idx = rng.permutation(len(ytr_all))[: a.train]
    Xtr, ytr = Xtr_all[idx], ytr_all[idx]

    mean = Xtr.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Xtr - mean, full_matrices=False)
    W = Vt[: a.pca].T / (sv[: a.pca] / np.sqrt(len(Xtr)))  # whitened PCA scores
    Ztr, Zte = (Xtr - mean) @ W, (Xte - mean) @ W

    configs = {
        "gaussian": None,
        "robustmax e=.01": gpc.likelihoods.RobustMax(K, eps=0.01),
        "robustmax e=.10": gpc.likelihoods.RobustMax(K, eps=0.10),
        "softmax": gpc.likelihoods.Softmax(K, num_samples=64),
    }
    if a.only:
        configs = {k: v for k, v in configs.items() if k in a.only.split(",")}
    print(f"MNIST: {len(ytr)} train, {len(yte)} test, PCA {a.pca}, M={a.inducing}, "
          f"hidden={a.hidden}, batch={a.batch}, iters={a.iters}, Adam lr={a.lr}")
    print(f"{'likelihood':<18}{'test acc':>10}{'test NLL':>10}{'train s':>10}")
    for name, lik in configs.items():
        kw = dict(layers=2, hidden=a.hidden, num_inducing=a.inducing, kernel="rbf")
        t0 = time.time()
        if lik is None:
            m = gpc.models.DeepGP(Ztr, 2.0 * np.eye(K)[ytr] - 1.0, beta=4.0, **kw)
        else:
            m = gpc.models.DeepGP(Ztr, ytr, likelihood=lik, **kw)
        m.optimize(iters=a.iters, batch_size=a.batch, adam_lr=a.lr, seed=a.seed + 1)
        dt = time.time() - t0
        if lik is None:
            mu, var = m.predict(Zte)
            p = gaussian_proba(mu, var, np.random.default_rng(7))
        else:
            p = m.predict_proba(Zte)
        acc, nll = metrics(p, yte)
        print(f"{name:<18}{acc:>10.4f}{nll:>10.4f}{dt:>10.0f}", flush=True)


if __name__ == "__main__":
    main()
