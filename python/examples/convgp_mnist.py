#!/usr/bin/env python3
"""CIP-0040 C7: MNIST with a convolutional first layer and inducing patches.

    PYTHONPATH=python python/.venv/bin/python python/examples/convgp_mnist.py DATA_DIR \
        [--train 10000] [--iters 1500] [--batch 128] [--inducing 100] [--stride 2] \
        [--layers 1] [--hidden 10]

DATA_DIR holds the four ``*-idx*-ubyte.gz`` files. One layer: 10 latent functions, each a
convolutional GP, with a robust-max likelihood. Two layers: the convolutional layer
feeds a ``hidden``-wide RBF layer, with the same likelihood.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gpc
from deepgp_mnist import metrics, read_idx

K = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--inducing", type=int, default=100)
    ap.add_argument("--patch", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=10)
    ap.add_argument("--weights", default="learned")
    ap.add_argument("--lengthscale", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.data_dir)
    rng = np.random.default_rng(a.seed)

    Xa = read_idx(d / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(float) / 255.0
    ya = read_idx(d / "train-labels-idx1-ubyte.gz").astype(int)
    Xte = read_idx(d / "t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(float)[: a.test] / 255.0
    yte = read_idx(d / "t10k-labels-idx1-ubyte.gz").astype(int)[: a.test]
    idx = rng.permutation(len(ya))[: a.train]
    Xtr, ytr = Xa[idx], ya[idx]

    conv = gpc.kern.Conv(
        gpc.kern.RBF(a.patch * a.patch, lengthscale=a.lengthscale),
        (28, 28, 1), (a.patch, a.patch), stride=a.stride, weights=a.weights,
    )
    kernel = conv if a.layers == 1 else [conv] + ["rbf"] * (a.layers - 1)
    m = gpc.models.DeepGP(
        Xtr, ytr, layers=a.layers, hidden=a.hidden, num_inducing=a.inducing,
        kernel=kernel, likelihood=gpc.likelihoods.RobustMax(K, eps=0.01),
    )
    print(f"MNIST conv GP: {len(ytr)} train, {len(yte)} test, layers={a.layers}, "
          f"patch={a.patch}, stride={a.stride} (P={conv.num_patches}), M={a.inducing}, "
          f"batch={a.batch}, iters={a.iters}, lr={a.lr}", flush=True)
    t0 = time.time()
    done = 0
    chunk = max(a.iters // 5, 1)
    while done < a.iters:
        n = min(chunk, a.iters - done)
        m.optimize(iters=n, batch_size=a.batch, adam_lr=a.lr, seed=a.seed + 1 + done)
        done += n
        sub = rng.permutation(len(yte))[:2000]
        acc, nll = metrics(m.predict_proba(Xte[sub]), yte[sub])
        print(f"  iter {done:>5}  test(2k) err {100*(1-acc):5.2f}%  NLL {nll:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)
    acc, nll = metrics(m.predict_proba(Xte), yte)
    print(f"final: test error {100*(1-acc):.2f}%  NLL {nll:.4f}  train {time.time()-t0:.0f}s")
    np.save("convgp_filters.npy", m.inducing_filters())


if __name__ == "__main__":
    main()
