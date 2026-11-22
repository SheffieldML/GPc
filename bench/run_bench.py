#!/usr/bin/env python3
"""CIP-0013: GPy-referenced bench harness with log-likelihood agreement gate.

Generates shared protocol data, runs GPc ``benchGPc`` and a matching GPy model,
compares log-likelihoods (plus a dense NumPy full-GP oracle), and prints timing
ratios only when the gate passes.

Protocol (cip0013.md): q=2, X ~ U[-3,3]^2, y = sum_q sin(x_q) + eps,
RBF(variance=1, lengthscale=1) + white/noise variance 0.01; sparse beta=100;
shared inducing subset; OPENBLAS_NUM_THREADS=1. Timing loops perturb
parameters between repeats (and include setOptParams / optimizer_array apply)
so caches cannot reuse factorisations; reported LL is at the base parameters.

Usage:
  python bench/run_bench.py [--kinds ftc,dtcvar,fitc] [--N 1000] [--M 50]
                            [--benchGPc ./benchGPc] [--out bench.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
LL_ATOL = 1e-4
LL_RTOL = 1e-5
# GPy sparse jitter can move LL by a fraction of a nat (cip0013.md).
GPY_SPARSE_ATOL = 0.5


def protocol_data(N: int, q: int, M: int, seed: int = 13):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(N, q))
    y = np.sin(X).sum(axis=1, keepdims=True) + rng.normal(0.0, 0.1, size=(N, 1))
    idx = np.sort(rng.choice(N, size=M, replace=False))
    Xu = X[idx].copy()
    return X, y, Xu


def write_data(data_dir: Path, X, y, Xu) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(data_dir / "X.txt", X)
    np.savetxt(data_dir / "y.txt", y)
    np.savetxt(data_dir / "Xu.txt", Xu)


def dense_rbf_white_ll(X, y, lengthscale=1.0, variance=1.0, white=0.01) -> float:
    """Independent dense GP log-likelihood (RBF + white diagonal)."""
    N = X.shape[0]
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = variance * np.exp(-0.5 * d2 / (lengthscale**2))
    K.flat[:: N + 1] += white
    # Cholesky of K; LL = -0.5 (y^T K^{-1} y + log|K| + N log 2π)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    quad = float(np.asarray(y.T @ alpha).reshape(-1)[0])
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (quad + logdet + N * math.log(2.0 * math.pi))


def ll_agree(a: float, b: float, n: int, sparse: bool) -> tuple[bool, str]:
    """Return (ok, reason). Allows FITC ±(N/2)log(2π) (CIP-0012 F5)."""
    if math.isfinite(a) and math.isfinite(b) and (
        abs(a - b) <= max(LL_ATOL, LL_RTOL * max(abs(a), abs(b), 1.0))
    ):
        return True, "match"
    two_pi = n * HALF_LOG_2PI
    if abs(abs(a - b) - two_pi) <= max(LL_ATOL, 1e-6 * two_pi):
        return True, "fitc_2pi_constant"
    atol = GPY_SPARSE_ATOL if sparse else LL_ATOL
    if abs(a - b) <= atol:
        return True, "sparse_jitter_tol"
    return False, f"disagree |{a}-{b}|={abs(a-b):.6g}"


def parse_bench_line(line: str) -> dict:
    out = {}
    for tok in line.strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def run_gpc(bench: Path, kind: str, data_dir: Path, N: int, q: int, M: int, reps: int) -> dict:
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [
        str(bench),
        "--kind",
        kind,
        "--data-dir",
        str(data_dir),
        "--N",
        str(N),
        "--q",
        str(q),
        "--M",
        str(M),
        "--reps",
        str(reps),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"benchGPc failed ({proc.returncode}): {proc.stderr.strip()}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("kind=")]
    if not lines:
        raise RuntimeError(f"benchGPc produced no kind= line; stdout={proc.stdout!r}")
    return parse_bench_line(lines[-1])


def run_gpy(kind: str, X, y, Xu, reps: int, warm: int = 2):
    import GPy
    from GPy.inference.latent_function_inference import FITC, VarDTC

    q = X.shape[1]
    kern = GPy.kern.RBF(q, variance=1.0, lengthscale=1.0) + GPy.kern.White(
        q, variance=0.01
    )
    if kind == "ftc":
        model = GPy.models.GPRegression(X, y, kern)
        # Likelihood noise sits on top of White; freeze it near zero so the
        # white kernel owns sigma_n^2=0.01 (matches GPc rbf+white).
        model.Gaussian_noise.variance = 1e-9
        model.Gaussian_noise.variance.fix()
    elif kind == "dtcvar":
        model = GPy.models.SparseGPRegression(X, y, kernel=kern, Z=Xu.copy())
        model.inference_method = VarDTC()
        model.Gaussian_noise.variance = 1.0 / 100.0  # beta=100
        model.Gaussian_noise.variance.fix()
    elif kind == "fitc":
        model = GPy.models.SparseGPRegression(X, y, kernel=kern, Z=Xu.copy())
        model.inference_method = FITC()
        model.Gaussian_noise.variance = 1.0 / 100.0
        model.Gaussian_noise.variance.fix()
    else:
        raise ValueError(kind)

    x0 = np.array(model.optimizer_array, dtype=float, copy=True)

    def _perturb(rep: int) -> np.ndarray:
        h = 1e-6 * (1.0 + np.abs(x0)) * (1.0 + 0.01 * (rep + 1))
        sign = np.where(((rep + np.arange(x0.size)) % 2) == 0, 1.0, -1.0)
        return x0 + sign * h

    # Warm-up with the same cache-busting pattern as the timed loop.
    for w in range(warm):
        model._objective_grads(_perturb(w))
    times = []
    for r in range(reps):
        x = _perturb(warm + r)
        t0 = time.perf_counter()
        _ll, _grad = model._objective_grads(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    # Gate LL at unperturbed protocol parameters (GPy objective is -LL).
    ll, _grad = model._objective_grads(x0)
    return {
        "ll": float(-ll),
        "median_ms": float(np.median(times)),
        "nparams": int(x0.size),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kinds", default="ftc,dtcvar,fitc")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--benchGPc", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("bench.csv"))
    ap.add_argument("--skip-gpy", action="store_true")
    args = ap.parse_args(argv)

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    repo = Path(__file__).resolve().parents[1]
    if args.benchGPc is None:
        args.benchGPc = repo / "benchGPc"
    args.benchGPc = args.benchGPc.resolve()
    if not args.benchGPc.exists():
        print(f"error: {args.benchGPc} not found; build with `make RELEASE=1 bench`", file=sys.stderr)
        return 2

    X, y, Xu = protocol_data(args.N, args.q, args.M, args.seed)
    dense_ll = dense_rbf_white_ll(X, y)

    rows = []
    with tempfile.TemporaryDirectory(prefix="gpc-bench-") as tmp:
        data_dir = Path(tmp)
        write_data(data_dir, X, y, Xu)

        for kind in kinds:
            sparse = kind != "ftc"
            gpc = run_gpc(args.benchGPc, kind, data_dir, args.N, args.q, args.M, args.reps)
            gpc_ll = float(gpc["ll"])

            gate_ok = True
            gate_notes = []

            if kind == "ftc":
                ok, why = ll_agree(gpc_ll, dense_ll, args.N, sparse=False)
                gate_notes.append(f"gpc_vs_numpy={why}")
                gate_ok = gate_ok and ok

            gpy = None
            if not args.skip_gpy:
                try:
                    gpy = run_gpy(kind, X, y, Xu, args.reps)
                    ok, why = ll_agree(gpc_ll, gpy["ll"], args.N, sparse=sparse)
                    gate_notes.append(f"gpc_vs_gpy={why}")
                    gate_ok = gate_ok and ok
                    if kind == "ftc":
                        ok2, why2 = ll_agree(gpy["ll"], dense_ll, args.N, sparse=False)
                        gate_notes.append(f"gpy_vs_numpy={why2}")
                        gate_ok = gate_ok and ok2
                except Exception as exc:  # noqa: BLE001 — report and fail the gate
                    gate_ok = False
                    gate_notes.append(f"gpy_error={exc}")

            ratio = ""
            gpy_ms = ""
            gpy_ll = ""
            if gpy is not None and gate_ok and gpy["median_ms"] > 0:
                ratio = f"{float(gpc['median_ms']) / gpy['median_ms']:.3f}"
                gpy_ms = f"{gpy['median_ms']:.4f}"
                gpy_ll = f"{gpy['ll']:.10g}"

            row = {
                "kind": kind,
                "N": args.N,
                "q": args.q,
                "M": 0 if kind == "ftc" else args.M,
                "gpc_ll": f"{gpc_ll:.10g}",
                "gpy_ll": gpy_ll,
                "numpy_ll": f"{dense_ll:.10g}" if kind == "ftc" else "",
                "gpc_median_ms": f"{float(gpc['median_ms']):.4f}" if gate_ok else "WITHHELD",
                "gpy_median_ms": gpy_ms if gate_ok else ("WITHHELD" if gpy else ""),
                "gpc_over_gpy": ratio if gate_ok else "WITHHELD",
                "gate": "pass" if gate_ok else "fail",
                "gate_notes": ";".join(gate_notes),
                "gpc_flags": gpc.get("flags", ""),
                "gpc_checks": gpc.get("gpc_checks", ""),
                "threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
            }
            rows.append(row)
            status = "PASS" if gate_ok else "FAIL"
            print(
                f"[{status}] {kind}: gpc_ll={gpc_ll:.6g}"
                + (f" gpy_ll={gpy['ll']:.6g}" if gpy else "")
                + (f" gpc_ms={gpc['median_ms']}" if gate_ok else " timing withheld")
                + (f" ratio={ratio}" if ratio else "")
                + f" ({';'.join(gate_notes)})"
            )

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0 if all(r["gate"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
