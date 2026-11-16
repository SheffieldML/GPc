#!/usr/bin/env python3
"""CIP-0012 mpmath reference for whitened sparse StableFactor gates.

Reads a diagDtcvar-style state dump and recomputes the whitened DTCVAR
log-likelihood at high precision (adaptive Cholesky jitter, matching
``scripts/diag_dtcvar_compare.py``).

Usage:
  make diagDtcvar && ./diagDtcvar
  python scripts/sparse_mp_reference.py diag_dtcvar_state.txt --approx dtcvar

Pin: ``mpmath`` (maintainer gate; optional CI later). See docs/numerics.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_dump(path: Path) -> dict:
    lines = path.read_text().splitlines()
    data: dict = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        key = parts[0]
        if key in {"approx", "N", "q", "d", "M", "inverseWidth", "variance", "beta", "ll"}:
            if key == "approx":
                data[key] = parts[1]
            elif key in {"N", "q", "d", "M"}:
                data[key] = int(float(parts[1]))
            else:
                data[key] = float(parts[1])
            i += 1
        elif key in {"X", "y", "Xu"}:
            i += 1
            n = int(data["N"] if key != "Xu" else data["M"])
            rows = []
            for _ in range(n):
                rows.append([float(x) for x in lines[i].split()])
                i += 1
            data[key] = rows
        elif key == "grads_ll":
            n = int(parts[1])
            i += 1
            data["grads"] = [float(lines[i + j]) for j in range(n)]
            i += n
        else:
            i += 1
    return data


def cholesky_jit(mp, A, jitter0=None):
    """Lower Cholesky with escalating diagonal jitter (diag_dtcvar_compare spirit)."""
    n = A.rows
    if jitter0 is None:
        jitter0 = mp.mpf("1e-8")
    for attempt in range(10):
        Aj = A.copy()
        jit = jitter0 * (mp.mpf(10) ** attempt if attempt else mp.mpf(1))
        for i in range(n):
            Aj[i, i] += jit
        try:
            return mp.cholesky(Aj)
        except Exception:
            continue
    raise RuntimeError("cholesky_jit failed")


def whitened_dtcvar_ll(mp, X, y, Xu, gamma, var, beta):
    """Whitened DTCVAR LL matching CGp StableFactor (spherical, d=1, no prior)."""
    N, M = len(X), len(Xu)
    q = len(X[0])
    gamma = mp.mpf(gamma)
    var = mp.mpf(var)
    beta = mp.mpf(beta)

    def rbf(A, B):
        rows = []
        for i in range(len(A)):
            row = []
            for j in range(len(B)):
                d2 = mp.mpf(0)
                for t in range(q):
                    d = mp.mpf(A[i][t]) - mp.mpf(B[j][t])
                    d2 += d * d
                row.append(var * mp.e ** (-mp.mpf("0.5") * gamma * d2))
            rows.append(row)
        return mp.matrix(rows)

    Kuu = rbf(Xu, Xu)
    Kuf = rbf(Xu, X)
    kff = [var] * N
    m = [mp.mpf(y[i][0] if isinstance(y[i], list) else y[i]) for i in range(N)]

    L = cholesky_jit(mp, Kuu)  # Kuu ≈ L L^T
    # B = L^{-1} Kuf  (equiv. U^{-T} Kuf with U = L^T)
    B = mp.matrix(M, N)
    for j in range(N):
        col = mp.matrix([Kuf[i, j] for i in range(M)])
        sol = mp.lu_solve(L, col)
        for i in range(M):
            B[i, j] = sol[i]

    C = mp.eye(M)
    for i in range(M):
        for k in range(M):
            s = mp.mpf(0)
            for j in range(N):
                s += B[i, j] * B[k, j]
            C[i, k] += beta * s
    Lc = cholesky_jit(mp, C)

    u = mp.matrix([sum(B[i, j] * m[j] for j in range(N)) for i in range(M)])
    # C ≈ Lc Lc^T; a = C^{-1} u
    a = mp.lu_solve(Lc.T, mp.lu_solve(Lc, u))

    qnorm = [sum(B[i, j] * B[i, j] for i in range(M)) for j in range(N)]
    logC = mp.mpf(0)
    for i in range(M):
        logC += mp.log(Lc[i, i])
    logC *= 2

    L2 = (
        logC
        - N * mp.log(beta)
        + beta * sum(mj * mj for mj in m)
        - beta * beta * sum(u[i] * a[i] for i in range(M))
        + beta * sum(kff[j] - qnorm[j] for j in range(N))
    )
    half = mp.mpf("0.5")
    return -half * L2 - N * half * mp.log(2 * mp.pi)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("state", nargs="?", help="diagDtcvar-style state dump")
    p.add_argument("--dps", type=int, default=60, help="mpmath decimal precision")
    p.add_argument("--approx", default="dtcvar", choices=["dtc", "dtcvar"])
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="|ll_dump - ll_mp| gate (LAPACK vs pure-Python chol ~1e-5)",
    )
    args = p.parse_args(argv)

    try:
        import mpmath as mp
    except ImportError:
        print("mpmath is required: pip install mpmath", file=sys.stderr)
        return 2

    mp.mp.dps = args.dps
    if not args.state:
        print(f"sparse_mp_reference: scaffold OK (mpmath dps={args.dps}).")
        return 0

    if args.approx != "dtcvar":
        print("Only --approx dtcvar is implemented in this scaffold.", file=sys.stderr)
        return 1

    data = parse_dump(Path(args.state))
    y = data["y"]
    ll_mp = whitened_dtcvar_ll(
        mp,
        data["X"],
        y,
        data["Xu"],
        data["inverseWidth"],
        data["variance"],
        data["beta"],
    )
    ll_dump = mp.mpf(data["ll"])
    diff = abs(ll_mp - ll_dump)
    print(f"ll_dump={ll_dump}")
    print(f"ll_mpmath_whitened={ll_mp}")
    print(f"abs_diff={diff}")
    ok = diff < args.tol
    print("PASS" if ok else "FAIL", f"(tol={args.tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
