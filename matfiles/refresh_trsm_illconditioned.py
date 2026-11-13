#!/usr/bin/env python3
"""Refresh ill-conditioned TRSM fixtures against modern LAPACK (via SciPy).

CIP-0003: right-side / upper / non-unit cases (TRSM4, TRSM11, TRSM12) from the
2005 MATLAB generator disagree with contemporary OpenBLAS/LAPACK at the level
seen on CI (~1e-8 to ~1e-2) because triu(U2) is extremely ill-conditioned.
Recompute those three results with scipy.linalg.solve_triangular so the
CMatrix::trsm wrapper is tested against the same library family CI uses.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent
PATH = HERE / "trsmMatrixTest.mat"


def refresh_right(uplo: str, trans: str, unit: bool, Afull: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    A = np.tril(Afull) if uplo == "L" else np.triu(Afull)
    if unit:
        np.fill_diagonal(A, 1.0)
    rhs = (alpha * B).T
    if trans == "N":
        XT = solve_triangular(A.T, rhs, lower=(uplo == "U"), unit_diagonal=unit)
    else:
        XT = solve_triangular(A, rhs, lower=(uplo == "L"), unit_diagonal=unit)
    return XT.T


def main() -> None:
    m = loadmat(PATH, squeeze_me=False)
    B = np.asarray(m["B"], dtype=np.float64)
    L2 = np.asarray(m["L2"], dtype=np.float64)
    U2 = np.asarray(m["U2"], dtype=np.float64)
    alpha = float(np.asarray(m["alpha"]).reshape(-1)[0])
    m["TRSM4"] = refresh_right("L", "T", False, L2, B, alpha)
    m["TRSM11"] = refresh_right("U", "N", False, U2, B, alpha)
    m["TRSM12"] = refresh_right("U", "T", False, U2, B, alpha)
    out = {k: v for k, v in m.items() if not k.startswith("__")}
    savemat(PATH, out, do_compression=False, long_field_names=True)
    print(f"Updated TRSM4/TRSM11/TRSM12 in {PATH}")


if __name__ == "__main__":
    main()
