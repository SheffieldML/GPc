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
from scipy.linalg import blas

HERE = Path(__file__).resolve().parent
PATH = HERE / "trsmMatrixTest.mat"


def trsm_right(Afull: np.ndarray, B: np.ndarray, alpha: float, *, lower: bool, trans: bool, unit: bool) -> np.ndarray:
    """Match CMatrix::trsm side=R via the same BLAS dtrsm interface family."""
    A = np.tril(Afull) if lower else np.triu(Afull)
    if unit:
        np.fill_diagonal(A, 1.0)
    Bb = np.array(B, dtype=np.float64, copy=True)
    return blas.dtrsm(
        alpha,
        A,
        Bb,
        side=1,
        lower=int(lower),
        trans_a=int(trans),
        diag=int(unit),
        overwrite_b=1,
    )


def main() -> None:
    m = loadmat(PATH, squeeze_me=False)
    B = np.asarray(m["B"], dtype=np.float64)
    L2 = np.asarray(m["L2"], dtype=np.float64)
    U2 = np.asarray(m["U2"], dtype=np.float64)
    alpha = float(np.asarray(m["alpha"]).reshape(-1)[0])
    m["TRSM4"] = trsm_right(L2, B, alpha, lower=True, trans=True, unit=False)
    m["TRSM11"] = trsm_right(U2, B, alpha, lower=False, trans=False, unit=False)
    m["TRSM12"] = trsm_right(U2, B, alpha, lower=False, trans=True, unit=False)
    out = {k: v for k, v in m.items() if not k.startswith("__")}
    savemat(PATH, out, do_compression=False, long_field_names=True)
    print(f"Updated TRSM4/TRSM11/TRSM12 in {PATH}")


if __name__ == "__main__":
    main()
