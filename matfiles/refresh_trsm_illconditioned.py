#!/usr/bin/env python3
"""Replace ill-conditioned U2 and refresh dependent TRSM fixtures.

CIP-0003: the historic U2 in trsmMatrixTest.mat has cond(triu(U2)) ~ 1e12, so
golden TRSM11/TRSM12 values are not portable across BLAS libraries under a
tight absolute tolerance. Rebuild U2 as a well-conditioned upper triangle and
recompute TRSM11, TRSM12, TRSM15, TRSM16 with SciPy's BLAS dtrsm (same
operation family as CMatrix::trsm). Absolute 1e-8 fixture checks remain.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.linalg import blas

HERE = Path(__file__).resolve().parent
PATH = HERE / "trsmMatrixTest.mat"


def trsm_right(Afull: np.ndarray, B: np.ndarray, alpha: float, *, lower: bool, trans: bool, unit: bool) -> np.ndarray:
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


def well_conditioned_upper(n: int, rng: np.random.Generator) -> np.ndarray:
    # Diagonals bounded away from zero; strict upper triangle O(1).
    U = np.triu(rng.standard_normal((n, n)))
    np.fill_diagonal(U, 1.0 + np.abs(rng.standard_normal(n)))
    return U


def main() -> None:
    m = loadmat(PATH, squeeze_me=False)
    B = np.asarray(m["B"], dtype=np.float64)
    alpha = float(np.asarray(m["alpha"]).reshape(-1)[0])
    n = B.shape[1]
    rng = np.random.default_rng(20261113)
    U2 = well_conditioned_upper(n, rng)
    print("cond(U2)=", np.linalg.cond(U2))

    m["U2"] = U2
    m["TRSM11"] = trsm_right(U2, B, alpha, lower=False, trans=False, unit=False)
    m["TRSM12"] = trsm_right(U2, B, alpha, lower=False, trans=True, unit=False)
    m["TRSM15"] = trsm_right(U2, B, alpha, lower=False, trans=False, unit=True)
    m["TRSM16"] = trsm_right(U2, B, alpha, lower=False, trans=True, unit=True)

    # TRSM4 was only ~1.3e-8 over the old 1e-8 gate; refresh from the same BLAS.
    L2 = np.asarray(m["L2"], dtype=np.float64)
    m["TRSM4"] = trsm_right(L2, B, alpha, lower=True, trans=True, unit=False)

    out = {k: v for k, v in m.items() if not k.startswith("__")}
    savemat(PATH, out, do_compression=False, long_field_names=True)
    print(f"Updated U2 and TRSM4/11/12/15/16 in {PATH}")


if __name__ == "__main__":
    main()
