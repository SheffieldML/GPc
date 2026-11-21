"""Multi-output stacking and ICM/LCM kernel sugar (GPy ``util.multioutput``).

Native twins: ``buildXY`` / ``stampOutputIndex`` / ``createIcmKern`` (CIP-000D /
CIP-0015). Covariance maths stay in ``CCoregionalizeKern`` / ``CTensorKern``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .. import kern as kern_mod


def build_XY(
    X_list: Sequence[np.ndarray],
    Y_list: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-output blocks and append the output index as the last column of ``X``.

    Matches GPy ``util.multioutput.build_XY`` / GPc ``buildXY``.
    """
    if len(X_list) == 0:
        raise ValueError("X_list must be non-empty")
    if len(X_list) != len(Y_list):
        raise ValueError("X_list and Y_list must have the same length")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    q: int | None = None
    for d, (X, Y) in enumerate(zip(X_list, Y_list)):
        Xa = np.asarray(X, dtype=float)
        Ya = np.asarray(Y, dtype=float)
        if Xa.ndim != 2:
            raise ValueError(f"X_list[{d}] must be 2-D")
        if Ya.ndim == 1:
            Ya = Ya[:, None]
        if Ya.ndim != 2 or Ya.shape[1] != 1:
            raise ValueError(f"Y_list[{d}] must be N x 1")
        if Ya.shape[0] != Xa.shape[0]:
            raise ValueError(f"X_list[{d}] and Y_list[{d}] row counts must match")
        if q is None:
            q = Xa.shape[1]
        elif Xa.shape[1] != q:
            raise ValueError("all X blocks must share the same input dimension")
        idx = np.full((Xa.shape[0], 1), float(d), dtype=float)
        xs.append(np.hstack([Xa, idx]))
        ys.append(Ya)
    assert q is not None
    return np.vstack(xs), np.vstack(ys)


def stamp_output_index(X: np.ndarray, output_index: int) -> np.ndarray:
    """Overwrite the last column of ``X`` with ``output_index`` (predict-time helper)."""
    Xa = np.array(X, dtype=float, copy=True)
    if Xa.ndim != 2 or Xa.shape[1] < 1:
        raise ValueError("X must be 2-D with at least one column")
    Xa[:, -1] = float(output_index)
    return Xa


def ICM(
    input_dim: int,
    num_outputs: int,
    kernel: kern_mod.Kern,
    W_rank: int = 1,
) -> kern_mod.Prod:
    """Intrinsic coregionalization: ``spatial ⊗ Coregionalize`` (GPy ``ICM``)."""
    if not isinstance(kernel, kern_mod.Kern):
        raise TypeError("kernel must be a gpc.kern.Kern")
    q = int(input_dim)
    D = int(num_outputs)
    spatial = kernel.with_active_dims(list(range(q)))
    coreg = kern_mod.Coregionalize(
        1, output_dim=D, rank=int(W_rank), active_dims=[q]
    )
    return spatial * coreg


def LCM(
    input_dim: int,
    num_outputs: int,
    kernels_list: Sequence[kern_mod.Kern],
    W_rank: int = 1,
) -> kern_mod.Kern:
    """Linear coregionalization: sum of ICMs (GPy ``LCM``)."""
    if len(kernels_list) < 1:
        raise ValueError("kernels_list must be non-empty")
    parts = [
        ICM(input_dim, num_outputs, k, W_rank=W_rank) for k in kernels_list
    ]
    if len(parts) == 1:
        return parts[0]
    return kern_mod.Add(*parts)


__all__ = ["build_XY", "stamp_output_index", "ICM", "LCM"]
