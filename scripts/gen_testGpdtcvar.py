#!/usr/bin/env python3
"""Build matfiles/testGpdtcvar.mat from testGpdtc.mat + C++ DTCVAR refs.

Steps (from repo root):
  1. This script writes matfiles/_gen_in.mat
  2. make USE_MATIO=1 genTestGpDtcvar && ./genTestGpDtcvar
  3. Re-run this script with --assemble
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "matfiles"
SRC = MAT / "testGpdtc.mat"
OUT = MAT / "testGpdtcvar.mat"
GEN_IN = MAT / "_gen_in.mat"


def write_inputs() -> None:
    d = loadmat(str(SRC), struct_as_record=False, squeeze_me=True)
    savemat(
        str(GEN_IN),
        {
            "X": np.asarray(d["X"], dtype=np.float64),
            "y": np.asarray(d["y"], dtype=np.float64).reshape(-1, 1),
            "X_u": np.asarray(d["gpInfoInit"].X_u, dtype=np.float64),
            "bias": np.array([[float(d["bias"])]], dtype=np.float64),
            "scale": np.array([[float(d["scale"])]], dtype=np.float64),
            "numActive": np.array([[float(d["numActive"])]], dtype=np.float64),
            "beta": np.array([[float(d["gpInfoInit"].beta)]], dtype=np.float64),
        },
        do_compression=False,
    )
    print(f"Wrote {GEN_IN}")


def _as_float64_matrix(path: Path, name: str) -> np.ndarray:
    d = loadmat(str(path), squeeze_me=False)
    return np.asarray(d[name], dtype=np.float64)


def assemble() -> None:
    d = loadmat(str(SRC), struct_as_record=False, squeeze_me=False)
    params = _as_float64_matrix(MAT / "_gen_params.mat", "params")
    grads = _as_float64_matrix(MAT / "_gen_grads.mat", "grads")
    ll = _as_float64_matrix(MAT / "_gen_ll.mat", "ll")
    Xu = _as_float64_matrix(MAT / "_gen_Xu.mat", "X_u")

    # approxInt: FTC=0, DTC=1, FITC=2, PITC=3, DTCVAR=4 (CGp.h)
    approx_int = np.array([[4.0]], dtype=np.float64)

    out = {
        "X": np.asarray(d["X"], dtype=np.float64),
        "y": np.asarray(d["y"], dtype=np.float64),
        "numActive": np.asarray(d["numActive"], dtype=np.float64),
        "approxInt": approx_int,
        "scale": np.asarray(d["scale"], dtype=np.float64),
        "bias": np.asarray(d["bias"], dtype=np.float64),
        "noiseInit": np.asarray(d["noiseInit"]),
        "params": params,
        "grads": grads,
        "ll": ll,
        "X_u": Xu,  # top-level for matio testGp path
    }

    # Preserve MATLAB structs for _NDLMATLAB testGp path; update approx labels.
    gi = copy.deepcopy(d["gpInfoInit"][0, 0])
    # scipy mat_struct: set fields used by CGp::fromMxArray
    gi.approx = "dtcvar"
    gi.X_u = Xu
    out["gpInfoInit"] = d["gpInfoInit"]  # keep array wrapper
    out["gpInfoInit"][0, 0].approx = "dtcvar"
    out["gpInfoInit"][0, 0].X_u = Xu

    out["kernInit"] = d["kernInit"]
    out["model"] = d["model"]
    out["model"][0, 0].approx = "dtcvar"
    out["model"][0, 0].X_u = Xu

    savemat(str(OUT), out, do_compression=False)
    print(f"Wrote {OUT}")
    print(f"  ll={ll.ravel()[0]:.16e}  nParams={params.size}  approxInt=4 (dtcvar)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--assemble", action="store_true", help="Merge C++ outputs into testGpdtcvar.mat")
    args = p.parse_args(argv)
    if args.assemble:
        assemble()
    else:
        write_inputs()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
