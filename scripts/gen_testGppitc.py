#!/usr/bin/env python3
"""Merge C++ PITC stamp into matfiles/testGppitc.mat (CIP-000F)."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat

root = Path(__file__).resolve().parents[1]
src = loadmat(str(root / "matfiles/testGpdtc.mat"), squeeze_me=True, struct_as_record=False)
params = np.asarray(loadmat(str(root / "matfiles/_gen_pitc_params.mat"))["params"], dtype=np.float64).ravel()
grads = np.asarray(loadmat(str(root / "matfiles/_gen_pitc_grads.mat"))["grads"], dtype=np.float64).ravel()
ll = float(np.asarray(loadmat(str(root / "matfiles/_gen_pitc_ll.mat"))["ll"]).ravel()[0])
approxInt = float(np.asarray(loadmat(str(root / "matfiles/_gen_pitc_approxInt.mat"))["approxInt"]).ravel()[0])

out = {
    "X": np.asarray(src["X"], dtype=np.float64),
    "y": np.asarray(src["y"], dtype=np.float64).reshape(-1, 1),
    "numActive": np.array([[float(np.asarray(src["numActive"]).ravel()[0])]], dtype=np.float64),
    "approxInt": np.array([[approxInt]], dtype=np.float64),
    "scale": np.array([[float(np.asarray(src["scale"]).ravel()[0])]], dtype=np.float64),
    "bias": np.array([[float(np.asarray(src["bias"]).ravel()[0])]], dtype=np.float64),
    "params": params.reshape(1, -1),
    "grads": grads.reshape(1, -1),
    "ll": np.array([[ll]], dtype=np.float64),
}
savemat(str(root / "matfiles/testGppitc.mat"), out, do_compression=False)
print("Wrote matfiles/testGppitc.mat")
