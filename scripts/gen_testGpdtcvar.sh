#!/usr/bin/env bash
# Generate matfiles/testGpdtcvar.mat via Octave+GPmat, then stamp C++ params/grads/ll.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d /opt/homebrew/Cellar/hdf5/1.14.5/lib ]]; then
  export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/hdf5/1.14.5/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

PYTHON="${ROOT}/.venv-diag/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  if python3 -c 'import scipy' 2>/dev/null; then
    PYTHON=python3
  else
    python3 -m venv /tmp/gpc-mat-venv 2>/dev/null || true
    /tmp/gpc-mat-venv/bin/pip -q install scipy numpy >/dev/null
    PYTHON=/tmp/gpc-mat-venv/bin/python
  fi
fi

echo "== Octave/GPmat first pass (structure + GPmat ll/grads) =="
octave --no-gui --quiet --eval "run('${ROOT}/scripts/octave/gen_testGpdtcvar.m');"

echo "== Prepare numeric inputs for C++ =="
"$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
root = Path(".")
d = loadmat(str(root / "matfiles/testGpdtcvar.mat"), struct_as_record=False, squeeze_me=True)
savemat(
    str(root / "matfiles/_gen_in.mat"),
    {
        "X": np.asarray(d["X"], dtype=np.float64),
        "y": np.asarray(d["y"], dtype=np.float64).reshape(-1, 1),
        "X_u": np.asarray(d["X_u"], dtype=np.float64),
        "bias": np.array([[float(np.asarray(d["bias"]).ravel()[0])]], dtype=np.float64),
        "scale": np.array([[float(np.asarray(d["scale"]).ravel()[0])]], dtype=np.float64),
        "numActive": np.array([[float(np.asarray(d["numActive"]).ravel()[0])]], dtype=np.float64),
        "beta": np.array([[float(np.asarray(d["gpInfoInit"].beta).ravel()[0])]], dtype=np.float64),
    },
    do_compression=False,
)
print("Wrote matfiles/_gen_in.mat")
PY

echo "== C++ genTestGpDtcvar =="
make USE_MATIO=1 genTestGpDtcvar
./genTestGpDtcvar

echo "== Octave second pass (re-save with C++ stamp, keep structs) =="
octave --no-gui --quiet --eval "run('${ROOT}/scripts/octave/gen_testGpdtcvar.m');"

echo "Done. Run: make USE_MATIO=1 testGp && ./testGp"
