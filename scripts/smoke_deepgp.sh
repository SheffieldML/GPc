#!/usr/bin/env bash
# CIP-0035: exercise ./deepgp learn / display for coverage + CI smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ./deepgp ]]; then
  echo "ERROR: ./deepgp missing; build with: make deepgp" >&2
  exit 1
fi

DATA="${DEEPGP_SMOKE_DATA:-examples/sinc.svml}"
OUTDIR="${DEEPGP_SMOKE_OUT:-/tmp/deepgp_smoke_$$}"
mkdir -p "$OUTDIR"
MODEL="$OUTDIR/sinc.model"

./deepgp -v 1 -s 1 learn -L 2 --hidden 1 -M 5 -# 5 \
  --batch 20 --samples 1 --inducing points --chi 1.0 \
  --adam-lr 1e-2 -k rbf \
  "$DATA" "$MODEL"
./deepgp -v 1 display "$MODEL"

WS_MODEL="$OUTDIR/sinc_ws.model"
./deepgp -v 1 -s 2 learn -L 2 -M 5 -# 3 \
  --inducing weightspace --residual off -k mlp \
  "$DATA" "$WS_MODEL"
./deepgp -v 1 display "$WS_MODEL"

# CIP-003E Phase 6: multi-class likelihoods from integer labels (dense CSV, last column = label).
CSV="$OUTDIR/blobs.csv"
python3 - "$CSV" <<'PY'
import sys, math
rows = []
for i in range(60):
    k = i % 3
    a = 2.0 * math.pi * k / 3.0
    r = 1.0 + 0.1 * math.sin(7.0 * i)
    rows.append("%.6f,%.6f,%d" % (r * math.cos(a) + 0.1 * math.cos(3.0 * i),
                                  r * math.sin(a) + 0.1 * math.sin(5.0 * i), k))
open(sys.argv[1], "w").write("\n".join(rows) + "\n")
PY
for LIK in softmax robustmax; do
  MC_MODEL="$OUTDIR/blobs_$LIK.model"
  ./deepgp -v 1 -s 3 learn -L 2 --hidden 2 -M 6 -# 5 --likelihood "$LIK" \
    "$CSV" "$MC_MODEL"
  ./deepgp -v 1 display "$MC_MODEL" | tee "$OUTDIR/display_$LIK.txt"
  grep -q "likelihood=$LIK" "$OUTDIR/display_$LIK.txt"
  grep -q "Dout=3" "$OUTDIR/display_$LIK.txt"
done
if ./deepgp -s 3 learn --likelihood softmax --num-classes 2 "$CSV" "$OUTDIR/bad.model" 2>/dev/null; then
  echo "ERROR: out-of-range labels were accepted" >&2
  exit 1
fi

echo "smoke_deepgp: OK ($OUTDIR)"
