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

echo "smoke_deepgp: OK ($OUTDIR)"
