#!/usr/bin/env bash
# CIP-0017: exercise ./bgplvm learn / display / gnuplot for coverage + CI smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ./bgplvm ]]; then
  echo "ERROR: ./bgplvm missing; build with: make bgplvm" >&2
  exit 1
fi

DATA="${BGPLVM_SMOKE_DATA:-testdata/cip0017/tiny.svml}"
OUTDIR="${BGPLVM_SMOKE_OUT:-/tmp/bgplvm_smoke_$$}"
mkdir -p "$OUTDIR"
MODEL="$OUTDIR/tiny.model"
STUB="$OUTDIR/tiny"

./bgplvm -v 1 -s 1 learn -x 3 -M 3 -# 5 -k rbfard -o "$OUTDIR/tiny" \
  "$DATA" "$MODEL"
./bgplvm -v 1 display "$MODEL"
# Default plane 0 1 and an alternate plane for q=3.
./bgplvm -v 1 gnuplot -r 12 -d 0 1 "$MODEL" "$STUB"
./bgplvm -v 1 gnuplot -r 12 -d 0 2 "$MODEL" "${STUB}_d02"
# Isotropic path + -i true alias; compound kerns for makeKernComponent.
./bgplvm -v 1 -s 2 learn -x 2 -M 3 -# 3 -k rbf -i true \
  "$DATA" "$OUTDIR/iso.model"
./bgplvm -v 1 -s 3 learn -x 2 -M 3 -# 2 \
  -k lin -k bias -k white \
  "$DATA" "$OUTDIR/cmpnd.model"
./bgplvm learn -h >/dev/null
./bgplvm display -h >/dev/null
./bgplvm gnuplot -h >/dev/null

test -f "${STUB}_plot.gp"
test -f "${STUB}_variance_matrix.dat"
test -f "${OUTDIR}/tiny_mu.dat"
test -f "${OUTDIR}/tiny_S.dat"
echo "OK: bgplvm smoke (learn/display/gnuplot -d)"
