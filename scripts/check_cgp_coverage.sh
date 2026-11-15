#!/usr/bin/env bash
# CIP-0011: fail if CGp.cpp lcov line rate is below THRESHOLD (default 95).
set -euo pipefail

THRESHOLD="${CGP_COVERAGE_THRESHOLD:-95}"
INFO="${1:-coverage.info}"
OUT="${2:-cgp_coverage.info}"

if [[ ! -f "$INFO" ]]; then
  echo "ERROR: missing lcov info file: $INFO" >&2
  exit 1
fi

lcov --ignore-errors unused,empty,mismatch \
  --extract "$INFO" '*/CGp.cpp' \
  --output-file "$OUT"

lcov --summary "$OUT" || true

# Prefer LH/LF records (stable across lcov versions).
lh="$(grep -E '^LH:' "$OUT" | head -1 | cut -d: -f2 | tr -d '[:space:]')"
lf="$(grep -E '^LF:' "$OUT" | head -1 | cut -d: -f2 | tr -d '[:space:]')"
if [[ -z "${lh:-}" || -z "${lf:-}" || "$lf" -eq 0 ]]; then
  echo "ERROR: could not parse CGp.cpp LH/LF from $OUT" >&2
  exit 1
fi

python3 - "$lh" "$lf" "$THRESHOLD" <<'PY'
import sys
lh, lf, thr = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
rate = 100.0 * lh / lf
print(f"CGp.cpp line coverage: {rate:.2f}% ({int(lh)} of {int(lf)}; threshold {thr:g}%)")
if rate + 1e-9 < thr:
    print(f"ERROR: CGp.cpp coverage {rate:.2f}% is below {thr:g}%", file=sys.stderr)
    sys.exit(1)
print("CGp.cpp coverage gate passed.")
PY
