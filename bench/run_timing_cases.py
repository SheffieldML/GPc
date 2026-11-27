#!/usr/bin/env python3
"""CIP-002D QW8: run fixed benchGPc cases; write JSON artifact (stdlib only).

Usage:
  python3 bench/run_timing_cases.py [--bench ./benchGPc] [--cases bench/timing_cases.json]
                                    [--out bench/timings.json] [--sha REF]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_bench_line(line: str) -> dict:
    out = {}
    for tok in line.strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def run_one(bench: Path, case: dict) -> dict:
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [
        str(bench),
        "--kind",
        case["kind"],
        "--kern",
        case.get("kern", "rbf"),
        "--N",
        str(case["N"]),
        "--q",
        str(case.get("q", 2)),
        "--reps",
        str(case.get("reps", 7)),
    ]
    M = int(case.get("M", 0) or 0)
    if case["kind"] != "ftc" and M > 0:
        cmd.extend(["--M", str(M)])
    for key in ("layers", "hidden", "batch", "inducing", "patch", "stride", "diag-samples"):  # CIP-003F deepgp, CIP-0040 convgp cases
        if key in case:
            cmd.extend(["--" + key, str(case[key])])
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchGPc failed ({proc.returncode}) for {case.get('id')}:\n"
            f"{proc.stderr or proc.stdout}"
        )
    line = ""
    for ln in proc.stdout.splitlines():
        if ln.startswith("kind="):
            line = ln
    if not line:
        raise RuntimeError(
            f"no kind= line from benchGPc for {case.get('id')}:\n{proc.stdout}"
        )
    parsed = parse_bench_line(line)
    row = {
        "id": case["id"],
        "kind": parsed.get("kind", case["kind"]),
        "kern": parsed.get("kern", case.get("kern", "rbf")),
        "N": int(parsed.get("N", case["N"])),
        "q": int(parsed.get("q", case.get("q", 2))),
        "M": int(parsed.get("M", case.get("M", 0) or 0)),
        "median_ms": float(parsed["median_ms"]),
        "allocs_per_eval": int(float(parsed.get("allocs_per_eval", "0"))),
        "nparams": int(parsed.get("nparams", "0")),
        "ll": float(parsed.get("ll", "nan")),
        "reps": int(parsed.get("reps", case.get("reps", 7))),
        "flags": parsed.get("flags", ""),
        "gpc_checks": parsed.get("gpc_checks", ""),
        "raw": line,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=Path("./benchGPc"))
    ap.add_argument(
        "--cases", type=Path, default=Path("bench/timing_cases.json")
    )
    ap.add_argument("--out", type=Path, default=Path("bench/timings.json"))
    ap.add_argument("--sha", default="")
    args = ap.parse_args()

    bench = args.bench.resolve()
    if not bench.is_file():
        print(f"error: bench binary not found: {bench}", file=sys.stderr)
        return 2
    cases_path = args.cases.resolve()
    spec = json.loads(cases_path.read_text())
    cases = spec["cases"]
    rows = []
    for case in cases:
        print(f"running {case['id']}...", flush=True)
        rows.append(run_one(bench, case))
        print(
            f"  {rows[-1]['id']}: median_ms={rows[-1]['median_ms']:.3f} "
            f"allocs={rows[-1]['allocs_per_eval']}",
            flush=True,
        )

    payload = {
        "protocol": spec.get("protocol", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ref": args.sha or os.environ.get("GITHUB_SHA", ""),
        "cell": os.environ.get("GPC_BENCH_CELL", "local"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "cases": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
