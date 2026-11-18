#!/usr/bin/env python3
"""Run the numerical-stability suite under StableFactor and Legacy, then compare.

Usage (from repo root)::

    python3 numerical_stability/compare_sparse_modes.py

Writes under numerical_stability/mode_compare/:
  stable/{suite.log,metrics.json,outcomes.json,report.md}
  legacy/{…}
  trend.md   (compare_metrics with --force-cross-cell)

Exit code follows the worse of the two suite runs (outcome gate), not metric trend.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> int:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        default="numerical_stability/mode_compare",
        help="Directory for stable/ and legacy/ artifacts",
    )
    p.add_argument(
        "--cell-base",
        default=os.environ.get("GPC_NS_CELL", "local"),
        help="Base GPC_NS_CELL label (mode suffix is added by the suite)",
    )
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args(argv)

    repo = Path.cwd()
    exe = repo / "testNumericalStability"
    out = repo / args.outdir
    (out / "stable").mkdir(parents=True, exist_ok=True)
    (out / "legacy").mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.setdefault("GPC_NS_BLAS", base_env.get("GPC_NS_BLAS", "unknown"))

    if not args.skip_build:
        rc = run(["make", "testNumericalStability"], env=base_env, cwd=repo)
        if rc != 0:
            return rc
    if not exe.is_file():
        print(f"missing binary: {exe}", file=sys.stderr)
        return 2

    worst = 0
    for mode in ("stable", "legacy"):
        env = base_env.copy()
        env["GPC_SPARSE_LINALG"] = mode
        env["GPC_NS_CELL"] = args.cell_base
        dest = out / mode
        log = dest / "suite.log"
        with open(log, "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [str(exe)],
                cwd=str(repo),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        print(f"{mode}: exit={proc.returncode} log={log}", flush=True)
        worst = max(worst, proc.returncode)
        rc = run(
            [
                "python3",
                "numerical_stability/report.py",
                str(log),
                "--md",
                str(dest / "report.md"),
                "--json",
                str(dest / "outcomes.json"),
                "--metrics-json",
                str(dest / "metrics.json"),
                "--ref",
                f"{args.cell_base}+{mode}",
            ],
            env=env,
            cwd=repo,
        )
        if rc != 0:
            worst = max(worst, rc)

    trend = out / "trend.md"
    rc = run(
        [
            "python3",
            "numerical_stability/compare_metrics.py",
            str(out / "stable" / "metrics.json"),
            str(out / "legacy" / "metrics.json"),
            "--md",
            str(trend),
            "--force-cross-cell",
            "--baseline-ref",
            f"{args.cell_base}+stable",
            "--head-ref",
            f"{args.cell_base}+legacy",
        ],
        env=base_env,
        cwd=repo,
    )
    if rc not in (0, 1):
        worst = max(worst, rc)
    print(f"Wrote {trend}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
