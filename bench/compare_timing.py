#!/usr/bin/env python3
"""CIP-002D QW8: compare timing artifact to same-workflow self-baseline.

Warn-only by default (exit 0). A case regresses if median_ms rises by more
than --threshold (default 0.25 = 25%) relative to baseline. Hard fail is
opt-in via --fail-on-regression once consecutive-run policy is ready.

Usage:
  python3 bench/compare_timing.py baseline.json head.json [--md trend.md]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_cases(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {c["id"]: c for c in data.get("cases", [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("head", type=Path)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 if any case exceeds threshold (default: warn only)",
    )
    ap.add_argument("--baseline-ref", default="main")
    ap.add_argument("--head-ref", default="HEAD")
    args = ap.parse_args()

    if not args.head.is_file():
        print(f"error: missing head timings: {args.head}", file=sys.stderr)
        return 2

    head = load_cases(args.head)
    base = load_cases(args.baseline)

    lines = [
        f"## Bench timing vs self-baseline (`{args.baseline_ref}` → `{args.head_ref}`)",
        "",
        f"Threshold: +{100.0 * args.threshold:.0f}% median_ms (warn-only"
        + (" until `--fail-on-regression`" if not args.fail_on_regression else "")
        + ").",
        "",
        "| case | baseline_ms | head_ms | delta | status |",
        "|---|---:|---:|---:|---|",
    ]

    regressions = []
    if not base:
        lines.append("| _(no baseline)_ | — | — | — | skip |")
        lines.append("")
        lines.append(
            "_No same-workflow baseline artifact yet; upload timings on green "
            "main before enabling hard gates._"
        )
    else:
        for cid, h in sorted(head.items()):
            b = base.get(cid)
            if b is None:
                lines.append(
                    f"| `{cid}` | — | {h['median_ms']:.3f} | — | new |"
                )
                continue
            bms = float(b["median_ms"])
            hms = float(h["median_ms"])
            if bms <= 0 or not math.isfinite(bms) or not math.isfinite(hms):
                lines.append(
                    f"| `{cid}` | {bms:.3f} | {hms:.3f} | — | skip |"
                )
                continue
            delta = (hms - bms) / bms
            if delta > args.threshold:
                status = f"WARN +{100.0 * delta:.1f}%"
                regressions.append(cid)
            elif delta < -args.threshold:
                status = f"faster {100.0 * delta:.1f}%"
            else:
                status = "ok"
            lines.append(
                f"| `{cid}` | {bms:.3f} | {hms:.3f} | {100.0 * delta:+.1f}% | {status} |"
            )
        lines.append("")
        if regressions:
            lines.append(
                f"**Regressions (>{100.0 * args.threshold:.0f}%):** "
                + ", ".join(f"`{c}`" for c in regressions)
            )
            lines.append("")
            lines.append(
                "_CIP-002D QW8: treat as warning until consecutive same-runner "
                "policy is enabled._"
            )
        else:
            lines.append("_No regressions above threshold._")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text)

    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
