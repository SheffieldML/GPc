#!/usr/bin/env python3
"""Render a Codecov-style numerical stability report (CIP-0019).

Reads OUTCOME lines from a suite log (or stdin) and writes:

* Markdown summary (for GitHub Actions job summary / PR comment)
* JSON outcome map (for compare.py --from-json)

Exit code is always 0 unless the input cannot be parsed at all.
Pass/fail of the suite itself remains the test binary's exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

OUTCOME_RE = re.compile(r"^OUTCOME name=(?P<name>\S+) result=(?P<result>\S+)\s*$")

ORDER = ("failed", "xpassed", "xfailed", "passed", "skipped", "error")
BADGE = {
    "passed": "✅",
    "failed": "❌",
    "xfailed": "⚠️",
    "xpassed": "🚨",
    "skipped": "⏭️",
    "error": "💥",
}


def parse_outcomes(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = OUTCOME_RE.match(line.strip())
        if m:
            out[m.group("name")] = m.group("result")
    return out


def render_markdown(
    outcomes: dict[str, str],
    *,
    title: str = "Numerical stability report",
    ref: str = "",
) -> str:
    counts = Counter(outcomes.values())
    total = len(outcomes)
    debt = counts.get("xfailed", 0)
    bad = counts.get("failed", 0) + counts.get("xpassed", 0) + counts.get("error", 0)
    status = "red" if bad else ("yellow" if debt else "green")
    status_label = {
        "green": "clean",
        "yellow": "known debt",
        "red": "regressions",
    }[status]

    lines = [
        f"# {title}",
        "",
        f"**Status:** {status_label}  ",
    ]
    if ref:
        lines.append(f"**Ref:** `{ref}`  ")
    lines.extend(
        [
            f"**Cases:** {total}  ",
            "",
            "| result | count |",
            "| --- | ---: |",
        ]
    )
    for key in ORDER:
        if counts.get(key):
            lines.append(f"| {BADGE.get(key, '')} `{key}` | {counts[key]} |")
    for key, n in sorted(counts.items()):
        if key not in ORDER:
            lines.append(f"| `{key}` | {n} |")

    lines.extend(["", "## Cases", ""])
    buckets = {k: [] for k in ORDER}
    other: list[tuple[str, str]] = []
    for name, result in outcomes.items():
        if result in buckets:
            buckets[result].append(name)
        else:
            other.append((name, result))
    for result in ORDER:
        for name in sorted(buckets[result]):
            lines.append(f"- {BADGE.get(result, '')} `{name}` — `{result}`")
    for name, result in sorted(other):
        lines.append(f"- `{name}` — `{result}`")

    lines.extend(
        [
            "",
            "---",
            "",
            "_CIP-0019 adversarial track (separate from matio reference tests)._",
            "_Strict xfail: `xpassed` fails CI until the marker is removed._",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "log",
        nargs="?",
        default="-",
        help="Suite log with OUTCOME lines (default: stdin)",
    )
    p.add_argument("--md", default="numerical_stability/report.md")
    p.add_argument("--json", dest="json_path", default="numerical_stability/outcomes.json")
    p.add_argument("--title", default="Numerical stability report")
    p.add_argument("--ref", default="")
    p.add_argument(
        "--github-summary",
        action="store_true",
        help="Also append markdown to $GITHUB_STEP_SUMMARY when set",
    )
    args = p.parse_args(argv)

    if args.log == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    outcomes = parse_outcomes(text)
    if not outcomes:
        print("report.py: no OUTCOME lines found", file=sys.stderr)
        return 1

    md = render_markdown(outcomes, title=args.title, ref=args.ref)
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(md)
    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)

    if args.github_summary:
        ghs = os.environ.get("GITHUB_STEP_SUMMARY")
        if ghs:
            with open(ghs, "a", encoding="utf-8") as fh:
                fh.write(md)
                fh.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
