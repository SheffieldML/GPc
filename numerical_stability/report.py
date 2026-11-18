#!/usr/bin/env python3
"""Render a Codecov-style numerical stability report (CIP-0019 + CIP-001A).

Reads OUTCOME and METRIC lines from a suite log (or stdin) and writes:

* Markdown summary (for GitHub Actions job summary / PR comment)
* JSON outcome map (for compare.py --from-json)
* JSON metrics document (for compare_metrics.py / baseline artifact)

Exit code is always 0 unless the input cannot be parsed at all.
Pass/fail of the suite itself remains the test binary's exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OUTCOME_RE = re.compile(r"^OUTCOME name=(?P<name>\S+) result=(?P<result>\S+)\s*$")
METRIC_RE = re.compile(
    r"^METRIC name=(?P<name>\S+) category=(?P<category>\S+) "
    r"value=(?P<value>\S+) budget=(?P<budget>\S+) "
    r"unit=(?P<unit>\S+) cell=(?P<cell>\S+)\s*$"
)

ORDER = ("failed", "xpassed", "xfailed", "passed", "skipped", "error")
BADGE = {
    "passed": "✅",
    "failed": "❌",
    "xfailed": "⚠️",
    "xpassed": "🚨",
    "skipped": "⏭️",
    "error": "💥",
}

CATEGORY_ORDER = (
    "A_machine",
    "B_linalg",
    "C_special",
    "D_analytic",
    "D_fd",
    "E_invariance",
    "F_model",
)


def parse_outcomes(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = OUTCOME_RE.match(line.strip())
        if m:
            out[m.group("name")] = m.group("result")
    return out


def parse_metrics(text: str) -> dict[str, dict[str, Any]]:
    """Parse METRIC lines → {name: {category, value, budget, unit, cell}}."""
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        try:
            value = float(m.group("value"))
            budget = float(m.group("budget"))
        except ValueError:
            continue
        out[m.group("name")] = {
            "category": m.group("category"),
            "value": value,
            "budget": budget,
            "unit": m.group("unit"),
            "cell": m.group("cell"),
        }
    return out


def default_cell_meta(cell_label: str = "") -> dict[str, str]:
    sparse = os.environ.get("GPC_SPARSE_LINALG", "stable").lower()
    if sparse in {"1", "stablefactor"}:
        sparse = "stable"
    elif sparse in {"0"}:
        sparse = "legacy"
    base = os.environ.get("GPC_NS_CELL", "local")
    label = cell_label or f"{base}+{sparse}"
    return {
        "label": label,
        "os": platform.system(),
        "blas": os.environ.get("GPC_NS_BLAS", "unknown"),
        "cxx": os.environ.get("CXX", "unknown"),
        "runner": os.environ.get("RUNNER_OS", platform.system()),
        "sparse_linalg": sparse if sparse in {"stable", "legacy"} else "stable",
    }


def build_metrics_doc(
    metrics: dict[str, dict[str, Any]],
    *,
    git_sha: str = "",
    cell_meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    cell_label = ""
    cleaned: dict[str, dict[str, Any]] = {}
    for name, rec in metrics.items():
        cell_label = cell_label or str(rec.get("cell", ""))
        cleaned[name] = {
            "category": rec["category"],
            "value": rec["value"],
            "budget": rec["budget"],
            "unit": rec["unit"],
        }
    meta = cell_meta if cell_meta is not None else default_cell_meta(cell_label)
    if cell_label and not meta.get("label"):
        meta = dict(meta)
        meta["label"] = cell_label
    return {
        "cell": meta,
        "git_sha": git_sha,
        "metrics": cleaned,
    }


def _utilisation(value: float, budget: float) -> float | None:
    if budget == 0.0:
        return None
    # Budgets may be negative thresholds (e.g. deep-left lnCum < -1e8).
    # Utilisation is meaningful for positive residual-style budgets.
    if budget < 0.0:
        return None
    return abs(value) / abs(budget)


def render_metrics_section(metrics: dict[str, dict[str, Any]]) -> list[str]:
    if not metrics:
        return []
    by_cat: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, rec in metrics.items():
        by_cat[rec["category"]].append((name, rec))

    lines = [
        "## Stability metrics (CIP-001A)",
        "",
        "| metric | category | value | budget | util | unit |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    cat_keys = [c for c in CATEGORY_ORDER if c in by_cat]
    cat_keys.extend(sorted(c for c in by_cat if c not in CATEGORY_ORDER))
    over = 0
    for cat in cat_keys:
        for name, rec in sorted(by_cat[cat]):
            util = _utilisation(rec["value"], rec["budget"])
            util_s = "—" if util is None else f"{100.0 * util:.1f}%"
            flag = ""
            if rec["budget"] > 0.0 and abs(rec["value"]) > rec["budget"]:
                flag = " ⚠️"
                over += 1
            elif rec["budget"] < 0.0 and rec["value"] > rec["budget"]:
                # threshold-style: value should stay below budget (more negative OK)
                flag = " ⚠️"
                over += 1
            lines.append(
                f"| `{name}` | `{cat}` | {rec['value']:.6g} | {rec['budget']:.6g} "
                f"| {util_s}{flag} | `{rec['unit']}` |"
            )
    cell = next(iter(metrics.values())).get("cell", "")
    lines.extend(
        [
            "",
            f"_Cell: `{cell}` · soft soak (over-budget does not fail the suite alone)._  ",
            f"_Metrics: {len(metrics)} · over budget: {over}_",
            "",
        ]
    )
    return lines


def render_markdown(
    outcomes: dict[str, str],
    *,
    title: str = "Numerical stability report",
    ref: str = "",
    metrics: dict[str, dict[str, Any]] | None = None,
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

    if metrics:
        lines.append("")
        lines.extend(render_metrics_section(metrics))

    lines.extend(
        [
            "",
            "---",
            "",
            "_CIP-0019 adversarial track (separate from matio reference tests)._",
            "_Strict xfail: `xpassed` fails CI until the marker is removed._",
            "_CIP-001A metrics: soft soak; trend vs main baseline on the pinned cell._",
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
        help="Suite log with OUTCOME/METRIC lines (default: stdin)",
    )
    p.add_argument("--md", default="numerical_stability/report.md")
    p.add_argument("--json", dest="json_path", default="numerical_stability/outcomes.json")
    p.add_argument(
        "--metrics-json",
        default="numerical_stability/metrics.json",
        help="Write metrics document (CIP-001A)",
    )
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

    metrics = parse_metrics(text)
    md = render_markdown(outcomes, title=args.title, ref=args.ref, metrics=metrics or None)
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    doc = build_metrics_doc(metrics, git_sha=args.ref)
    metrics_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(md)
    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {metrics_path}", file=sys.stderr)

    if args.github_summary:
        ghs = os.environ.get("GITHUB_STEP_SUMMARY")
        if ghs:
            with open(ghs, "a", encoding="utf-8") as fh:
                fh.write(md)
                fh.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
