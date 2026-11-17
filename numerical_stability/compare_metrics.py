#!/usr/bin/env python3
"""Compare numerical stability metrics vs a pinned-cell baseline (CIP-001A).

Reads two metrics.json documents (baseline from main artifact, head from PR)
and classifies each shared metric as forward / backward / stable / over_budget.

Exit code is 0 during soft soak unless --hard-gate is set (then non-zero on
over-budget or severe backward). Cross-cell diffs refuse unless --force-cross-cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FORWARD = "forward"
BACKWARD = "backward"
STABLE = "stable"
OVER_BUDGET = "over_budget"
MISSING = "missing"
NEW = "new"

# Relative / absolute noise floors by category (BLAS micro-noise).
CATEGORY_EPS = {
    "A_machine": {"rel": 0.05, "abs": 1e-16},
    "B_linalg": {"rel": 0.10, "abs": 1e-14},
    "C_special": {"rel": 0.05, "abs": 1e-14},
    "D_analytic": {"rel": 0.05, "abs": 1e-12},
    "D_fd": {"rel": 0.15, "abs": 1e-8},
    "E_invariance": {"rel": 0.10, "abs": 1e-12},
    "F_model": {"rel": 0.10, "abs": 1e-8},
}
DEFAULT_EPS = {"rel": 0.10, "abs": 1e-12}


def load_doc(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_label(doc: dict[str, Any]) -> str:
    cell = doc.get("cell") or {}
    if isinstance(cell, dict):
        return str(cell.get("label") or cell.get("runner") or "")
    return str(cell)


def classify_pair(
    base_val: float,
    head_val: float,
    budget: float,
    category: str,
    *,
    unit: str = "abs",
) -> str:
    """Lower residual is better. Over-budget takes priority as a label flag."""
    eps = CATEGORY_EPS.get(category, DEFAULT_EPS)
    # Absolute-style compare for residual magnitudes.
    scale = max(abs(base_val), abs(head_val), eps["abs"])
    delta = head_val - base_val
    # Prefer smaller |error| when both same sign residual; for threshold metrics
    # with negative budgets, "better" means more negative / farther under threshold.
    if budget < 0.0:
        # Threshold: value should be <= budget. Improvement = head more negative.
        if head_val < base_val - max(eps["abs"], eps["rel"] * abs(base_val)):
            label = FORWARD
        elif head_val > base_val + max(eps["abs"], eps["rel"] * abs(base_val)):
            label = BACKWARD
        else:
            label = STABLE
        if head_val > budget:
            return OVER_BUDGET
        return label

    # Residual-style: smaller abs(value) is forward.
    base_m = abs(base_val)
    head_m = abs(head_val)
    thr = max(eps["abs"], eps["rel"] * max(base_m, 1e-300))
    if head_m < base_m - thr:
        label = FORWARD
    elif head_m > base_m + thr:
        label = BACKWARD
    else:
        label = STABLE
    if budget > 0.0 and head_m > budget:
        return OVER_BUDGET
    return label


def diff_metrics(
    baseline: dict[str, Any],
    head: dict[str, Any],
    *,
    force_cross_cell: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    b_cell = cell_label(baseline)
    h_cell = cell_label(head)
    if b_cell and h_cell and b_cell != h_cell and not force_cross_cell:
        raise ValueError(
            f"cross-cell compare refused: baseline={b_cell!r} head={h_cell!r} "
            "(pass --force-cross-cell to override)"
        )

    b_mets = baseline.get("metrics") or {}
    h_mets = head.get("metrics") or {}
    buckets: dict[str, list[dict[str, Any]]] = {
        FORWARD: [],
        BACKWARD: [],
        STABLE: [],
        OVER_BUDGET: [],
        MISSING: [],
        NEW: [],
    }

    for name in sorted(set(b_mets) | set(h_mets)):
        b = b_mets.get(name)
        h = h_mets.get(name)
        if b is None:
            buckets[NEW].append({"name": name, "head": h})
            continue
        if h is None:
            buckets[MISSING].append({"name": name, "baseline": b})
            continue
        cat = h.get("category") or b.get("category") or ""
        budget = float(h.get("budget", b.get("budget", 0.0)))
        unit = str(h.get("unit") or b.get("unit") or "abs")
        b_val = float(b["value"])
        h_val = float(h["value"])
        label = classify_pair(b_val, h_val, budget, cat, unit=unit)
        buckets[label].append(
            {
                "name": name,
                "category": cat,
                "baseline": b_val,
                "head": h_val,
                "delta": h_val - b_val,
                "budget": budget,
                "unit": unit,
                "label": label,
            }
        )
    return buckets


def utilisation(value: float, budget: float) -> str:
    if budget <= 0.0:
        return "—"
    return f"{100.0 * abs(value) / abs(budget):.1f}%"


def render_markdown(
    buckets: dict[str, list[dict[str, Any]]],
    *,
    cell: str = "",
    baseline_ref: str = "",
    head_ref: str = "",
) -> str:
    rows: list[dict[str, Any]] = []
    for key in (OVER_BUDGET, BACKWARD, FORWARD, STABLE):
        rows.extend(buckets.get(key, []))

    lines = [
        "## Numerical stability trend" + (f" (cell: `{cell}`)" if cell else ""),
        "",
    ]
    if baseline_ref or head_ref:
        lines.append(f"- Baseline: `{baseline_ref}`")
        lines.append(f"- Head: `{head_ref}`")
        lines.append("")

    if not rows and not buckets.get(NEW) and not buckets.get(MISSING):
        lines.append("_No overlapping metrics to compare (baseline may be empty)._")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| metric | category | baseline | head | Δ | util | label |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    arrow = {FORWARD: "▼ forward", BACKWARD: "▲ backward", STABLE: "· stable", OVER_BUDGET: "⚠️ over budget"}
    for row in sorted(rows, key=lambda r: (r.get("category", ""), r["name"])):
        lines.append(
            f"| `{row['name']}` | `{row['category']}` | {row['baseline']:.6g} | "
            f"{row['head']:.6g} | {row['delta']:+.3g} | {utilisation(row['head'], row['budget'])} | "
            f"{arrow.get(row['label'], row['label'])} |"
        )

    n_fwd = len(buckets.get(FORWARD, []))
    n_back = len(buckets.get(BACKWARD, []))
    n_over = len(buckets.get(OVER_BUDGET, []))
    lines.extend(
        [
            "",
            f"Forward: **{n_fwd}** · Backward: **{n_back}** · Over budget: **{n_over}** "
            f"· New: {len(buckets.get(NEW, []))} · Missing: {len(buckets.get(MISSING, []))}",
            "",
            "_Soft soak (CIP-001A): trend is advisory; outcome gate remains CIP-0019._",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path, help="Baseline metrics.json")
    p.add_argument("head", type=Path, help="Head metrics.json")
    p.add_argument("--md", default="", help="Write markdown report path")
    p.add_argument("--force-cross-cell", action="store_true")
    p.add_argument(
        "--hard-gate",
        action="store_true",
        help="Exit 1 on over-budget or backward (default: soft soak, exit 0)",
    )
    p.add_argument("--baseline-ref", default="")
    p.add_argument("--head-ref", default="")
    args = p.parse_args(argv)

    if not args.baseline.is_file():
        print(f"compare_metrics: baseline missing: {args.baseline}", file=sys.stderr)
        md = render_markdown({}, cell="", baseline_ref=args.baseline_ref, head_ref=args.head_ref)
        md = "## Numerical stability trend\n\n_No baseline artifact yet — seed on next main run._\n"
        if args.md:
            Path(args.md).write_text(md, encoding="utf-8")
            print(md)
        return 0

    baseline = load_doc(args.baseline)
    head = load_doc(args.head)
    try:
        buckets = diff_metrics(baseline, head, force_cross_cell=args.force_cross_cell)
    except ValueError as exc:
        print(f"compare_metrics: {exc}", file=sys.stderr)
        return 2

    cell = cell_label(head) or cell_label(baseline)
    md = render_markdown(
        buckets,
        cell=cell,
        baseline_ref=args.baseline_ref or str(args.baseline),
        head_ref=args.head_ref or str(args.head),
    )
    print(md)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")

    if args.hard_gate and (buckets.get(OVER_BUDGET) or buckets.get(BACKWARD)):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
