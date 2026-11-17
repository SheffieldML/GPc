#!/usr/bin/env python3
"""Cross-branch numerical stability comparison for GPc (CIP-0019).

Runs the *same* suite sources against two git trees (baseline vs candidate)
and classifies each OUTCOME name as improvement / regression / stable / debt.

Exit code is non-zero iff any regression (or suite drift under --strict-suite).

Example::

    python3 numerical_stability/compare.py \\
      --baseline main --candidate HEAD --suite-from candidate \\
      --report numerical_stability/last-compare.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

OUTCOME_RE = re.compile(r"^OUTCOME name=(?P<name>\S+) result=(?P<result>\S+)\s*$")

VALID_OUTCOMES = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
)

IMPROVEMENT = "improvement"
REGRESSION = "regression"
STABLE = "stable"
KNOWN_DEBT = "known_debt"
IGNORE = "ignore"
SUITE_DRIFT = "suite_drift"


def classify(baseline: str | None, candidate: str | None) -> str:
    if baseline is None or candidate is None:
        if baseline == candidate:
            return IGNORE
        return SUITE_DRIFT

    if baseline not in VALID_OUTCOMES or candidate not in VALID_OUTCOMES:
        raise ValueError(f"unknown outcomes: {baseline!r} vs {candidate!r}")

    if baseline == "skipped" and candidate == "skipped":
        return IGNORE

    if baseline == "xfailed" and candidate in {"passed", "xpassed"}:
        return IMPROVEMENT

    if baseline == "passed" and candidate in {"failed", "error"}:
        return REGRESSION
    if baseline == "xfailed" and candidate in {"failed", "error"}:
        return REGRESSION
    if baseline == "skipped" and candidate in {"failed", "error"}:
        return REGRESSION

    if baseline == "xfailed" and candidate == "xfailed":
        return KNOWN_DEBT

    if baseline == candidate:
        return STABLE

    if {baseline, candidate} <= {"passed", "skipped", "xpassed"}:
        return STABLE if baseline == candidate else IMPROVEMENT if candidate == "passed" else STABLE

    if baseline in {"failed", "error"} and candidate in {"passed", "xfailed", "xpassed"}:
        return IMPROVEMENT

    if baseline in {"failed", "error"} and candidate in {"failed", "error"}:
        return STABLE

    return STABLE


def parse_outcomes(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = OUTCOME_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        result = m.group("result")
        if result not in VALID_OUTCOMES:
            result = "error"
        out[name] = result
    return out


def diff_outcomes(
    baseline: dict[str, str],
    candidate: dict[str, str],
) -> dict[str, list[tuple[str, str | None, str | None]]]:
    buckets: dict[str, list[tuple[str, str | None, str | None]]] = defaultdict(list)
    for nodeid in sorted(set(baseline) | set(candidate)):
        b = baseline.get(nodeid)
        c = candidate.get(nodeid)
        buckets[classify(b, c)].append((nodeid, b, c))
    return dict(buckets)


def write_report(
    path: Path,
    buckets: dict[str, list[tuple[str, str | None, str | None]]],
    baseline_ref: str,
    candidate_ref: str,
) -> None:
    lines = [
        f"# Numerical stability compare",
        "",
        f"- Baseline: `{baseline_ref}`",
        f"- Candidate: `{candidate_ref}`",
        "",
    ]
    order = [IMPROVEMENT, REGRESSION, KNOWN_DEBT, STABLE, SUITE_DRIFT, IGNORE]
    for label in order:
        items = buckets.get(label, [])
        lines.append(f"## {label} ({len(items)})")
        lines.append("")
        if not items:
            lines.append("_none_")
            lines.append("")
            continue
        for name, b, c in items:
            lines.append(f"- `{name}`: {b} → {c}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_worktree(repo: Path, ref: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    proc = run(["git", "worktree", "add", "--detach", str(dest), ref], cwd=repo)
    if proc.returncode != 0:
        raise RuntimeError(f"git worktree add failed for {ref}: {proc.stderr}")


def copy_suite(suite_src: Path, tree: Path) -> None:
    dest = tree / "numerical_stability"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        suite_src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.o", "last-compare.md", "*.pyc"),
    )


def build_and_run(tree: Path) -> dict[str, str]:
    # Prefer make.linux / existing make.inc in the worktree.
    build = run(["make", "testNumericalStability"], cwd=tree)
    if build.returncode != 0:
        raise RuntimeError(
            f"build failed in {tree}:\n{build.stdout}\n{build.stderr}"
        )
    exe = tree / "testNumericalStability"
    proc = run([str(exe)], cwd=tree)
    # Non-zero exit is fine (failed/xpassed); we still parse outcomes.
    outcomes = parse_outcomes(proc.stdout + "\n" + proc.stderr)
    if not outcomes:
        raise RuntimeError(
            f"no OUTCOME lines from {exe}:\n{proc.stdout}\n{proc.stderr}"
        )
    return outcomes


def resolve_suite(suite_from: str, baseline_tree: Path, candidate_tree: Path, repo: Path) -> Path:
    if suite_from == "baseline":
        return baseline_tree / "numerical_stability"
    if suite_from == "candidate":
        return candidate_tree / "numerical_stability"
    path = Path(suite_from)
    if not path.is_absolute():
        path = (repo / path).resolve()
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="origin/main")
    p.add_argument("--candidate", default="HEAD")
    p.add_argument(
        "--suite-from",
        default="baseline",
        help="baseline | candidate | PATH to numerical_stability/",
    )
    p.add_argument("--work-dir", default="")
    p.add_argument("--report", default="numerical_stability/last-compare.md")
    p.add_argument("--strict-suite", action="store_true")
    p.add_argument("--keep-work-dir", action="store_true")
    p.add_argument(
        "--from-json",
        nargs=2,
        metavar=("BASE", "CAND"),
        help="Skip builds; compare two JSON outcome maps {name: result}",
    )
    args = p.parse_args(argv)

    if args.from_json:
        baseline = json.loads(Path(args.from_json[0]).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.from_json[1]).read_text(encoding="utf-8"))
        buckets = diff_outcomes(baseline, candidate)
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        write_report(report, buckets, args.from_json[0], args.from_json[1])
        if buckets.get(REGRESSION):
            return 1
        if args.strict_suite and buckets.get(SUITE_DRIFT):
            return 1
        return 0

    repo = Path.cwd()
    if not (repo / ".git").exists() and not (repo / "makefile").exists():
        print("Run from the GPc repository root.", file=sys.stderr)
        return 2

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="gpc-ns-"))
    work.mkdir(parents=True, exist_ok=True)
    baseline_tree = work / "baseline"
    candidate_tree = work / "candidate"

    try:
        ensure_worktree(repo, args.baseline, baseline_tree)
        ensure_worktree(repo, args.candidate, candidate_tree)
        suite_src = resolve_suite(args.suite_from, baseline_tree, candidate_tree, repo)
        if not suite_src.is_dir():
            raise RuntimeError(f"suite directory missing: {suite_src}")
        # Mount the chosen suite into both trees before build.
        copy_suite(suite_src, baseline_tree)
        copy_suite(suite_src, candidate_tree)

        base_out = build_and_run(baseline_tree)
        cand_out = build_and_run(candidate_tree)
        (work / "baseline_outcomes.json").write_text(json.dumps(base_out, indent=2), encoding="utf-8")
        (work / "candidate_outcomes.json").write_text(json.dumps(cand_out, indent=2), encoding="utf-8")

        buckets = diff_outcomes(base_out, cand_out)
        report = Path(args.report)
        if not report.is_absolute():
            report = repo / report
        report.parent.mkdir(parents=True, exist_ok=True)
        write_report(report, buckets, args.baseline, args.candidate)
        print(f"Wrote {report}")
        print(
            f"improvements={len(buckets.get(IMPROVEMENT, []))} "
            f"regressions={len(buckets.get(REGRESSION, []))} "
            f"debt={len(buckets.get(KNOWN_DEBT, []))} "
            f"drift={len(buckets.get(SUITE_DRIFT, []))}"
        )
        if buckets.get(REGRESSION):
            return 1
        if args.strict_suite and buckets.get(SUITE_DRIFT):
            return 1
        return 0
    finally:
        # Drop worktrees from the main repo index even if we keep files.
        run(["git", "worktree", "prune"], cwd=repo)
        if not args.keep_work_dir and not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
