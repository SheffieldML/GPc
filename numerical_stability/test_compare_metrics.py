#!/usr/bin/env python3
"""Unit tests for numerical_stability.compare_metrics (CIP-001A)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from compare_metrics import (
    BACKWARD,
    FORWARD,
    OVER_BUDGET,
    STABLE,
    classify_pair,
    diff_metrics,
    main as compare_main,
)


class TestClassifyPair(unittest.TestCase):
    def test_forward_smaller_residual(self):
        self.assertEqual(
            classify_pair(1e-4, 1e-5, 1e-3, "B_linalg"),
            FORWARD,
        )

    def test_backward_larger_residual(self):
        self.assertEqual(
            classify_pair(1e-5, 1e-4, 1e-3, "B_linalg"),
            BACKWARD,
        )

    def test_stable_within_noise(self):
        self.assertEqual(
            classify_pair(1e-6, 1.05e-6, 1e-3, "B_linalg"),
            STABLE,
        )

    def test_over_budget(self):
        self.assertEqual(
            classify_pair(1e-3, 2e-2, 1e-2, "D_fd"),
            OVER_BUDGET,
        )

    def test_negative_threshold_over(self):
        # lnCum deep left: budget -1e8; value must stay below budget.
        self.assertEqual(
            classify_pair(-2e8, -5e7, -1e8, "C_special"),
            OVER_BUDGET,
        )


class TestDiffMetrics(unittest.TestCase):
    def test_diff_and_cross_cell(self):
        base = {
            "cell": {"label": "ubuntu-openblas"},
            "metrics": {
                "a": {"category": "A_machine", "value": 1e-12, "budget": 1e-10, "unit": "abs"},
                "b": {"category": "B_linalg", "value": 1e-6, "budget": 1e-3, "unit": "abs"},
            },
        }
        head = {
            "cell": {"label": "ubuntu-openblas"},
            "metrics": {
                "a": {"category": "A_machine", "value": 1e-14, "budget": 1e-10, "unit": "abs"},
                "b": {"category": "B_linalg", "value": 5e-5, "budget": 1e-3, "unit": "abs"},
                "c": {"category": "F_model", "value": 0.1, "budget": 1e-4, "unit": "rel"},
            },
        }
        buckets = diff_metrics(base, head)
        self.assertEqual([r["name"] for r in buckets[FORWARD]], ["a"])
        self.assertEqual([r["name"] for r in buckets[BACKWARD]], ["b"])
        self.assertEqual([r["name"] for r in buckets["new"]], ["c"])

        bad_head = dict(head)
        bad_head["cell"] = {"label": "macos-accelerate"}
        with self.assertRaises(ValueError):
            diff_metrics(base, bad_head)
        # Override allowed
        diff_metrics(base, bad_head, force_cross_cell=True)

    def test_cli_no_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            head = td_path / "metrics.json"
            head.write_text(
                json.dumps(
                    {
                        "cell": {"label": "x"},
                        "metrics": {
                            "m": {
                                "category": "A_machine",
                                "value": 1.0,
                                "budget": 2.0,
                                "unit": "abs",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            md = td_path / "trend.md"
            rc = compare_main(
                [str(td_path / "missing.json"), str(head), "--md", str(md)]
            )
            self.assertEqual(rc, 0)
            self.assertIn("No baseline", md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
