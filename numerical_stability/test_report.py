#!/usr/bin/env python3
"""Unit tests for numerical_stability.report (CIP-0019 + CIP-001A)."""

from __future__ import annotations

import unittest

from report import build_metrics_doc, parse_metrics, parse_outcomes, render_markdown


class TestReport(unittest.TestCase):
    def test_parse_and_status_debt(self):
        text = """
OUTCOME name=a result=passed
OUTCOME name=b result=xfailed
OUTCOME name=c result=passed
"""
        outcomes = parse_outcomes(text)
        md = render_markdown(outcomes, ref="abc")
        self.assertIn("known debt", md)
        self.assertIn("`xfailed` | 1", md)
        self.assertIn("`passed` | 2", md)
        self.assertIn("**Ref:** `abc`", md)

    def test_status_regressions(self):
        md = render_markdown({"a": "failed", "b": "passed"})
        self.assertIn("regressions", md)
        self.assertIn("❌ `a`", md)

    def test_status_clean(self):
        md = render_markdown({"a": "passed", "b": "passed"})
        self.assertIn("clean", md)

    def test_parse_metrics_and_section(self):
        text = """
OUTCOME name=a result=passed
METRIC name=pdinv_resid_cond_100 category=B_linalg value=1.2e-8 budget=1e-4 unit=abs cell=ubuntu-openblas
METRIC name=logDet_scale_relerr category=A_machine value=1e-12 budget=1e-8 unit=rel cell=ubuntu-openblas
"""
        metrics = parse_metrics(text)
        self.assertEqual(set(metrics), {"pdinv_resid_cond_100", "logDet_scale_relerr"})
        self.assertEqual(metrics["pdinv_resid_cond_100"]["category"], "B_linalg")
        md = render_markdown(parse_outcomes(text), metrics=metrics)
        self.assertIn("Stability metrics", md)
        self.assertIn("`pdinv_resid_cond_100`", md)
        doc = build_metrics_doc(metrics, git_sha="deadbeef")
        self.assertEqual(doc["git_sha"], "deadbeef")
        self.assertIn("pdinv_resid_cond_100", doc["metrics"])


if __name__ == "__main__":
    unittest.main()
