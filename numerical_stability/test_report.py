#!/usr/bin/env python3
"""Unit tests for numerical_stability.report (CIP-0019)."""

from __future__ import annotations

import unittest

from report import parse_outcomes, render_markdown


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


if __name__ == "__main__":
    unittest.main()
