#!/usr/bin/env python3
"""Unit tests for numerical_stability.compare outcome matrix (CIP-0019)."""

from __future__ import annotations

import unittest

from compare import (
    IMPROVEMENT,
    KNOWN_DEBT,
    REGRESSION,
    STABLE,
    SUITE_DRIFT,
    classify,
    diff_outcomes,
    parse_outcomes,
)


class TestClassify(unittest.TestCase):
    def test_improvement_xfail_to_pass(self):
        self.assertEqual(classify("xfailed", "passed"), IMPROVEMENT)
        self.assertEqual(classify("xfailed", "xpassed"), IMPROVEMENT)

    def test_regression_pass_to_fail(self):
        self.assertEqual(classify("passed", "failed"), REGRESSION)
        self.assertEqual(classify("xfailed", "failed"), REGRESSION)

    def test_stable_and_debt(self):
        self.assertEqual(classify("passed", "passed"), STABLE)
        self.assertEqual(classify("xfailed", "xfailed"), KNOWN_DEBT)

    def test_suite_drift(self):
        self.assertEqual(classify(None, "passed"), SUITE_DRIFT)
        self.assertEqual(classify("passed", None), SUITE_DRIFT)


class TestParseAndDiff(unittest.TestCase):
    def test_parse_outcomes(self):
        text = """
GPc numerical stability suite
OUTCOME name=a result=passed
OUTCOME name=b result=xfailed
noise
"""
        self.assertEqual(parse_outcomes(text), {"a": "passed", "b": "xfailed"})

    def test_diff_buckets(self):
        base = {"a": "xfailed", "b": "passed", "c": "xfailed"}
        cand = {"a": "passed", "b": "failed", "c": "xfailed", "d": "passed"}
        buckets = diff_outcomes(base, cand)
        self.assertEqual([t[0] for t in buckets[IMPROVEMENT]], ["a"])
        self.assertEqual([t[0] for t in buckets[REGRESSION]], ["b"])
        self.assertEqual([t[0] for t in buckets[KNOWN_DEBT]], ["c"])
        self.assertEqual([t[0] for t in buckets[SUITE_DRIFT]], ["d"])


if __name__ == "__main__":
    unittest.main()
