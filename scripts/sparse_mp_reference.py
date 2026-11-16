#!/usr/bin/env python3
"""CIP-0012 mpmath reference for whitened sparse StableFactor gates.

Reads a state dump (K_uu post-jitter, K_uf, m, beta, diagK) and recomputes
LL / q / selected grads at high precision. Pair with diagSparseStable dumps.

Scaffolding only: extend as dump format and gate thresholds land.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "state",
        nargs="?",
        help="Path to a diagSparseStable / diagDtcvar state dump (future)",
    )
    p.add_argument("--dps", type=int, default=60, help="mpmath decimal precision")
    args = p.parse_args(argv)

    try:
        import mpmath as mp
    except ImportError:
        print("mpmath is required: pip install mpmath", file=sys.stderr)
        return 2

    mp.mp.dps = args.dps
    if not args.state:
        print(
            "sparse_mp_reference: scaffold OK "
            f"(mpmath dps={args.dps}). Pass a state dump to evaluate."
        )
        return 0

    print(
        f"sparse_mp_reference: state loader not wired yet for {args.state!r}; "
        "see CIP-0012 Testing Strategy.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
