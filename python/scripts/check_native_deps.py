#!/usr/bin/env python3
"""CIP-002C Phase B: assert the installed _gpc has no absolute Homebrew BLAS paths.

Run after installing a repaired wheel (delocate / auditwheel). Contributor editable
builds that intentionally RPATH Homebrew are out of scope.
"""

from __future__ import annotations

import pathlib
import platform
import re
import subprocess
import sys


def _main() -> int:
    import gpc._gpc as _gpc  # noqa: PLC0415 — after wheel install

    so = pathlib.Path(_gpc.__file__).resolve()
    system = platform.system()
    print(f"check_native_deps: {_gpc.__file__} ({system})")

    if system == "Darwin":
        out = subprocess.check_output(["otool", "-L", str(so)], text=True)
        bad = [
            line
            for line in out.splitlines()
            if re.search(r"/opt/homebrew|/usr/local/opt/", line)
        ]
        if bad:
            print("absolute Homebrew dependencies remain:", file=sys.stderr)
            print("\n".join(bad), file=sys.stderr)
            return 1
        # Also scan bundled .dylibs next to the extension if present.
        dylib_dir = so.parent / ".dylibs"
        if dylib_dir.is_dir():
            for dylib in dylib_dir.glob("*.dylib"):
                dout = subprocess.check_output(["otool", "-L", str(dylib)], text=True)
                dbad = [
                    line
                    for line in dout.splitlines()[1:]  # skip self id
                    if re.search(r"/opt/homebrew|/usr/local/opt/", line)
                ]
                if dbad:
                    print(f"absolute Homebrew deps in {dylib.name}:", file=sys.stderr)
                    print("\n".join(dbad), file=sys.stderr)
                    return 1
        print("macOS: no absolute Homebrew load paths")
        return 0

    if system == "Linux":
        out = subprocess.check_output(["ldd", str(so)], text=True)
        if "not found" in out:
            print("ldd reports missing libraries:", file=sys.stderr)
            print(out, file=sys.stderr)
            return 1
        # Prefer bundled openblas (auditwheel *.libs) over bare system absolute.
        for line in out.splitlines():
            if "openblas" not in line.lower():
                continue
            if "not found" in line:
                print(line, file=sys.stderr)
                return 1
            # System OpenBLAS without a wheel .libs path is a soft warning only if
            # auditwheel left it external; fail when clearly unresolved.
            print(f"  ldd: {line.strip()}")
        print("Linux: ldd OK")
        return 0

    print(f"skip native-dep check on {system}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
