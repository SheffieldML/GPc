---
id: "2026-11-15_gnuplot-term-and-style"
title: "Fix gnuplot terminal selection and modernise plot style"
status: "Proposed"
priority: "Low"
created: "2026-11-15"
last_updated: "2026-11-15"
category: "features"
related_cips: []
owner: "Neil Lawrence"
dependencies: []
tags:
- backlog
- features
- gnuplot
- visualisation
- macos
- gpy
---

# Task: Gnuplot terminal and modern style

## Description

Generated `*_plot.gp` scripts hard-code `set term x11` (classification paths in `gp.cpp`, `ivm.cpp`, `ppa.cpp`). On current macOS Homebrew builds, X11 is usually unavailable (`brew install gnuplot --with-x` is gone), so `gnuplot sinc_plot.gp` fails or opens nothing useful. Regression scripts often omit an explicit terminal entirely and rely on the install default.

Separately, the plots still look like early-2000s gnuplot defaults: bare line styles, no filled confidence band, no shared colour language with GPy. GPy’s matplotlib defaults (`GPy/plotting/Tango.py`, `matplot_dep/defaults.py`) give a clear target — Tango medium/light blue for mean and CI, black/x markers for data, dark red for inducing points.

Scope is small: make the scripts runnable on a modern gnuplot install, and restyle them so README example figures feel closer to GPy without pulling in a Python plotting stack.

## Acceptance Criteria

- [ ] Generated scripts pick a usable interactive terminal on macOS/Linux (prefer `qt` or `aqua` where present; fall back documented) instead of hard-coded `x11`
- [ ] Optional file output path uses a vector/raster terminal that works without X11 (e.g. `pngcairo` / `pdfcairo`) for README figures
- [ ] README notes match current Homebrew gnuplot install (drop obsolete `--with-x`)
- [ ] 1-D GP mean + ± (±2σ) use GPy-like colours: mean `mediumBlue` (`#3465a4`), CI fill/edge `lightBlue`/`darkBlue`, data markers black, inducing/`X_u` `darkRed`
- [ ] Classification / IVM / PPA / GPLVM scripts share the same palette and line/point conventions where applicable
- [ ] `./gp gnuplot … && gnuplot *_plot.gp` works on a stock Homebrew gnuplot (no XQuartz required)

## Implementation Notes

- Touch points: `gp.cpp` / `ivm.cpp` / `ppa.cpp` / `gplvm.cpp` gnuplot emitters; README examples under sinc / unitsquare / oil100.
- Prefer emitting a short preamble (terminal probe or `#` comment with override) rather than platform `#ifdef` only — users can `GNUTERM` / edit the `.gp`.
- Confidence band: gnuplot `filledcurves` between the ±2σ tables already written as `*_error_bar_data.dat`, instead of two bare lines.
- Keep `set term table` for contour extraction; only change the interactive / PNG display terminal after `reset`.

## Related

- Code: `gp.cpp` (`CClgp::gnuplot`), `ivm.cpp`, `ppa.cpp`, `gplvm.cpp`
- Docs: `README.md` (sinc gnuplot note ~L179)
- Style reference: GPy `GPy/plotting/Tango.py`, `GPy/plotting/matplot_dep/defaults.py`

## Progress Updates

### 2026-11-15

Task created with Proposed status after CIP-0012 land; macOS term mismatch and GPy-aligned styling called out as follow-ups.
