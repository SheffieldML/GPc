# Tutorial: Probabilistic Point Assimilation (`./ppa`)

CIP-0023 restores Nat King & Neil Lawrence’s **Probabilistic Point Assimilation** (PPA) as a full-data variational EM classifier alongside `./ivm`.

## Build and help

```bash
make ppa
./ppa -h
./ppa learn -h
```

## Small smoke learn

Fixture data (30 points from the MATLAB `demPpa1` set):

```bash
./ppa -v 1 learn -o classification -k rbf testdata/ppa/phase0.svml /tmp/phase0.model
./ppa display /tmp/phase0.model
```

## Gnuplot decision boundary

Two-input probit models plot like `./ivm gnuplot` (positive / negative points, full-data “active” set, 0.5 decision contour and 0.25/0.75 probability contours):

```bash
./ppa gnuplot -r 30 testdata/ppa/phase0.svml /tmp/phase0.model phase0
gnuplot phase0_plot.gp
# or: gnuplot -e "GPLOT_OUT='phase0.png'" phase0_plot.gp
```

Verification without MATLAB in CI:

```bash
make testPpa
./testPpa
```

## Notes

- PPA stores a dense \(N\times N\) kernel; it is not an IVM-style active-set method.
- Default path uses **scalar** site precision \(B\) and a **kernel M-step** (`ppaKernelLogLikelihood`). β updates are not wired in the CLI yet.
- Octave reference dumps and the upstream toolbox quirks are recorded in [`cip/cip0023/phase0_notes.md`](../../cip/cip0023/phase0_notes.md).
