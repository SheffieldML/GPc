---
id: "0011"
title: "Users can load tutorial and modelling data via CSV and pods, with SVM-light still supported"
status: "In Progress"
priority: "Medium"
created: "2026-11-18"
last_updated: "2026-11-18"
related_tenets: ["ecosystem-interoperability", "flexibility-with-performance", "assert-and-verify"]
stakeholders: ["maintainers", "modellers", "GPy users", "CLI users"]
tags: ["interop", "data", "csv", "pods", "svml", "tutorials", "cli"]
---

# REQ-0011: Users can load tutorial and modelling data via CSV and pods, with SVM-light still supported

## Description

Training and tutorial **data** enter GPc through formats the wider SheffieldML / mlatcl stack already uses: dense **CSV/TSV** on disk for the native CLI and C++ demos, and **`pods.datasets`** in Python examples (same pattern as GPy). SVM-light (`.svml`) remains a supported reader for existing `examples/*.svml` workflows; it is no longer the preferred or documented-first path for new tutorials.

This requirement is about **observation matrices**, not model serialisation. Model interchange stays under [REQ-000A](./req000A_shared-model-interchange.md). Parsing stays at the I/O boundary (Flexibility Without a Performance Tax); loaders fail loudly on shape/format errors (Assert Invariants).

**Why this matters**: Interoperate With the GP Ecosystem still names SVM-light in the tenet text, but GPy tutorials fetch oil, olympic marathon, swiss roll, and related sets via [`pods`](https://github.com/sods/ods). Sparse SVM-light encoding is a poor fit for dense GP tutorial matrices and blocks easy sharing with pandas / NumPy / MATLAB without Anton Schwaighofer’s SVML toolbox. Aligning on CSV + pods removes that friction without inventing a private SheffieldML data schema.

**Who benefits**: CLI users following tutorials; Python façade users (CIP-001D Phase E); maintainers verifying against the same datasets GPy examples use.

## Acceptance Criteria

- [ ] Native CLI / `CClctrl` (or equivalent) can load dense **CSV/TSV** into `X` and `y` with a documented layout (e.g. last column = target, or paired `*.X.csv` / `*.y.csv`)
- [ ] File-format selection prefers CSV for new docs; SVM-light and (where compiled) MATLAB `.mat` remain selectable or auto-detected by extension
- [ ] At least one primary tutorial dataset (e.g. sinc or olympic marathon) is available as CSV under `examples/` and documented as the preferred CLI input
- [ ] Existing `.svml` files under `examples/` still load; README / tutorials that still cite `.svml` either keep working or are updated to CSV with a one-line SVML compatibility note
- [ ] Python examples / notebooks that need third-party data use **`pods`** as an optional extra (`gpc[examples]` or documented `pip install pods`), not a core `gpc` dependency — consistent with [REQ-000E](./req000E_gpy-facing-python-facade.md) / CIP-001D Phase E
- [ ] Loaders assert dimension consistency (`X` rows == `y` rows; non-empty) and raise typed file/format errors on malformed input — no silent truncation
- [ ] Non-goals for first delivery: OpenML/Parquet as a native C++ dependency; replacing `.model` streams; putting CSV/`pods` I/O in the SCG inner loop; mandating HDF5 for tutorials (optional later via existing `_HDF5` paths)

## Notes (Optional)

HOW: [CIP-0022](../cip/cip0022.md) (Proposed). Orthogonal to [REQ-000A](./req000A_shared-model-interchange.md) (models) and complementary to [REQ-0009](./req0009_gpy-tutorial-parity-native.md) / [CIP-0015](../cip/cip0015.md) (native tutorials) and [CIP-001D](../cip/cip001D.md) Phase E (`pods` notebooks).

## References

- **Related Tenets**: ecosystem-interoperability, flexibility-with-performance, assert-and-verify
- **External**: [pods](https://github.com/sods/ods), GPy `examples/` + `pods.datasets`, SVM-light format (legacy)

## Progress Updates

### 2026-11-18
Requirement created after agreeing CSV + pods as the preferred data path; SVM-light retained for compatibility.

### 2026-11-18 (implementation)
CIP-0022 accepted and in progress on `cip0022-csv-pods-data`: `readCsvDataFile`, extension auto-detect, example CSVs, `testClctrl` parity.
