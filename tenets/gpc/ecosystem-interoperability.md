---
id: "ecosystem-interoperability"
title: "Interoperate With the GP Ecosystem"
status: "Active"
created: "2026-11-13"
last_reviewed: "2026-11-13"
review_frequency: "Annual"
conflicts_with: ["flexibility-with-performance"]
tags:
  - tenet
  - interoperability
  - formats
  - usability
  - legacy
---

## Tenet: ecosystem-interoperability

**Title**: Interoperate With the GP Ecosystem

**Description**: GPc is one node in a wider Gaussian process toolchain. Users should move data and models across that toolchain without bespoke converters for every experiment. The durable commitment is shared, inspectable formats: plain-text `.model` streams people can read, and SVM-light-style data shared with other SheffieldML GP software. Portability across compilers and OSes (gcc, MSVC, architecture make files) is part of the same idea: meet users where their scientific stack already is.

MATLAB bridges (`CMatInterface`, mex, `.mat` fixtures) were the original interoperability path and still explain much of the test and visualisation design. They also date the setup. Near-term work is to keep the current stack working and verified, including whatever MATLAB-gated paths still unlock today’s fixtures. Dropping or replacing those bridges belongs in a later release, with an explicit migration story so users and fixtures are not stranded.

**Quote**: *"Speak the formats the community already uses."*

**Examples**:
- Loading tutorial data from SVM-light files shared with other SheffieldML GP software
- Writing human-readable `.model` files via `CStreamInterface` versioned key-value streams
- Keeping stream field names versioned so old `.model` files remain loadable
- Retaining MSVC and multi-architecture build paths so interoperability is not Linux-only
- Getting the present build and MATLAB-dependent verification working before deciding what to drop
- Planning mex/`CMatInterface` retirement as a future-release CIP once a replacement verification path exists

**Counter-examples**:
- Inventing a binary-only checkpoint format with no text escape hatch
- Changing stream field names without a version bump, breaking old `.model` files
- Deleting MATLAB bridges in the middle of a "make it work again" effort, leaving tests unable to run
- Treating mex maintenance as sacred merely because it shipped in 2005
- Using a data format that only one private preprocessing script understands

**Conflicts**:
- Can conflict with "Flexibility Without a Performance Tax" when I/O bridges or text formats add cost
- Resolution: Keep interoperability at the boundaries (load, save, tests, optional language bridges); never in the SCG inner loop
- Can conflict with maintenance cost when legacy MATLAB tooling bit-rots
- Resolution: Stabilize and verify first; retire MATLAB bridges in a later release once durable formats and a non-MATLAB verification path are in place

**Version**: 1.2 (2026-11-13)
