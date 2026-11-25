---
id: "2026-11-25_cip003b-gp-sample-cli"
title: "CIP-003B: gp sample command"
status: "Completed"
priority: "High"
created: "2026-11-25"
last_updated: "2026-11-25"
category: "features"
related_cips: ["003B"]
owner: "Neil Lawrence"
dependencies: ["2026-11-25_cip003b-csv-sample-output"]
tags:
- backlog
- hmc
- sampling
- cli
---

# Task: `gp sample` command

## Description

Add a `sample` command to the `gp` executable that loads a trained exact-GP model, attaches priors, runs HMC and writes a CSV of samples. Noise priors are not stored in the `.model` stream (that needs a format version bump), so priors are supplied on the command line.

Usage: `gp [options] sample example_file [model_file] [samples_file]`.

## Acceptance Criteria

- [x] Options: `-n` number of samples, `-b` burn-in (step-size adaptation), `-e` initial step size, `-l` leapfrog steps, `-p path=prior` repeatable, `--fix path` repeatable, `--no-jacobian`, `--allow-improper`, `--unconstrained`
- [x] Prior specs `gamma:a,b` and `gaussian:precision` on kernel paths and `likelihood.variance`
- [x] Refuses an improper target with a message naming the parameters, unless `--allow-improper`
- [x] Writes CSV to the named file, or stdout with `-`
- [x] End-to-end run on a toy data set: posterior mean of each parameter is finite, acceptance rate reported, CSV loads
- [x] Help text for the command

## Implementation Notes

- Uses `CGp::addPriorByPath` and `CKern::addPriorAt` so priors can go on compound kernel components.
- Only exact FTC models are supported; other models fail with the `NotImplementedError` from `CGp::sampledParams`.

## Related

- CIP: 003B (Phase 5)

## Progress Updates

### 2026-11-25
Task created and started.

### 2026-11-25
Implemented and verified: testHmc (CSV round trip, addPriorByPath) passes; end-to-end `gp sample` run on a toy GP gives 3000 samples, acceptance 0.89, positive constrained values. Also fixed unset betaTransform for FTC models read from a stream (gp relearn segfault).
