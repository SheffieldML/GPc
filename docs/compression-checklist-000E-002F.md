# Compression Checklist: CIP-000E, 0012, 001A, 0020, 001D, 002F (batch)

**CIPs**: [000E](../cip/cip000E.md) · [0012](../cip/cip0012.md) · [001A](../cip/cip001A.md) · [0020](../cip/cip0020.md) · [001D](../cip/cip001D.md) · [002F](../cip/cip002F.md)  
**Type**: Feature/numerics (000E, 0012, 001A) · Infrastructure (0020) · Guides / Python (001D, 002F)  
**Closed**: 2026-11-16 … 2026-11-20  
**Compression date**: 2026-11-20  
**Spec**: [`.vibesafe/documentation.yml`](../.vibesafe/documentation.yml)

## Pre-Compression Review

- [x] Read closed CIPs
- [x] Key decisions identified
- [x] Tenets/requirements noted (REQ-0002, 0001, 0007, 000E, 0017)
- [x] Consulted documentation.yml targets

## Documentation Updates

- [x] [docs/numerics.md](numerics.md) — CIP-000E / 0012 lineage note; CIP-001A metric categories + OUTCOME vs METRIC split
- [x] [docs/build-and-ci.md](build-and-ci.md) — CIP-0020 already present; pointer to python-facade + CIP-0021 cutover
- [x] [docs/python-facade.md](python-facade.md) — CIP-001D locked decisions + CIP-002F help/`__doc__` policy (new)
- [x] [docs/workflow.md](workflow.md) — PYTHON layer row
- [x] [`python/README.md`](../python/README.md) — pointer to python-facade.md
- [x] Set `compressed: true` on CIP-000E, 0012, 001A, 0020, 001D, 002F

## Traceability

Each docs page links back to the source CIPs and related requirements. CIP frontmatter retains `related_requirements` / `related_cips`.
