# Compression Checklist: CIP-0027, 0033, 0034 (+ 0019)

**CIPs**: [0027](../cip/cip0027.md) · [0033](../cip/cip0033.md) · [0034](../cip/cip0034.md) · [0019](../cip/cip0019.md)  
**Type**: Architecture / likelihood–inference (0027, 0033, 0034) · Numerics CI (0019)  
**Closed**: 2026-11-20 … 2026-11-21  
**Compression date**: 2026-11-21  
**Spec**: [`.vibesafe/documentation.yml`](../.vibesafe/documentation.yml)

## Pre-Compression Review

- [x] Read closed CIPs
- [x] Key decisions identified (likelihood-owned noise; EP/Laplace on FTC; site workspaces; soak New:0 debt)
- [x] Requirements: REQ-0014, REQ-0019 (Validated); REQ-0002 (0019)
- [x] Consulted documentation.yml targets

## Documentation Updates

- [x] [docs/target-architecture.md](target-architecture.md) — realised rules 1–4; CIP-0033/0034 map; update “GPc today”
- [x] [docs/python-facade.md](python-facade.md) — `GPClassification` (EP/Laplace FTC)
- [x] [docs/numerics.md](numerics.md) — pointer already covers CIP-0019 soak / KNOWN_DEBT; mark 0019 compressed
- [x] Tutorials / checklist already updated under 0033/0034 PRs
- [x] Set `compressed: true` on CIP-0027, 0033, 0034, 0019
- [x] Close CIP-0034 frontmatter (`Closed`)

## Traceability

Architecture and façade pages link back to CIPs and REQ-0019 / REQ-0014. CIP frontmatter retains `related_requirements` / `related_cips`.
