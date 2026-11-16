# Maintainer workflow

How planning artefacts relate to day-to-day work on GPc.

**Design**: [CIP-0007](../cip/cip0007.md) · tenets under `tenets/gpc/`

## Layers

| Layer | Location | Role |
| --- | --- | --- |
| WHY | `tenets/gpc/` | Design principles (combined export: `tenets/vibesafe-tenets.md`) |
| WHAT | `requirements/` | Testable requirements (REQ-0001–0004) |
| HOW | `cip/` | Code Improvement Plans (hex ids: `cip0001.md`, …) |
| DO | `backlog/` | Dated tasks linked to CIPs |
| DOCUMENT | `docs/` + [README](../README.md) | Compressed outcomes for maintainers and users |
| TUTORIALS | [`docs/tutorials/`](tutorials/) | Native CLI/demo tutorials (CIP-0015 Stage 1) |

## Day-to-day

1. Run `./whats-next` (or the vibesafe `whats_next.py` against this tree) for status and suggested next steps.
2. Validate structure: `validate_vibesafe_structure.py` (requirement statuses use `Validated`, not `Completed`).
3. Substantial changes: write or update a CIP first; name branches `cipXXXX-…`; reference the CIP in commits.
4. After a CIP is Closed, compress durable decisions into the targets in [`.vibesafe/documentation.yml`](../.vibesafe/documentation.yml), then set `compressed: true` in the CIP frontmatter.
5. User-facing build/test commands stay in the README; rationale and numerics policy live under `docs/`.

## Documentation system

Plain Markdown under `docs/`. No Sphinx/MkDocs requirement yet. If the maintainer guide set grows or hosted HTML is needed, revisit the `system` field in `.vibesafe/documentation.yml`.

## Dead-code policy

Do not leave unreachable tails after `throw`/`return`, empty stub bodies, or large commented-out archives that git history already preserves. Prefer a one-line pointer to a CIP (or backlog item) over commented-out implementations. Intentional `NotImplementedError` stubs that define API surface may remain; cover them with expect-throw tests rather than deleting them.

## Proposed modernization sequence

After the CIP-0001–0007 stabilization pass, a structural review ranked remaining C++ hygiene work as follows:

1. [CIP-0008](../cip/cip0008.md) — `CMatrix` copy/move assignment (ownership / double-free risk) — **Proposed**
2. [CIP-0009](../cip/cip0009.md) — explicit GP / GPLVM cache invalidation (reduce `mutable` working state) — **Closed**
3. [CIP-000A](../cip/cip000A.md) — `unique_ptr` for compound/tensor kernel components — **Proposed** (after 0008)
4. [CIP-000B](../cip/cip000B.md) — remove dead paths and fix contradicting comments — **Implemented** (parallel with coverage work)

Accept and implement 0008 then 000A unless a concrete bug forces a reorder; 000B may land independently.
