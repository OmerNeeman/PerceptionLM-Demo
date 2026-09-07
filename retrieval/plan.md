# Plan — Aerial Tile Retrieval

Tier **Standard**: sequential gated stages, one in flight. No stage starts
until the previous is green with pasted evidence.

**Gate status: APPROVED 2026-09-07.** Owner directed implementation to begin in
a fresh session — read `HANDOFF.md` first.

**Ordering principle: vertical slice before breadth.** Get one AOI working end
to end and playable, *then* fan out. The full 8-scene pyramid is 108,542 tiles;
one AOI is 11,109. There is no reason to spend the 10x before the pipeline has
ever returned a correct result.

---

## Milestones

| | Milestone | Stages | What exists at the end |
|---|---|---|---|
| **M1** | **First running demo** | S0-S5 | One AOI (`X605_Y3388`, the tent camp) indexed at 3 scales, queryable, exported as a standalone HTML you can play with |
| M2 | Full index | S6 | All 8 scenes, all AOIs, per-AOI exports |
| M3 | Measured | S7 | recall@k on `leb` against the 45-class labels |
| M4 | Optional A/B | S8 | ColQwen2.5 late-interaction comparison |

---

## Stages

| # | Stage | Proves | Depends | Dispatch |
|---|---|---|---|---|
| S0 | **Env bring-up + embedder bake-off** | F-0, F-2, F-3, N-3 | — | new session |
| S1 | **Source classifier + inventory** | D-1, D-2 | — | subagent |
| S2 | **Tile planner + geo round-trip** | F-1, D-3, D-4, F-10 | S1 | subagent |
| S3 | **Embedding pipeline (one AOI)** | F-1a, F-2, N-2, N-4 | S0, S2 | new session |
| S4 | **Retrieval core** | F-3, F-4, F-5, F-6, N-1 | S3 | subagent |
| S5 | **HTML exporter + UX** → **M1** | F-2a, F-8, N-6, U-1..U-7 | S4 | new session |
| S6 | **Fan out to all 8 scenes** | F-1 at scale, F-7 | S5 | new session |
| S7 | **Evaluation report** | E-1, E-2 | S6 | subagent |
| S8 | **ColQwen2.5 A/B** *(optional)* | F-11 | S6 | new session |
| S9 | **Cold start + pre-delivery** | N-5, N-7 | all | fresh session |

### Why this order

**S0 first because it is the only irreversible decision.** Changing embedder
invalidates every vector in the index, so it is settled before anything is
built — and settled by measurement (F-0), comparing PE-Core-G14-448,
PE-Core-L14-336 and RemoteCLIP on the same queries over one subregion. Minutes
of compute to avoid a wrong multi-hour build. It also proves the environment:
`PYTHONNOUSERSITE=1`, fp16 not bf16, GPU 0 actually visible.

**S1 implements what `docs/DATA.md` already measured.** The facts are known;
the stage exists to make them *reproducible in code* — the >= 3 bands and > 64
distinct values rule that separates 12 source rasters from ~230 derived ones.
Do not re-run the survey.

**S2 builds COGs for `leb` only.** The six 0.1 m scenes are already tiled
256x256 (~2.9x read amplification). `leb`'s strip layout is the anomaly at
~45x. Converting all eight would be wasted work.

**S3 covers one AOI only.** `X605_Y3388` — the tent camp, 11,109 tiles across
`[448, 224, 112]`. It is the scene the previous attempt was aiming at
(`vlm_logs` prompts were "tents" / "find all tents in the frame"), it has
individually identifiable cars, and it is the canonical demo input in the
sibling project's own docs.

**S5 is the milestone.** After it, the owner has something to play with. Every
later stage widens or measures; none is required for the demo to exist.

**S7 is late and non-gating** — the owner set a qualitative gate; these numbers
inform the product-scoping decision rather than blocking a stage.

**S8 is optional and can never ship.** A late-interaction index is ~256 KB per
tile, ~2.8 GB for one AOI; it cannot be exported. Its only purpose is to tell
us whether the pyramid is leaving small-object recall on the table.

---

## WIP cap

**One stage in flight.** Ceiling per stage: one dispatch plus one re-dispatch
after a send-back. A second send-back on the same stage stops and escalates —
never burn past it.

---

## Progress ledger

| Stage | Status | Last action | Next action |
|---|---|---|---|
| intake | done | 10 questions ratified across 3 rounds | — |
| recon | done | 8 source scenes measured; `docs/DATA.md` written | trust it, do not repeat |
| research | done | patch-token late interaction rejected on published evidence; pyramid adopted | RS-CLIP survey still owed (spend limit) |
| gate | **approved** | owner signed spec, F-7 amendment, `sin` exclusion, E-1 upgrade | — |
| S0 | **ready** | brief written: `briefs/S0.md` | dispatch in the fresh session |

---

## Tech-debt ledger

Empty. Entries need owner sign-off and may record only genuine imperfections —
an unmet `spec.md` criterion is never ledgerable; that is a send-back or an
amendment.

---

## Ship & operate

Single-user local system. `git revert` is the rollback. The release path is the
cold start documented in `CLAUDE.md`. The index is rebuildable from source
imagery, so it needs no backup. Observability is the build log: tiles/sec,
wall time, failure counts.
