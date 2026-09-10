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
| S1a | **Config + device abstraction** + draft `INSTRUCTIONS.md` (setup half) | **N-8**, N-9 (part) | S1 | subagent |
| S2 | **Tile planner + geo round-trip** | F-1, D-3, D-4, F-10 | S1a | subagent |
| S3 | **Embedding pipeline (one AOI)** | F-1a, F-2, N-2, N-4 | S0, S2 | new session |
| S4 | **Retrieval core** | F-3, F-4, F-5, F-6, **F-2a**, N-1 | S3 | subagent |
| S5 | **HTML exporter + UX** → **M1** | F-8, N-6, **D-5**, U-1..U-7 | S4 | new session |
| S6 | **Fan out to all 8 scenes** | F-1 at scale, F-7 | S5 | new session |
| S5a | **Tile Q&A interface** *(scope open — see below)* | **F-9**, **U-8** | S5 | subagent |
| S7 | **Evaluation report** | E-1, **E-1a**, E-2 | S6 | subagent |
| S8 | **ColQwen2.5 A/B** *(optional)* | F-11 | S6 | new session |
| S9 | **Cold start + pre-delivery + `INSTRUCTIONS.md`** | N-5, N-7, **N-9** | all | fresh session |

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

### `INSTRUCTIONS.md` is written in three touches, not one

Each touch happens when its content first becomes **true**, because a
portability document written ahead of portable code is fiction:

1. **At S1a** — install, environment, data-root config, and the device/dtype
   rules. This is the half that is genuinely cross-platform and testable as
   soon as N-8 lands, and it is the half most likely to block a foreign
   machine.
2. **At M1 (after S5)** — the real run commands: build an index, query it,
   produce the export. Written from commands that have actually been run.
3. **At S9** — **verified by execution.** A fresh session with no access to
   this conversation follows it on a foreign machine and reports where it got
   stuck. Anything it had to ask is a defect in the document, not a question to
   answer in chat.

**S1a exists because portability is a code property, not a documentation
task** *(owner request, 2026-09-07)*. An audit of S0's and S1's output found
`float16` hardcoded in `src/embedders.py` and the data root hardcoded in
`src/inventory.py`. No `INSTRUCTIONS.md`, however good, can make that run on a
Windows machine with an Ampere card — and on Ampere and later, bf16 is the
*better* choice, so the correct dtype is the opposite of this machine's.

It runs **before S2**, not at S9, for one reason: S2, S3, S4 and S5 would each
copy the pattern, and retrofitting five modules is strictly more work than
fixing two. It is deliberately small — a config module, a device/dtype
function, and a retrofit of the two existing modules — and it is the only stage
permitted to edit S0's and S1's files.

**F-2a moves from S5 to S4** *(PM decision, 2026-09-07)*. The PCA-to-128-d
retrieval@10 overlap is a **retrieval measurement**, and S4 is the stage that
owns retrieval maths and already has the full-precision vectors in hand. Doing
it at S5 would mean discovering the accuracy cost of the size budget *while
building the exporter*, with the milestone in the way. Measured at S4, S5's
exporter just consumes an already-validated basis. The requirement itself is
**unchanged** — only where it is proven moved. It must be measured **per
scale**, since F-4 now ranks per scale and a single pooled overlap number
would hide a scale-specific loss.

**S5 is the milestone.** After it, the owner has something to play with. Every
later stage widens or measures; none is required for the demo to exist.

**S6 is worth more than "optional breadth."** The owner will query damage
(rubble, debris, collapsed roof), and that content is **not present** on the
`X605_Y3388` demo AOI — it is in `leb`'s 2022->2025 pair and `gaza`
(`notes.md#owner-ratify-2`). M1 still ships on one AOI, but S6 is where half
the intended query vocabulary becomes answerable at all.

### Two holes found by a cross-artifact check, 2026-09-07

Both were found by mechanically diffing `spec.md` against this file. Recording
how they were found, because the check is worth repeating at each gate.

**1. F-9 had no owning stage at all.** The swappable tile-Q&A interface has a
coverage-matrix proof (`test_qa_interface`) and a UX requirement that depends
on it (U-8), but no stage in this plan built it. Now **S5a**. It was invisible
because U-8 was written into S5 as part of the range "U-1..U-8", which read as
covered.

**2. F-0, F-1a, F-2a, F-11 and E-1a had no coverage-matrix row.** Added. F-1a
and F-2a are load-bearing (one embedder per index; the PCA accuracy cost) and
were relying on prose alone.

### F-8 and U-8 conflict — flagged, owner decision needed before S5

**F-8** requires the export to be a single file that opens and queries **with
no server and no network**, asserting **zero external `src`/`href`**.
**U-8** requires clicking a result to give a question box whose answer appears
attached to that tile.

**A standalone offline file cannot run a VLM.** Text-to-tile retrieval works
offline — the embeddings and the PCA basis are embedded, and a query vector can
be projected in-page. Q&A cannot: it needs a live model.

So one of three things is true, and the spec does not say which:
- **(a)** Q&A is a **local-app feature only**; the export is retrieval-only and
  U-8 narrows to "a tile opens larger" without the question box. *PM
  recommendation* — it keeps F-8's offline guarantee, which is the property
  that makes the demo shareable, and the owner's gate is typing queries and
  looking at results.
- **(b)** The export gets a question box that calls a local endpoint, and F-8's
  "no network" assertion narrows to "no *third-party* network".
- **(c)** Q&A is deferred past M1 entirely.

**Not resolved unilaterally** — it is a scope question about what M1 contains.
To be put to the owner at the **S4 gate**, which is the last point before the
S5 brief has to commit. S5a is parked until then; it does not block S1-S5.

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
| research | done | patch-token late interaction rejected on published evidence; pyramid adopted | **RS-CLIP debt discharged empirically by S0** — RemoteCLIP beat PE Core on our own imagery; GeoRSCLIP/SkyScript remain unsurveyed but are drop-in under F-1a |
| gate | **approved** | owner signed spec, F-7 amendment, `sin` exclusion, E-1 upgrade | — |
| anchor | **ratified** | primary user = owner scoping a product; licence fork and query vocabulary settled | `CLAUDE.md` pending flag removed |
| S0 | **GREEN** | **RemoteCLIP-ViT-L-14 (768-d) chosen by measurement.** PM re-ran 12/12 and re-read the vehicle crops independently; both worker concerns put to owner and decided; licence verified Apache 2.0 | — |
| S1 | **GREEN** | send-back closed. PM verified against the **reviewer's own** fixtures, not the worker's: G1/G2 (99.5% different) now rejected, true contained crop still caught, `leb` multi-date pair intact, geographic path in metres, guard fires at 24.6% on EPSG:3994 and is silent on all 8 scenes, inventory byte-identical across rebuilds. **39 tests green on PM re-run** | — |
| S1a | **GREEN** | N-8 met. PM re-verified the dtype rule at 6 fabricated cases (cc 7.5 -> fp16, cc 8.0/8.6/9.0 -> bf16, cpu/mps -> fp32); `src/` greps clean; config fails loudly naming the env var; `posix_key` canonicalises Windows separators. **44 tests green on PM re-run.** `INSTRUCTIONS.md` setup half written, one real defect found and fixed by PM (`einops` gap) | — |
| S2 | **GREEN** | counts verified from the **regenerated artifacts** (108,542 / 11,109 / 1,012), COGs byte-identical over 6 random windows with 45x -> ~1.3-2.9x amplification, **79 tests green**. One disclosed incident (a test wrote under the data root) forensically verified as zero-impact; standing rule added to `CLAUDE.md` | — |
| S3 | **GREEN** | 9,631 vectors over the demo AOI at 242 tiles/sec. Mandatory review found 3 defects, all fixed and PM-verified (NaN/Inf now raise, resume validates state, sanity bound now tested). **PM ran a real end-to-end query: "a car" returns 5/5 genuine vehicles.** N-4 amended, owner-signed | — |
| S4 | **GREEN** | 116 tests. PM verified the acceptance criterion that matters: `index/tileplan/*.json` **survived a full suite run** at 108,542 across 3 scales. Export 384-d int8, 3.70 MB. Latency 9-18 ms vs 200 ms. U-3 amended, `EXPORT_DIM` raised |
| S5 | **GREEN -> M1** | local app live. **135 tests.** PM ran it: `a car` +0.2650 at 112 px through the API, matching the direct measurement; 9 abuse cases, **zero tracebacks**; first query 7.9 s, then 20-70 ms | owner explores |
| S6 | **GREEN** | all 8 scenes: 104,374 + 4,168 = **108,542** exact. `leb` splits 20,944/20,944 so F-7 halves precisely. N-1 **65.7 ms** end-to-end vs 200 ms. 237 tiles/sec | — |
| S5b | **GREEN (handback unfinished)** | `export.html` **10.13 MB** / 16 MB cap, **zero external refs**, basemap + client-side crop, U-5 caveat intact. Agent died before the size table and ranking-parity check | PM to re-verify parity |
| S5c | **GREEN** | export rebuilt: opens with results not a tag cloud; `leb` exportable at 128-d with a 24 px legibility floor. Both parity **1.000**, both zero external refs, 10.13 / 14.93 MiB | — |
| S11a | **GREEN** | PE-Core-L14-336 indexed over the identical 104,374 tiles; model-scoped layout; RemoteCLIP moved not re-embedded. **171 tests.** S0's vehicle gap does not survive full scale (both 5/5); control margin shown to be an artifact of the control set | — |
| S11b | **in flight** | model selector + side-by-side compare view | PM verification |
| polish | **not done** |
| S1a | **queued** | added at owner request; portability audit found `float16` and the data root hardcoded | brief + dispatch after S1 gates |

| M1 hand-over | pending | owner will explore the demo themselves once PM has verified it | after S5 |

**Owner decisions taken at the S0 gate** (both were unspecified points the
worker correctly refused to invent):
1. **Ranking is per scale**, three rankings labelled by ground extent — not one
   pooled ranking. 112 px would otherwise dominate 9/9 queries.
2. **The empty state uses a relative gap**, not an absolute cosine cut-off. No
   fixed threshold is usable; a known-absent query outscored a real one.

**Carried into S3/S4 as measured expectations, not bugs:** beige corrugated
metal is retrieved by `sand`; attribute queries do not discriminate (`a white
car` ~ `car`, and the colour word makes results *worse*); `tent` verdicts on
the calibration subregion are base-rate inflated.

---

## Tech-debt ledger

Entries need owner sign-off and may record only genuine imperfections — an
unmet `spec.md` criterion is never ledgerable; that is a send-back or an
amendment.

| # | Debt | Why taken | Cost to carry | Sign-off |
|---|---|---|---|---|
| ~~1~~ | ~~RemoteCLIP's licence terms are unverified~~ | **DISCHARGED 2026-09-07, same day.** Verified **Apache 2.0** at the upstream repo. No licence cost to the measured winner; the fallback was never needed | — | — |
| 2 | **Training-corpus provenance not diligenced.** RemoteCLIP's weights are Apache 2.0, but its training corpora (RET-3 / SEG-4 / DET-10, aggregated from third-party remote-sensing datasets) do not state their own terms | Irrelevant to a scoping demo; relevant to a commercial launch. Diligencing dataset lineage is a legal task, not an engineering one, and would stall M1 for a question M1 exists to inform | One legal review, only if the product is built. Does not affect the demo, the index, or any spec criterion | Owner sign-off needed if carried past a build decision |
| 3 | **Sub-pixel-offset duplicates fail open.** A duplicate shifted 0.5 px (5 cm) is admitted, so the same ground is indexed twice | The dedup offset test requires integrality to 1e-3 px. Fixing it means resampling comparison, which is real work; the consequence is an inflated index and slightly skewed retrieval, not lost data — unlike finding 1, which silently *dropped* a scene and is being fixed | Duplicate tiles in results. Plausible route in is a re-gridded export, which the corpus does not currently contain | Owner sign-off needed if it ever appears |
| 4 | **D-1 cannot distinguish derived output from imagery.** Three segmentation visualisations pass the pixel rule; so does uniform random noise | The rule tests photographic value statistics, not provenance — that is what makes it filename-independent, which is the property worth far more. The three are excluded incidentally (PNG carries no georeference) | If such a file ever arrives as a georeferenced 10 cm GeoTIFF it would be indexed as source — the project's signature failure. Recorded in `spec.md` D-1 as a known exposure | PM; escalate if the corpus gains georeferenced visualisations |
| 5 | **Subtle stored-vector corruption is undetectable.** A vector 0.5% off-norm — 20x fp16's noise ceiling, 500x the spec tolerance — passes the 1e-2 sanity bound and is renormalised to a perfect 1.0 on load | The bound cannot be tightened: anything between ~3e-4 and 1e-2 is genuinely indistinguishable from fp16 quantisation noise, so tightening would start rejecting healthy vectors. Gross breaks (>1%) and non-finite values **do** raise | Closing it means a per-vector checksum or an fp32 local index. Both cost more than the risk on an index that rebuilds from source in **40 seconds** | **Owner, 2026-09-07** |
| 6 | **Two S4 tests rebuild `index/export/` derivatives against the real index.** Same family as the tileplan defect | Materially different: deterministic, seconds, and it *regenerates* rather than truncates. The expensive `index/emb/` pass is untouched | Trivial; fix when S5b touches the exporter anyway | PM, low-cost |
| 7 | `accelerate` 1.14.0 installed but unused — no candidate needed sharded loading | Installed speculatively during S0 before that was known | Negligible; one unused package in the env. Recorded so a future reproducibility audit does not treat it as load-bearing | PM, low-cost |

Neither is an unmet spec criterion.

---

## Ship & operate

Single-user local system. `git revert` is the rollback. The release path is the
cold start documented in `CLAUDE.md`. The index is rebuildable from source
imagery, so it needs no backup. Observability is the build log: tiles/sec,
wall time, failure counts.
