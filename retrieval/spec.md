# Spec — Aerial Tile Retrieval

Source of truth. Code exists only to satisfy this file. Every requirement is
a checkable assertion with a concrete expected result.

**Owner sign-off: APPROVED 2026-09-07.** Design-review gate passed. Owner
directed that implementation begin in a fresh session (see `HANDOFF.md`).

Config symbols used below: `SCALES` (default `[448, 224, 112]` — a **list**,
not a scalar; see `notes.md#research-1`), `OVERLAP` (default 0.0),
`EMB_DTYPE` (export precision), `EXPORT_DIM` (PCA target, default 128),
`TOP_K`.

---

## 1. Domain requirements

**D-1 — Only source imagery is indexed.**
A raster is source imagery iff it has >= 3 bands AND its sampled pixel values
show > 64 distinct levels in each of bands 1-3. Derived rasters (binary masks,
class rasters, fused CD output) are excluded.
*Expected:* on `leb`, exactly 2 of 31 rasters qualify — `2022-10-29.tif` and
`2025-06-06.tif`. The classifier must return exactly that set, by value
inspection, never by filename pattern.

**D-2 — All reported ground distances are true ground metres.**
For any Mercator CRS, ground distance = projected distance x cos(latitude).
*Expected:* for `leb` (EPSG:3857, lat 33.10707, pixel 0.125002 m) the reported
GSD is 10.47 cm/px +/- 0.01 and a 448 px tile reports 46.91 m +/- 0.05. A
result of 12.5 cm/px or 56.0 m is a failure, not a rounding difference.

**D-3 — The data directory is read-only.**
No process writes, moves, renames or converts anything under
`/home/omer/PycharmProjects/Dynamic-Terrain/data`.
*Expected:* a test asserts every output path resolves inside
`retrieval/index/`; recursive mtime/checksum of the data tree is unchanged
across a full index build.

**D-4 — Tile identity is stable and reproducible.**
A tile's id is derived from `(aoi, source_file, date, col, row, TILE_PX)`.
*Expected:* two index builds over unchanged inputs produce identical tile ids
and identical tile counts.

**D-5 — Query capability is bounded by effective resolution.**
Effective optical resolution is ~35-45 cm (measured; the 10.47 cm grid is
upsampled ~3-4x). The system supports presence and coarse class; it does not
support fine attributes.
*Expected:* documented in the UI (see U-5). No spec item, test, or demo copy
claims attribute-level discrimination such as vehicle colour or model.

---

## 2. Functional requirements

**F-1 — Tiling covers the full raster extent, at every scale.**
For each `s` in `SCALES` the raster is tiled independently at `s` x `s` px.
*Expected:* for every scale, every pixel of the raster falls in >= 1 tile of
that scale; edge tiles are padded to full size, not dropped (dropping 376 rows
at 448 px loses 3.8% of scene height). Assert exact per-scale counts and zero
coverage gaps at each scale. Over the 8 indexed scenes at
`[448, 224, 112]`, `OVERLAP`=0: **5,198 + 20,704 + 82,640 = 108,542** tiles.

**F-0 — The embedder is chosen by measurement, not by assumption.** *(S0)*
Before the full index build, embed one calibration subregion with each
candidate and compare retrieval on the same query set: **PE-Core-G14-448**
(1280-d), **PE-Core-L14-336** (1024-d, ~6x faster), and **RemoteCLIP** —
which is pretrained on aerial/satellite imagery and may outperform both on
this domain. All are off-the-shelf and require no training.
*Expected:* a table of candidate x query with top-5 hits and a stated choice.
Changing embedder later invalidates the entire index, so this decision is made
first and recorded in the manifest.

> **SETTLED at S0, 2026-09-07 — `notes.md#s0-bakeoff`, evidence in
> `briefs/S0_result.md`.** Chosen: **RemoteCLIP-ViT-L-14, 768-d**
> (`chendelong/RemoteCLIP` @ `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`).
> It is the only candidate whose known-absent control is beaten by all 8 real
> queries (+0.0135 margin; PE-Core-L14 scored **-0.0157** — its
> `aircraft carrier` outranked its own `car`). 4/5 genuine vehicle crops at
> 112 px vs 3/5 and 2/5; 265 tiles/sec vs 84.5 and 11.6.
> **This is F-0's output, not an amendment** — F-0 exists to be decided by
> measurement, and it was. PM re-ran the suite (12/12) and re-read the crops
> independently.
>
> **Also settled by S0, and load-bearing for every later stage:** 448 px
> surfaces **zero** vehicles for **every** candidate (0/5, 0/5, 0/5), while
> 112 px takes top-1 for `car` in all three. Without the finest pyramid level
> this system cannot find a vehicle at all. `SCALES` is not a tuning
> preference; it is the reason retrieval works.

**F-1a — One embedder serves every scale.**
Different models occupy different embedding spaces, so a query vector cannot be
compared across them.
*Expected:* the index manifest records exactly one model id and revision, and a
build that would mix embedders fails loudly rather than producing a mixed
index. Every tile is resized to the model's native input, so a 112 px crop and
a 448 px crop cost the same forward pass.

**F-2 — Tile embeddings are unit-normalised, text-aligned pooled vectors.**
One pooled vector per tile per scale — **the model's own text-aligned image
embedding**, whatever produces it: the projected CLS token for RemoteCLIP /
OpenCLIP ViT-L-14, the attention-pool output for PE Core. Take it from the
model's public `encode_image` path; do not reach inside for an intermediate.
**Patch tokens are never used at any scale** — see `notes.md#research-1` and
`notes.md#arch-confirm`. The reason survives the embedder change: patch tokens
of a CLIP-family tower never entered the contrastive objective, so they are not
text-comparable, and the published attempt scored 2.5 nDCG@5 against 51.4 for
simply pooling. Small objects are handled by the `SCALES` pyramid instead — S0
measured that this works (112 px takes top-1 for `car`; 448 px finds none).
*Expected:* `||v||_2 = 1.0 +/- 1e-5` for every tile; dimensionality equals the
model's — **768 for the chosen RemoteCLIP-ViT-L-14** (1280 for
PE-Core-G14-448, 1024 for L14-336 if ever reverted to); the stored index records
model id, revision, and the scale each vector came from.

**F-2a — Export embeddings are PCA-reduced and quantised.**
The local index keeps full precision; the export reduces to `EXPORT_DIM` by PCA
fitted on the indexed corpus, then int8.
*Expected:* the PCA basis is stored with the export so a query vector can be
projected identically; retrieval@10 overlap between full-precision and exported
embeddings is measured and reported (this is the accuracy cost of the size
budget, and it must be a number, not an assumption).

**F-3 — Text queries embed through the same model's text tower.**
*Expected:* `||q||_2 = 1.0 +/- 1e-5`; identical query text yields a bitwise
identical vector within one process and within 1e-6 across processes.

**F-4 — Ranking is cosine similarity, returned with scores.**
Since vectors are unit-normalised, cosine = dot product.
*Expected:* top-`TOP_K` results are sorted by score descending; scores lie in
[-1, 1]; a query embedding compared against itself scores 1.0 +/- 1e-5.
Results are exact (brute force), not approximate.

> **CLARIFIED — owner signed off 2026-09-07** (`notes.md#s0-bakeoff`).
> **Ranking is per scale: one ranking per entry in `SCALES`, never a single
> pooled ranking across scales.** Each is labelled with its true ground extent
> (11 m / 22 m / 47 m).
> *Why:* S0 measured that 112 px takes top-1 for **9/9** queries on both PE
> candidates, so a pooled ranking is dominated by the finest scale even where
> that scale is the wrong one to look at — an 11.2 m crop physically cannot
> contain a road as a linear object, which is why `dirt road` and `sand`
> converge there.
> Rejected alternatives: within-scale score normalisation (the fusion weights
> would be a tuning knob with **no ground truth to tune against** — `Car` has
> zero labels), and a scale selector (U-6 already has filters to keep legible).
> *This is a clarification, not an amendment:* F-4 specified cosine, sorted,
> bounded and exact, and said **nothing** about how scales combine. The point
> was **unspecified, not weakened** — no criterion was loosened, and per
> methodology §8 the worker correctly stopped rather than inventing a fusion
> rule. `TOP_K` now means top-`TOP_K` *per scale*.
> *Expected, added:* a query returns exactly `len(SCALES)` rankings, each
> sorted descending and each tagged with its scale and true ground extent; no
> result appears in a ranking for a scale it was not embedded at.

**F-5 — The index persists and reloads without recomputation.**
*Expected:* build once, reload in a fresh process, and the same query returns
byte-identical top-`TOP_K` ids and scores. Reload of the full index completes
in < 5 s.

**F-6 — AOI filter restricts candidates by geographic bbox.**
*Expected:* given a lat/lon bbox, only tiles whose ground footprint intersects
it are candidates. A bbox covering one known tile returns that tile and no
other; an empty bbox returns no results and does not raise.

**F-7 — Date filter restricts candidates by acquisition date.**
*Expected:* on `leb`, filtering to `2025-06-06` halves the candidate pool to
that date's tiles only; the returned tile ids all carry that date.

> **AMENDMENT ACCEPTED — owner signed off 2026-09-07.**
> **Was:** date filter over all indexed imagery.
> **Now:** the date filter is implemented and tested against **`leb` only**,
> the sole true multi-date pair. The manifest mechanism is built generally so
> other AOIs can be dated later without code change — owner's words: "accept
> narrowing to leb, add infrastructure for future use". Scenes with no known
> date carry `date: unknown` and are **excluded** from date-filtered queries
> rather than silently included.
> *Why an amendment and not a send-back:* the criterion is unsatisfiable by any
> implementation — five of eight indexed footprints carry no date in the file,
> and one non-indexed mosaic spans an 8.6-year acquisition window. Evidence:
> `notes.md#recon-2`, `docs/DATA.md`.
> *Expected:* filtering `leb` to `2025-06-06` halves that AOI's candidate pool
> and every returned id carries that date; an undated AOI returns nothing under
> a date filter and does not raise.

**F-8 — The exporter writes a standalone HTML under the size cap.**
*Expected:* for a chosen AOI + date subset the output is a single file,
<= 16 MB, containing embedded thumbnails and embeddings, opening and querying
with no server and no network. Assert the byte size and assert zero external
`src`/`href` references.

**F-9 — Tile question answering sits behind a swappable interface.**
*Expected:* the interface accepts `(tile, question) -> str`. Two backends are
selectable by config; switching backends requires no change outside config.
The default backend is a commercially-licensable ungated model. Absence of
PerceptionLM weights must not break import, index build, or retrieval.

**F-10 — Every result carries its map location.**
*Expected:* each returned tile reports its lat/lon bbox and its source file,
date and pixel offsets. Round-tripping a tile id through the geotransform
reproduces its pixel offsets exactly.

---

## 3. Non-functional requirements

**N-1 — Query latency.** Top-10 over the full index in < 200 ms on CPU.
*Expected:* measured, pasted. Baseline: numpy does 5000 x 1280 in 0.37 ms, so
this has wide margin; the assertion guards against an accidental O(n) Python
loop or a sklearn wrapper (measured 68x slower).

**N-2 — Unattended full index build.** Records tiles/sec and total wall time;
completes without human intervention; resumable after interruption without
re-embedding completed tiles.

**N-3 — fp16 on GPU, never bf16.** `CUDA_VISIBLE_DEVICES=0`.
*Expected:* a test asserts the loaded model dtype is `torch.float16` and that
`torch.cuda.is_available()` is True — the latter guards trap 1 in `CLAUDE.md`,
where a shadowed CPU torch makes the build silently run on CPU.

**N-4 — Determinism.** Fixed seeds; same tile bytes produce the same embedding
to within 1e-5 across runs on the same device.

**N-5 — Nothing large is committed.** No weights, tiles, embeddings, imagery
or COGs in git.
*Expected:* `git status` clean after a full build; a test asserts every
artifact path is gitignored.

**N-6 — Export size ceiling is hard.** 16 MB. The exporter must fail loudly
with the computed size and the tile count that caused it, never silently
truncate results.

**N-7 — Cold-start reproducibility.** A fresh agent follows `CLAUDE.md` and
builds the index without asking a question. Verified by an actual fresh
session, not by inspection.

**N-8 — Portable across OS, GPU generation, and CPU-only.** *(added
2026-09-07 at owner request — see `notes.md#owner-portability`.)*
Nothing machine-specific may be embedded in `src/`. The system runs on Linux
and Windows, on any CUDA GPU generation, and with no GPU at all.
*Expected, each separately checkable:*
- **No absolute machine-specific path anywhere in `src/`.** A test greps the
  package for `/home/`, `C:\\`, `anaconda3` and the data-root literal and
  fails on any hit. The data root, index root and model cache come from
  **config with environment-variable override**, and the config carries no
  default that only exists on one machine.
- **dtype is selected by device capability, never hardcoded.** A pure function
  maps device -> dtype and is asserted directly: CUDA compute capability
  **< 8.0 -> `float16`** (Turing/Volta: bf16 is emulated and 5.6x slower);
  **>= 8.0 -> `bfloat16` permitted** (Ampere and later have native bf16);
  **CPU or MPS -> `float32`** (fp16 on CPU is unsupported or pathologically
  slow). The current machine is cc 7.5, so it must still resolve to
  `float16` — the existing behaviour is a *case* of the rule, not the rule.
- **Runs with no GPU.** Forcing device `cpu` completes an index build and a
  query, at documented reduced throughput. Asserted by an actual CPU-forced
  run over a small tile set, not by inspection.
- **Path handling is OS-agnostic.** `pathlib` throughout; no string
  concatenation of separators, no POSIX-only assumptions. Tile ids and manifest
  keys use a single canonical separator so an index built on one OS loads on
  the other. A test asserts a manifest written with Windows-style inputs reads
  back identically.
- **No interpreter path in code.** A specific interpreter may appear in
  documentation as an example; it may not appear in importable code.

**N-9 — `INSTRUCTIONS.md` cold-starts a foreign machine.** *(added
2026-09-07 at owner request.)*
A single top-level document that a person or a fresh agent can follow on a
machine that is **not** this one, to install, build an index, query it, and
produce the export.
*Expected:* it states prerequisites and how to check them; covers **Linux and
Windows** side by side; covers **CUDA, and CPU-only**; names the dtype rule and
why it differs per GPU generation; explains where imagery is expected and how
to point the system at a different location; gives copy-pasteable commands per
platform; lists expected wall times with the hardware they assume; and has a
troubleshooting section for the failure modes actually encountered (shadowed
CPU torch, bf16 on pre-Ampere, `pyproj` PROJ database, GDAL/rasterio install,
the Mercator GSD correction). **Verified by execution, not by review** — a
fresh session follows it with no prior context and no access to this
conversation, and reports where it got stuck. Anything it had to ask is a
defect in the document.

---

## 4. UX requirements

**U-1 — Opens in a working state.** Example queries visible and clickable at
rest; no blank box requiring the user to guess vocabulary.

**U-2 — Results are a scored, located grid.** Each hit shows its thumbnail,
similarity score, date, and lat/lon. Score formatting is fixed-width so a
column of scores is scannable. **Results are grouped into one row per scale,
each row labelled with its true ground extent** (per F-4's clarification) — so
the user can see which scale found a hit, and scores are only ever compared
within a row.

**U-3 — Empty state guides.** A query with no hits above threshold explains
that and suggests a broader term. Never a blank grid, never a stack trace.

> **CLARIFIED — owner signed off 2026-09-07** (`notes.md#s0-bakeoff`).
> **The threshold is a per-query relative gap, never an absolute cosine
> cut-off.** A weak match is one whose top score fails to stand out from that
> query's corpus mean by a stated multiple of the score spread; the multiple
> must be **visible in the UI**, not buried in code.
> *Why:* S0 measured that absolute cosine magnitude carries no cross-query
> meaning. PE-Core-L14's known-absent `aircraft carrier` scored **+0.2002,
> above its own best `car` at +0.1845**; PE-Core-G14's control margin is
> +0.0019; even the chosen RemoteCLIP's +0.0135 sits against a 0.0415 spread.
> Any fixed cut-off would either suppress real hits or pass pure noise.
> *This is a clarification, not an amendment:* U-3 required only that the empty
> state **exist and guide**, never that the threshold be absolute. Nothing was
> weakened. E-2's known-negative query supplies the calibration data.

**U-4 — Feedback on anything over 300 ms**, specifically the first query
(model/index warm-up) and any Q&A call.

**U-5 — The resolution limit is stated in the UI.** The effective ~35-45 cm
resolution and what it means — presence and coarse class, not fine attributes
— appears in the interface, not only in the README. This is an honesty
requirement: the demo scopes a product decision (`notes.md#intake`) and must
not overstate what the imagery supports.

**U-6 — Filter state is always legible.** Active AOI and date filters are
visible with the result count they produced, and are clearable in one action.

**U-7 — Responsive from 375 px.** Result grid reflows; no horizontal page
scroll.

**U-8 — A tile opens.** Clicking a result gives a larger view plus its
question box; the answer appears attached to that tile, not in a global log.

---

## 5. Evaluation — reported, not gating

The owner set the acceptance gate as qualitative (judged by eye,
`notes.md#intake`). These numbers are produced as **information** for the
product-scoping decision and do not block a stage.

**E-1 — recall@k against segmentation labels, honestly scoped.**
Source: `tile_cropped_x3308_y3674_z0.125.tif`, the 45-class label raster over
part of the `leb` AOI. **Measured limits, established before writing this —
do not re-derive:**

- It covers **993 x 500 m = 0.497 km2, only 23% of the `leb` footprint**
  (210 tiles at 448 px, 882 at 224, 3528 at 112).
- **49.8% of it is `Unclassified`**, leaving ~0.25 km2 of usable label.
- **`Car` has zero pixels.** 33 of 45 classes are present; `Car` is not one.
- It is **EPSG:4326** while `leb` imagery is EPSG:3857 — it must be warped onto
  the image grid before scoring. That warp is part of this stage.
- The three best-supported classes (`Batha` 13.9%, `DryGrassland` 12.8%,
  `Garigue` 6.5%) are Mediterranean shrubland succession stages — specialist
  ecology terms a web-trained text tower has almost no signal on. Report them
  if you like, but do not treat a low score there as a system failure.
- 33 of the 45 classes are lithology or pedology (`LimestoneRockyTerrain` vs
  `DolomiteRockyTerrain` vs `NariRockyTerrain`, `Rendzina`, `TerraRosa`,
  `ClayeySoil`, …). These are **not visual categories** — the distinction is
  made by field geology, not appearance. **Excluded from evaluation**, and the
  report must say why rather than showing them as failures.

**Evaluate these eight** — CLIP-nameable and actually supported. Report
recall@1/5/10 and precision@5 for each, **with its tile support beside it**:

| class | % of labelled area |
|---|---|
| `DirtRoad` | 1.74 |
| `Shadow` | 1.03 |
| `Pavement` | 0.53 |
| `House` | 0.52 |
| `DirtRoadB` | 0.35 |
| `Maquis` | 0.34 |
| `UnirrigatedOrchard` | 0.12 |
| `PavedRoad` | 0.11 |

A tile counts as a true hit if the class covers >= a stated minimum fraction of
it; state the fraction. Query with a natural phrasing, not the raw class token
("a dirt road", not `DirtRoadB`), and record the phrasing used.

**E-1a — vehicles have no ground truth.** `Car` is the most important query
class and the hardest, and the label raster contains none. Two options, in
order of preference:
1. Judge vehicle retrieval **by eye** — save the top-10 crops for "a white
   car" / "cars" / "vehicles" and read them back. Consistent with the owner's
   qualitative gate.
2. Optionally use the `leb` segmentation **output** rasters (which carry a car
   class) as **pseudo-labels**. Weaker evidence — they are model predictions,
   not truth — and any number derived from them must be labelled as such.

*Expected:* one table of the eight classes with support and phrasing, a
separate by-eye verdict on vehicles, and an explicit list of what was excluded
and why. **No pass/fail** — the gate is qualitative by owner choice.

**E-2 — Known-negative behaviour.** Query a term with no referent in the
imagery (e.g. "aircraft carrier"). Report the top-5 scores.
*Expected:* reported. A retrieval system that returns confident hits for
absent content is the failure mode this exposes.

---

## 6. Coverage matrix

| Req | Proof |
|---|---|
| F-0 | **SATISFIED at S0** — `briefs/S0_result.md`: candidate x query table, throughput, stated choice; PM re-ran 12/12 and re-read crops |
| F-1a | `test_one_embedder_per_index` — manifest carries exactly one model id + revision; a build that would mix embedders raises |
| F-2a | `test_pca_export_roundtrip` — basis stored and re-projectable; retrieval@10 overlap full-precision vs exported, **reported per scale** |
| E-1a | by-eye vehicle verdict on saved top-10 crops (no labels exist; non-gating) |
| F-11 | *optional (S8)* — same E-1 table for ColQwen2.5 side by side |
| D-1 | `test_source_classifier` — exactly 2 of 31 leb rasters, by value inspection |
| D-2 | `test_gsd_correction` — 10.47 cm/px, 46.91 m tile; rejects 12.5/56.0 |
| D-3 | `test_data_dir_readonly` — output paths + tree checksum unchanged |
| D-4 | `test_tile_id_stable` — two builds, identical ids |
| D-5 | U-5 copy assertion + spec review |
| F-1 | `test_tiling_covers_extent` — zero gaps, count 1012 |
| F-2 | `test_embed_unit_norm` — norms, dim, recorded model id |
| F-3 | `test_text_embed_deterministic` |
| F-4 | `test_cosine_ranking` — sorted, bounded, self-similarity 1.0; `test_ranking_is_per_scale` — one ranking per scale, no cross-scale pooling |
| F-5 | `test_index_roundtrip` — byte-identical results after reload |
| F-6 | `test_aoi_filter` — single-tile bbox, empty bbox |
| F-7 | `test_date_filter` — leb pool halves |
| F-8 | `test_export_size` — <= 16 MB, no external refs |
| F-9 | `test_qa_interface` — backend swap by config; import survives missing PLM |
| F-10 | `test_tile_geo_roundtrip` |
| N-1 | measured latency, pasted |
| N-2 | build log: tiles/sec, wall time, resume |
| N-3 | `test_dtype_and_cuda` |
| N-4 | `test_embedding_determinism` |
| N-5 | `test_artifacts_gitignored` |
| N-6 | `test_export_size_fails_loudly` |
| N-7 | fresh-session cold start |
| N-8 | `test_no_machine_paths` (grep of `src/`), `test_dtype_for_device` (cc<8 -> fp16, cc>=8 -> bf16, cpu -> fp32), `test_cpu_only_build`, `test_manifest_path_portable` |
| N-9 | fresh session executes `INSTRUCTIONS.md` on a foreign machine and reports blockers; anything it asked = defect |
| U-1..U-8 | adversarial UX review against the rendered export; U-3 additionally `test_weak_match_is_relative` — a known-negative query is flagged weak, and no absolute cosine constant appears in the threshold path |
| E-1, E-2 | eval report, non-gating |

**Value -> dependents** (re-verify these when the value changes):
`TILE_PX` -> F-1 tile counts, D-2 ground extent, F-8 export size, E-1
thresholds. `EMB_DTYPE` -> F-8 size, F-2/F-4 numeric tolerances.
Embedder choice -> F-2 dimensionality, F-8 size, all of section 5.

**Embedder changed at S0 (1280 -> 768). Dependents re-verified:** F-2
dimensionality updated to 768. F-8 export size — *unaffected*: the export is
PCA-reduced to `EXPORT_DIM`=128 before int8, so source dim never enters the
budget, and headroom improves. Section 5 — structurally unaffected, but every
number in it must be computed with RemoteCLIP, and S0's measured confusions
(beige corrugated metal retrieved by `sand`; no attribute discrimination) are
expected to appear there. F-4 numeric tolerances unchanged.

---

## 6a. Optional comparison — late interaction (off the shelf, no training)

**F-11 — ColQwen2.5 A/B on one AOI.** *Optional, after the pyramid works.*
Index one AOI with an off-the-shelf late-interaction visual retriever and
compare small-object recall against the pyramid on the same queries.
Use **ColQwen2.5** rather than ColPali: same recipe, built on Qwen2-VL
(Apache 2.0, ungated) instead of the gated PaliGemma.
*Expected:* the same E-1 table for both approaches, side by side.

Two constraints to respect and state in the report:
- **Local only.** 1,024 patch vectors per tile is ~256 KB/tile; one 10240²
  AOI is ~2.8 GB. It cannot be exported to a 16 MB static file, so this path
  never becomes the shipped demo — it is evidence about whether the pyramid is
  leaving small-object recall on the table.
- **Domain shift is real and unmeasured.** ColQwen2.5 was contrastively tuned
  for *document* retrieval (127,460 query-page pairs of PDFs, charts, figures).
  Zero-shot transfer to nadir aerial imagery is not established by any paper.
  Report the result honestly whichever way it falls.

Do **not** attempt MaxSim over PE-Core's own patch tokens. That is the
published failure mode — `notes.md#research-1`.

## 7. Deliberately excluded

- Change-aware retrieval across dates (offered, not chosen — `notes.md`).
- Approximate nearest neighbour / faiss. numpy exact top-k is faster at this
  scale; ANN would add a dependency and an accuracy caveat for no gain.
- Fine-tuning any model. Zero-shot only.
- Detection or segmentation output. This system retrieves and answers; it
  does not draw boxes or masks.
- Serving to users other than the owner. Single-user, local.

---

## 8. Corpus — measured

Source imagery only. Derived rasters (<= 6 distinct values per band) excluded
per D-1; the full derived inventory is in `notes.md#recon-2`.

### In the index — groups A + C, one coherent scale

| AOI | Scene(s) | W x H | CRS | true GSD | date | 448 tiles (valid) |
|---|---|---|---|---|---|---|
| leb | `2022-10-29.tif` | 20179x9784 | 3857 | 10.47 cm | 2022-10-29 | 945 |
| leb | `2025-06-06.tif` | 20179x9784 | 3857 | 10.47 cm | 2025-06-06 | 945 |
| AYOSH | `X693_Y3500.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 484 |
| AYOSH | `X693_Y3501.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 352 |
| gaza | `X625_Y3404.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 484 |
| gaza | `X625_Y3405.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 484 |
| gaza | `X625_Y3406.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 484 |
| (loose) | `X605_Y3388.tif` | 10240x10240 | 32636 | 10.00 cm | unknown | 397 |
| | | | | | **total** | **~4,146** |

A and C differ by 4.7% in true scale, so a tile means the same thing in both.
They sit in different CRSs; only the geo-lookup layer handles that.

### Excluded, with cause

| What | Cause |
|---|---|
| `sin/Sini_Oct_Det_2025.tif` | 415.7 cm/px. A 448 tile spans 1.86 km; nothing man-made resolvable. 86% of the nominal tile budget for zero query value. Shares EPSG:3857 with `leb` at 40x different scale. |
| Teheran (66) + iran (1), group B | 50 cm/px. A 448 tile spans 224 m — a city block. Vehicles not resolvable. Admissible later as a **separate coarse tier**, never in the same embedding space. |
| `/data/gaza.tiff` | Not aerial. No CRS, no geotransform; a ground-level press photograph. |
| `iran/Ax0*_y06.tif` | md5-identical to `Teheran/`'s. |
| ~230 derived rasters | D-1. |

**Do not fold groups into one index.** A single text query asked of tiles whose
ground footprints differ by three orders of magnitude is not one retrieval
problem.

---

## 9. Still open — none blocking

All gate questions are closed. Owner approved the spec, signed the F-7
amendment, approved the `sin` exclusion ("too heavy") and the E-1 upgrade.

Carried forward as **future work**, explicitly not in this build:
- **Remote-sensing CLIP survey never completed** (spend limit). RemoteCLIP is
  folded into F-0 as a candidate; GeoRSCLIP / RS5M / SkyScript remain
  unsurveyed and are drop-in swaps under F-1a if F-0 suggests the domain
  matters.
- **Open-vocabulary detection** (OWLv2, GroundingDINO, YOLO-World) returns
  boxes rather than ranked tiles. Owner specified tiles, so it is out of scope
  — but it is the natural follow-on if "find all tents in the frame" later
  means boxes.
- **Training-free dense-CLIP surgery** (MaskCLIP, SCLIP, GEM, ClearCLIP,
  TraceCLIP) and **SegEarth-OV** (CVPR 2025, remote-sensing OVSS). All
  training-free, so all compatible with the owner's no-training constraint —
  but evaluated on segmentation mIoU, never on ranked retrieval. A genuine
  research gap; not a demo dependency.
- **Change-aware retrieval** across `leb`'s two epochs.
- **Group B coarse tier** (Teheran/iran, 50 cm) as a separate index.
- **Problem-anchor ratification** — workaround cost and why-now remain open;
  `CLAUDE.md`'s problem statement stays marked pending. Non-blocking, since
  the gate is qualitative.
