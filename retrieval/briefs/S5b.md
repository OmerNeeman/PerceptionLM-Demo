# Brief — S5b: the shareable standalone HTML export

**Dispatch:** subagent. **Proves:** **F-8**, **N-6**, **D-5**, **U-1..U-7** in
the exported artifact. **Depends on:** S4 (export basis, green).
**Runs alongside S6** — it is indexing scenes and editing `app.py`. **Touch
neither.** Your work is new files.

## Goal

One HTML file the owner can send someone. It opens with **no server and no
network**, and answers queries from a **precomputed query set**.

## Read F-8's amendment first

Free-text querying lives **only** in the local app. Measured reason: RemoteCLIP's
text tower is **123.7 M params — 247 MB fp16, 124 MB int8** against a **16 MB**
cap. **15.5x the entire budget.** No compression closes that; do not try.

So the export ships **~200 precomputed query vectors** spanning the four
vocabularies the owner named — small objects, structures, terrain, damage —
clickable, with type-to-filter **over that set**. A typed phrase that is not in
the set must say so plainly and suggest the nearest ones. **Never fake free
text**, and never render a box that looks like it accepts anything.

**No question box.** A VLM cannot run here either. Do not render a disabled one.

## The budget — and the design that makes it fit

The naive approach is one thumbnail per tile: **9,631 thumbnails** for this AOI.
That does not fit sensibly.

**Ship one downsampled basemap of the scene and crop client-side.** The scene is
10240x10240; a JPEG basemap at ~4096 px is a few MB, and every tile is a crop of
it at known pixel offsets (tile ids carry **grid indices**: pixel = index x
scale). At 0.4 scale a 112 px tile renders ~45 px — enough for a result grid,
and it can be drawn into a `<canvas>` from one image. This is what
`notes.md#research-1` costed originally: "128-d int8 + 1.5 MB basemap".

Indicative budget — **measure, do not trust these**:

| component | approx |
|---|---|
| 384-d int8 vectors, 9,631 tiles | 3.70 MB |
| ~200 query vectors | 0.08 MB |
| basemap JPEG | measure; aim 3-6 MB |
| tile metadata (id, scale, offsets, lat/lon) | measure; consider a packed array, not per-tile JSON objects |
| **hard cap** | **16 MB** |

If it does not fit, reduce **basemap resolution** first — it degrades gracefully;
dropping tiles or precision does not.

## N-6 — the cap is hard

The exporter **fails loudly** with the computed size and the tile count that
caused it. It must **never silently truncate results** to fit.

## What the page must do

- **U-1:** example queries visible and clickable at rest.
- **U-2:** results are a scored, located grid, **one row per scale**, each row
  labelled with its true ground extent, scores fixed-width, **compared only
  within a row**.
- **U-3:** the confidence band, from the stored background set, **shown not
  gating**, worded **"may not be present"**.
- **U-5:** the resolution caveat, verbatim in spirit with the local app —
  ~35-45 cm, presence and coarse class, **no attribute search**. Non-negotiable:
  this file may be seen by people who never speak to the owner.
- **U-6:** filter state legible with its result count, clearable in one action.
- **U-7:** responsive from 375 px, no horizontal page scroll.
- **U-8 (export form):** a tile opens larger with score, scale, date, lat/lon.
  No question box.

## Acceptance criteria — test-first, paste RED then GREEN

- `test_export_is_single_file_under_cap` (F-8, N-6) — assert **byte size <= 16 MB**
- `test_export_has_zero_external_refs` (F-8) — no `src`/`href` to any external
  origin; no CDN, no font, no analytics. Assert by parsing, not by eyeballing.
- `test_export_size_fails_loudly` (N-6) — force an oversize build; it raises
  naming the computed size and tile count, and writes nothing truncated.
- `test_export_query_matches_local` — for 5 queries in the precomputed set, the
  export's top-10 per scale matches the local app's **exported-basis** ranking.
  This is the one that proves the artifact is not subtly wrong.
- `test_resolution_caveat_in_export` (U-5, D-5).
- All existing tests stay green.

Then **open the file and use it**: click an example, type something in the set,
type something **not** in the set, open a tile, resize to 375 px, and — the
point of the whole stage — **load it with the network disabled**. Report what
happened.

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python
```

**Env vars do not persist between Bash calls — prefix every invocation inline.**

- **Do not edit `src/app.py`, `src/embed_index.py`, `src/cog.py`, or anything
  under `index/emb/`** — S6 is working there.
- Data directory READ-ONLY. **Tests must not write under the real index root**
  (fixed twice already) — use `tmp_path`; write the export to `index/export/`.
- Build against `X605_Y3388`, but take the AOI as a **parameter** — S6 is adding
  seven more scenes and the owner will want `leb`'s destruction pair exported
  next.
- dtype via `device.py`, paths via `config.py`, ids via `posix_key()`.
- Never commit the export, thumbnails or basemap.

## Report back

Status, QA checklist with pasted RED-then-GREEN, full test run, **the component
size table with real measured bytes and the final file size**, what happened for
each manual case (especially **network disabled**), a screenshot of the rendered
page, deviations, blockers.

**Flag as a blocker rather than working around it if:** the file cannot fit
16 MB at a basemap resolution that still makes the grid legible, or the export's
rankings disagree with the local app's exported-basis rankings.
