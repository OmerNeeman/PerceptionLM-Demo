# Brief — S5c: make the export usable, and make `leb` fit

**Dispatch:** subagent. Two defects, both owner-reported.

## Defect 1 — the export opens as a wall of text and shows no imagery

Screenshot the current `index/export/X605_Y3388/export.html` and look at it. On
load it renders **226 query chips in four long lists and nothing else** — no
tiles, no basemap, no imagery of any kind, then a large empty page. Someone
opening it **cannot tell it is an image-search tool.**

It satisfies U-1's letter (examples visible and clickable) and fails its intent
("opens in a working state; no blank box requiring the user to guess
vocabulary"). The owner's words: *"looks bad"*.

**What it must do instead:**

1. **Run a default query on load so results are visible immediately.** The
   aerial tiles are the product — they must be on screen before any click.
   Pick a query with strong hits for the AOI (`a car` or `tents` for
   `X605_Y3388`; something from the damage set for `leb`) and label it clearly
   as an example, not a search the user made.
2. **Show ~8-12 curated chips, not 226.** Two or three per category, chosen to
   demonstrate range. The existing filter box already reaches the rest — say so
   next to it ("226 queries available — start typing"). Optionally a "show all"
   toggle, collapsed by default.
3. **Make the results grid the hero.** Thumbnails large enough to read; the
   112 px row is where small objects live and it is currently the least legible.
4. Keep everything that is already right: the resolution caveat, one row per
   scale labelled by ground extent, scores compared only within a row, the
   confidence band worded "may not be present", the tile modal.

This is a **layout and default-state change**, not a redesign. Do not touch
retrieval maths, the payload format, or the caveat wording.

## Defect 2 — `leb` cannot be exported at all

```
ExportSizeError: leb: export does not fit under the 16777216 byte cap even at
the smallest tried basemap resolution (512 px): computed size = 22907790 bytes
(22.91 MB) for 41888 tiles. Nothing was written.
```

**The failure is correct behaviour** — N-6 working as designed, refusing to
truncate and naming both numbers. Do not weaken it.

The cause is arithmetic: `leb` is two dates over one footprint, **41,888 tiles**
against `X605_Y3388`'s 9,631. At `EXPORT_DIM` 384, int8, the **vectors alone are
16.1 MB** — over the cap before any basemap. Reducing basemap resolution cannot
save it, which is why the existing fallback exhausted itself.

**Add `EXPORT_DIM` to the fallback ladder.** When basemap reduction is not
enough, step the dimension down (384 → 256 → 192 → 128) and report which was
used. The PM measured the fidelity cost on the demo AOI — retrieval@10 overlap
against full precision:

| dim | int8 MB per 41,888 tiles | measured overlap (on X605) |
|---|---|---|
| 384 | 16.1 | 0.833 |
| 256 | 10.7 | 0.800 |
| 192 | 8.0 | 0.754 |
| 128 | 5.4 | 0.713 |

**Keep both dates in one file.** `leb`'s whole value is the 2022-10-29 →
2025-06-06 destruction pair — intact village core against the same pixels as
rubble — and splitting it into two files destroys the comparison that makes it
worth exporting. It needs **two basemaps**, one per date, and the date control
must switch between them.

**Report the dimension actually used and re-measure the overlap for `leb` at
that dimension.** Do not quote the table above — it was measured on a different
AOI. The exported file must state its own fidelity somewhere a reader can find.

## Acceptance criteria — test-first, paste RED then GREEN

- `test_export_opens_with_results` — the built HTML contains rendered result
  markup for the default query, not only chips. **RED is available now.**
- `test_export_chip_count_is_curated` — the landing state shows a bounded
  number of chips (assert <= 20), with the full set still reachable.
- `test_export_dim_fallback` — an AOI that cannot fit at 384 steps down and
  records the dimension used; one that fits is unaffected.
- `test_leb_export_builds_under_cap` — `leb` produces a file <= 16 MB
  containing **both** dates.
- `test_export_size_fails_loudly` — still raises when even the smallest
  configuration will not fit. **Do not weaken N-6.**
- Existing export tests and the full suite stay green.

Then **screenshot both exports at 1280 px and 375 px and look at them.** Chrome
is available:
`google-chrome --headless=new --disable-gpu --no-sandbox --window-size=1280,2000 --virtual-time-budget=15000 --screenshot=OUT.png "file:///ABS/PATH.html"`
Judge them as a stranger would. Say what you see.

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python
```

**Env vars do not persist between Bash calls — prefix every invocation inline.**

- **Zero external references** stays absolute: no CDN, no font, no analytics.
  The file must work with the network off.
- Data directory READ-ONLY. Do not rebuild `index/emb/`. Tests use `tmp_path`.
- Rebuild **both** exports when done: `X605_Y3388` and `leb`.
- Ranking parity must hold — the PM verified 1.000 for `X605_Y3388` and will
  re-check both.

## Report back

Status, RED-then-GREEN, full test run, **both file sizes**, the dimension and
measured overlap for `leb`, screenshots described, deviations, blockers.

---

# ADDENDUM — the fallback ladder optimises the wrong thing

`leb` now builds, but it is **visually unusable at the fine scales** and the
cause is a bad budget allocation, not a bug.

Measured in the shipped files:

| | X605_Y3388 | leb |
|---|---|---|
| basemap | 4096x4096, **3.95 MB** | 1280x621, **0.28 MB** x2 |
| basemap scale | 0.400 | **0.063** |
| a 112 px tile renders as | **45 px** | **7 px** |
| vectors | 3.70 MB | **10.7 MB** (256-d) |
| total | 10.13 MB | 15.62 MB |

**leb spends 10.7 MB on vectors and 0.59 MB on imagery.** The 112 px crops are
7 real pixels upscaled to a 150 px card — mush. A viewer cannot tell whether a
result is rubble, which is the entire question that AOI exists to answer.

**This is the PM's fault, not yours.** The original brief said "if it does not
fit, reduce basemap resolution first — it degrades gracefully". That was correct
for `X605_Y3388`, where basemap reduction alone sufficed, and wrong for `leb`,
where it starved the imagery to protect a fidelity number.

**Corrected principle: legibility has a floor; fidelity does not.**

The export exists to be *looked at*. A retrieval@10 overlap of 0.77 against 0.71
is invisible to a human; a 7 px thumbnail against a 24 px one decides whether
the file is usable at all.

## What to change

1. **Enforce a minimum basemap scale** such that a tile at the *finest* scale
   renders at **>= 24 px** before display upscaling. For `leb` (20179 px wide,
   finest tile 112 px) that means scale >= ~0.214, i.e. a basemap >= ~4,300 px
   wide.
2. **Reorder the ladder:** hold the basemap at or above that floor and step
   `EXPORT_DIM` down **first** (384 → 256 → 192 → 128) to fund it. Only if the
   smallest dimension still will not fit may the basemap go below the floor —
   and then say so explicitly in the file's footer.
3. `X605_Y3388` must be **unaffected** — it already sits at 0.400, far above the
   floor, and fits at 384-d. Verify it is byte-comparable in structure and that
   its overlap is still ~0.85.
4. Re-measure and report `leb`'s dimension and overlap at the new allocation.
   Indicative: 192-d costs ~8.0 MB of vectors and frees ~7 MB for two basemaps,
   which should clear the floor comfortably.

## Acceptance

- `test_basemap_scale_floor` — the finest-scale tile renders at >= 24 px in
  every built export, or the footer states that the floor was breached and why.
- `leb`'s 112 px crops are **visibly legible**. Screenshot at 1280 px and look:
  if you cannot tell rubble from intact roofs, it has not passed.
- N-6 still raises when nothing fits. Do not weaken it.
- Both exports rebuilt, both <= 16 MB, zero external refs, parity intact.
