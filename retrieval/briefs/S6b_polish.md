# Brief — polish pass: three small things that make it feel finished

**Dispatch:** subagent. **Proves:** nothing new — this is quality, not scope.
**Runs alongside S6**, on **disjoint files**. Coordinate by not touching
`embed_index.py`, `cog.py`, or anything under `index/emb/`.

## The three items

**1. The tile modal hides its metadata below a scrollbar.** Clicking a result
gives the enlarged crop, but score, scale, date and lat/lon sit under the fold.
U-2 and F-10 want them legible. Bring them above the fold, or make the modal lay
out so both the crop and its facts are visible without scrolling at 1200 px and
at 375 px (U-7).

**2. A `pyproj` warning greets the owner at startup:**
`UserWarning: pyproj unable to set PROJ database path`. It is **known and
harmless** — this project routes all CRS work through rasterio/GDAL precisely
because `pyproj.CRS` cannot open its database here, and uses `pyproj.Geod` only
for pure ellipsoidal geodesy, which needs no database. But it is the first thing
the owner sees and it reads like a fault.

Suppress **that specific warning** in the app's startup path — the same way
`open_clip`'s "initialized randomly" line is already suppressed. **Do not**
blanket-suppress warnings, and do not touch the library. Leave it visible in
test output, where it belongs.

**3. Two S4 tests regenerate `index/export/` derivatives against the real
index.** Same family as the tileplan defect that has now been fixed twice, and
materially milder — deterministic, seconds, regenerates rather than truncates —
but tests should not write to shipped artifacts at all. Point them at
`tmp_path`.

After your change, run the full suite and confirm **both**
`index/tileplan/*.json` (108,542 across three scales) and
`index/export/X605_Y3388/` are untouched — check file mtimes, not just that the
tests pass. **The artifact is the acceptance criterion, not the report.**

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data \
    /home/omer/anaconda3/envs/geo/bin/python
```

**Env vars do not persist between Bash calls — prefix every invocation inline.**

- Change **no behaviour**. All **135** existing tests must stay green, and that
  is how you demonstrate it.
- Data directory READ-ONLY. Do not rebuild any index.
- **Do not touch `src/embed_index.py`, `src/cog.py`, or `index/emb/`** — S6 is
  working there concurrently.
- N-8: no machine paths in `src/`, docstrings included.

## Report back

Status, RED-then-GREEN per item, full test run, the **post-suite mtime check**
on both artifacts, before/after screenshots of the modal at 1200 px and 375 px,
deviations, blockers.

---

# ADDENDUM — after S6 landed

**4. Record the measured memory footprint in `INSTRUCTIONS.md`.** S6 measured
**peak RSS 1,419.6 MiB** with the full 104,374-tile index loaded (vectors are
321 MB fp32; the rest is backgrounds plus rasterio/GDAL/numpy baseline). §1
currently says "16 GB is comfortable" without a measured figure. State the real
number so someone on an 8 GB machine can judge.

Also update §9: the app now serves **all 8 scenes across 4 AOIs**
(`X605_Y3388`, `leb`, `gaza`, `AYOSH`) with an AOI selector, and the date filter
is live for `leb` (2022-10-29 / 2025-06-06, splitting exactly 20,944 each).

**5. `app.py` is now free.** S6 has finished editing it, so the modal and
warning-suppression work in items 1 and 2 has no conflict. S5b is still running
but touches only new exporter files.

**Two numbers worth keeping consistent:** S6 reports N-1 end-to-end at mean
65.7 ms / max 104.5 ms, while the PM measured the raw dot product alone at
6.2 ms / 10.7 ms over the same 104,374 vectors. Both are inside the 200 ms
budget and they measure different things — do not "fix" the discrepancy, and if
you quote a latency anywhere user-facing, quote the **end-to-end** one.
