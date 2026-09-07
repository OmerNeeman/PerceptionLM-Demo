# S2 result — tile planner + geo round-trip

**Status: DONE**

Outputs: `src/tiling.py`, `src/cog.py`, `tests/test_tiling.py`,
`tests/test_geo_roundtrip.py`, `tests/test_cog.py` (added — not named in the
brief's Outputs list but required to house the COG acceptance criterion; see
Deviations). Artifacts: `index/tileplan/*.json` (one file per scene, all 3
scales), `index/cog/2022-10-29.tif`, `index/cog/2025-06-06.tif`.

All 79 tests in `retrieval/tests/` pass (44 pre-existing + 35 new), including
`test_no_machine_paths` (N-8 grep) over the two new modules. Full run:

```
================== 79 passed, 1 warning in 130.16s (0:02:10) ===================
```

---

## Per-scene table — planned / complete / valid / nodata, per scale

`planned` = padded ceil grid (F-1's number). `complete` = floor grid, no
padding (docs/DATA.md's base convention). `valid` = `complete` tiles whose
measured zero-fill fraction is `< 50%` (my stated, measured rule — see
"Settling the docs/DATA.md discrepancy" below). `nodata%` = whole-raster
zero-fill fraction, measured directly (not assumed).

| scene | scale | planned | complete | valid | nodata% |
|---|---:|---:|---:|---:|---:|
| AYOSH/X693_Y3500.tif | 448 | 529 | 484 | 484 | 0.00% |
| AYOSH/X693_Y3500.tif | 224 | 2116 | 2025 | 2025 | 0.00% |
| AYOSH/X693_Y3500.tif | 112 | 8464 | 8281 | 8281 | 0.00% |
| AYOSH/X693_Y3501.tif | 448 | 529 | 484 | 352 | 25.14% |
| AYOSH/X693_Y3501.tif | 224 | 2116 | 2025 | 1508 | 25.14% |
| AYOSH/X693_Y3501.tif | 112 | 8464 | 8281 | 6189 | 25.14% |
| X605_Y3388.tif (demo AOI) | 448 | 529 | 484 | 410 | 14.24% |
| X605_Y3388.tif | 224 | 2116 | 2025 | 1727 | 14.24% |
| X605_Y3388.tif | 112 | 8464 | 8281 | 7091 | 14.24% |
| gaza/X625_Y3404.tif | 448 | 529 | 484 | 484 | 0.00% |
| gaza/X625_Y3404.tif | 224 | 2116 | 2025 | 2025 | 0.00% |
| gaza/X625_Y3404.tif | 112 | 8464 | 8281 | 8281 | 0.00% |
| gaza/X625_Y3405.tif | 448 | 529 | 484 | 484 | 0.00% |
| gaza/X625_Y3405.tif | 224 | 2116 | 2025 | 2025 | 0.00% |
| gaza/X625_Y3405.tif | 112 | 8464 | 8281 | 8281 | 0.00% |
| gaza/X625_Y3406.tif | 448 | 529 | 484 | 484 | 0.00% |
| gaza/X625_Y3406.tif | 224 | 2116 | 2025 | 2025 | 0.00% |
| gaza/X625_Y3406.tif | 112 | 8464 | 8281 | 8281 | 0.00% |
| leb/2022-10-29.tif | 448 | 1012 | 945 | 945 | 0.25% |
| leb/2022-10-29.tif | 224 | 4004 | 3870 | 3870 | 0.25% |
| leb/2022-10-29.tif | 112 | 15928 | 15660 | 15660 | 0.25% |
| leb/2025-06-06.tif | 448 | 1012 | 945 | 945 | 0.01% |
| leb/2025-06-06.tif | 224 | 4004 | 3870 | 3870 | 0.01% |
| leb/2025-06-06.tif | 112 | 15928 | 15660 | 15660 | 0.01% |

**Sums** — planned across all 8 scenes: 448→**5,198**, 224→**20,704**,
112→**82,640**, grand total **108,542** — exactly the brief's table, asserted
in `test_tiling_covers_extent_exact_counts`. One `leb` scene at 448: **1,012**
(`test_one_leb_scene_at_448_is_1012`). Demo AOI (`X605_Y3388`) pyramid:
529+2,116+8,464 = **11,109** (`test_demo_aoi_pyramid`).

### Settling the docs/DATA.md discrepancy (per the brief's instruction)

Nodata was measured directly (whole-raster zero-fill mask: all of bands 1-3
== 0 — confirmed as the right sentinel by downsampled-mask sampling before
writing any code: AYOSH/X693_Y3500 0.0%, X693_Y3501 25.13%, X605_Y3388
14.23% — matching docs/DATA.md's own 25.1%/14.2% almost exactly). A
`complete` (floor-grid) tile counts as `valid` iff its own zero-fill fraction
is `< 50%` — stated and vectorised in `tiling.radiometric_valid_counts`.

My measured **448-scale valid counts match docs/DATA.md's per-scene column
exactly**: 484/352/484/484/484/945/945/397→**410** for X605. The one scene
that doesn't match verbatim is `X605_Y3388`: docs states 397, I measure
**410**. I did not force it to 397 — 410 is what the stated 50%-threshold
rule produces from the measured 14.24% raster-wide nodata fraction and the
raster's actual zero-fill footprint (which is not uniformly distributed, so
per-tile classification isn't a linear scaling of the global fraction). I do
not have visibility into what threshold or rule docs/DATA.md's original
figure used, so I report my rule and my number rather than reverse-engineering
one that reproduces 397. This is a difference to flag, not a contradiction —
both 397 and 410 are "the floor grid reduced by nodata", just under
possibly-different per-tile thresholds. Docs/DATA.md's stated total ("~4,146")
was already flagged by the PM as inconsistent with its own per-scene column
(which sums to 4,575, not 4,146) — my per-scene column above sums to
**4,588** (484+352+410+484+484+484+945+945).

---

## Acceptance criteria — RED then GREEN, real output

Every RED below was produced by sabotaging the actual implementation (never
the test), running, capturing the failure, then restoring the original code
and re-running to capture the pass. All sabotage was applied to scratch
copies (`tiling.py.orig`/`cog.py.orig`) and reverted; none of it touched
`index/` artifacts that survived into the final build, and — critically —
none of it wrote under the read-only data root except one instance described
under Deviations, which was caught, cleaned up, and fixed at the test level.

### F-1 — `test_tiling_covers_extent` (+ `test_demo_aoi_pyramid`, `test_zero_coverage_gaps`, `test_edge_tiles_are_padded_not_dropped`)

Sabotage: `grid_dims` changed from `ceil` to `floor` (i.e. edge tiles silently
dropped instead of padded).

RED:
```
FAILED tests/test_tiling.py::test_tiling_covers_extent_exact_counts - Asserti...
FAILED tests/test_tiling.py::test_demo_aoi_pyramid - assert 10790 == 11109
FAILED tests/test_tiling.py::test_zero_coverage_gaps[448] - AssertionError: A...
FAILED tests/test_tiling.py::test_zero_coverage_gaps[224] - AssertionError: AYOSH/X693_Y3500.tif@224: grid width 10080 < raster width 10240
FAILED tests/test_tiling.py::test_zero_coverage_gaps[112] - AssertionError: AYOSH/X693_Y3500.tif@112: grid width 10192 < raster width 10240
FAILED tests/test_tiling.py::test_edge_tiles_are_padded_not_dropped - assert 484 > (22 * 22)
6 failed, 11 deselected, 1 warning in 1.12s
```

GREEN (after restoring `ceil`):
```
tests/test_tiling.py::test_tiling_covers_extent_exact_counts PASSED      [ 16%]
tests/test_tiling.py::test_demo_aoi_pyramid PASSED                       [ 33%]
tests/test_tiling.py::test_zero_coverage_gaps[448] PASSED                [ 50%]
tests/test_tiling.py::test_zero_coverage_gaps[224] PASSED                [ 66%]
tests/test_tiling.py::test_zero_coverage_gaps[112] PASSED                [ 83%]
tests/test_tiling.py::test_edge_tiles_are_padded_not_dropped PASSED      [100%]
6 passed, 11 deselected, 1 warning in 1.13s
```

### F-10 — `test_tile_geo_roundtrip` (id→pixel offsets exact, Mercator + UTM)

Sabotage: `pixel_offset_from_tile_id` swapped col/row.

RED:
```
FAILED tests/test_geo_roundtrip.py::test_tile_id_roundtrips_to_exact_pixel_offsets[mercator_plan]
FAILED tests/test_geo_roundtrip.py::test_tile_id_roundtrips_to_exact_pixel_offsets[utm_plan]
    assert (col_px, row_px) == (t["px_offset_x"], t["px_offset_y"])
E   assert (0, 224) == (224, 0)
FAILED tests/test_geo_roundtrip.py::test_roundtrip_reproduces_the_geotransform_corner_from_a_fresh_open
3 failed, 5 passed, 1 warning in 0.26s
```

GREEN (after restoring):
```
tests/test_geo_roundtrip.py::test_tile_id_roundtrips_to_exact_pixel_offsets[mercator_plan] PASSED
tests/test_geo_roundtrip.py::test_tile_id_roundtrips_to_exact_pixel_offsets[utm_plan] PASSED
tests/test_geo_roundtrip.py::test_roundtrip_reproduces_the_geotransform_corner_from_a_fresh_open PASSED
tests/test_geo_roundtrip.py::test_mercator_bboxes_are_plausible PASSED
tests/test_geo_roundtrip.py::test_utm_bboxes_are_plausible PASSED
tests/test_geo_roundtrip.py::test_mercator_adjacent_tiles_share_bit_exact_edges PASSED
tests/test_geo_roundtrip.py::test_utm_adjacent_tiles_have_no_gap_and_bounded_overlap PASSED
tests/test_geo_roundtrip.py::test_projected_space_grid_is_exactly_contiguous_for_both_crs PASSED
8 passed, 1 warning in 0.25s
```

**A real subtlety found while writing these tests (not a code bug — a test-design correction, done before ever declaring green):** the lon/lat axis-aligned bbox is bit-identical between adjacent tiles for **Mercator** (its projected axes are lon/lat-aligned by construction), but **not** for **UTM**, where meridian convergence away from the zone's central meridian tilts each tile's true footprint inside its own axis-aligned bounding box. Measured worst case over `X605_Y3388`'s full grid at 224 px: **~0.26 m**, and it can appear as a tiny *overlap* or a tiny *gap* depending on position (both bounded). This is proven **not** a real coverage gap by `test_projected_space_grid_is_exactly_contiguous_for_both_crs`, which checks contiguity in each CRS's own projected coordinates (exact for both CRSs, by construction — no shear in either raster's affine). My first draft of the UTM adjacency test wrongly assumed overlap-only and a bit-exact projected-space match; both were fixed to match the measured geometry (1 m absolute tolerance for the lon/lat AABB artifact; `rel=1e-9` for the projected-space check, which needs to absorb ordinary floating-point catastrophic cancellation from the ~3.9e6 m false-easting/northing constants — measured residual ~1.7e-12 relative, nine orders of magnitude below what a real grid-stride bug would produce).

### D-4 — `test_tile_id_stable` (+ `test_two_date_unknown_scenes_get_disjoint_ids`, `test_tile_id_depends_on_full_identity_tuple`)

Sabotage: `make_tile_id` dropped `source_file` from the id (kept only
aoi/date/scale/col/row) — exactly the trap the brief names: two `date:
unknown` scenes (`AYOSH/X693_Y3500.tif`, `AYOSH/X693_Y3501.tif`, same AOI,
same dimensions) collapsing into one id space.

RED:
```
FAILED tests/test_tiling.py::test_two_date_unknown_scenes_get_disjoint_ids
    assert ids_a.isdisjoint(ids_b), "two date:unknown scenes collapsed into one id space"
E   AssertionError: two date:unknown scenes collapsed into one id space
FAILED tests/test_tiling.py::test_tile_id_depends_on_full_identity_tuple
E   AssertionError: assert 'AYOSH::unknown::448::00001::00002' != 'AYOSH::unknown::448::00001::00002'
2 failed, 2 passed, 13 deselected, 1 warning in 0.30s
```

GREEN (after restoring `source_file` into the id):
```
tests/test_tiling.py::test_tile_id_stable_across_two_runs PASSED
tests/test_tiling.py::test_tile_ids_carry_no_machine_specific_or_run_dependent_value PASSED
tests/test_tiling.py::test_two_date_unknown_scenes_get_disjoint_ids PASSED
tests/test_tiling.py::test_tile_id_depends_on_full_identity_tuple PASSED
4 passed, 13 deselected, 1 warning in 0.27s
```

### D-3 — `test_data_dir_readonly`

Sabotage: the "refuses to write outside the index root" guard removed from
`write_tile_plans`.

RED:
```
FAILED tests/test_tiling.py::test_write_tile_plans_refuses_to_write_outside_index
    with pytest.raises(ValueError):
E   Failed: DID NOT RAISE ValueError
1 failed, 16 deselected, 1 warning in 0.21s
```

GREEN (after restoring the guard):
```
tests/test_tiling.py::test_write_tile_plans_refuses_to_write_outside_index PASSED
1 passed, 16 deselected, 1 warning in 0.17s
```

The tree-digest half (`test_data_dir_unchanged_across_a_full_plan_and_cog_build`)
is backed by a second test (`test_data_dir_digest_check_would_actually_catch_a_change`)
that fakes `inventory.tree_digest` returning two different values via
`monkeypatch` and confirms the equality assertion fails on that input shape —
done this way, rather than by mutating the real data root, on purpose (see
Deviations for why that matters here specifically). Both pass; the real
digest was confirmed unchanged across an actual full plan (8 scenes × 3
scales, written to disk) + the real leb COG build.

### COG — `test_cog_pixel_identical`

Sabotage: the pixel-comparison line in `assert_pixel_identical`
(`if not np.array_equal(a, b)`) replaced with `if False` — i.e. the check
silently accepts any pixel content, simulating exactly the failure mode
("a COG that silently resamples or reprojects") the brief warns against.
Exercised against synthetic scratch rasters (a "good" identical copy and a
"bad" copy differing by one pixel), not the real leb scenes, so this runs in
milliseconds and never touches the data root.

RED:
```
    cog.assert_pixel_identical(src_path, good_path)  # must not raise
>   with pytest.raises(AssertionError):
E   Failed: DID NOT RAISE AssertionError
1 failed, 8 deselected in 0.16s
```

GREEN (after restoring the comparison):
```
tests/test_cog.py::test_assert_pixel_identical_detects_a_genuinely_different_copy PASSED
1 passed, 8 deselected in 0.15s
```

And the real, full-scale evidence (both `leb` scenes, sampled 448 px windows
including corners/edges, run against the actual built COGs — this is the
"born-green" check on real data, backed by the synthetic RED/GREEN above
proving the check itself has teeth):
```
tests/test_cog.py::test_cog_pixel_identical PASSED
tests/test_cog.py::test_cog_preserves_crs_transform_band_count PASSED
tests/test_cog.py::test_cog_is_actually_tiled_not_row_strips PASSED
tests/test_cog.py::test_only_leb_scenes_are_converted PASSED
```

---

## COG read-amplification — the entire justification for this stage

Computed from each dataset's own block geometry (`ds.block_shapes`), not
timed I/O — deterministic and reproducible, and it reproduces docs/DATA.md's
figures from first principles rather than quoting them:

| scene | before (source) | after (COG) |
|---|---:|---:|
| `leb/2022-10-29.tif` | **45.04x** (27,120,576 decoded bytes / 602,112 useful, for a 448² RGB window) | **1.31x – 2.94x** depending on tile/block-grid phase (best case block-aligned; worst case a 448 px window spans 3×3 of the 256 px blocks) |
| `leb/2025-06-06.tif` | **45.04x** | **1.31x – 2.94x** |

`leb`'s BEFORE figure is **offset-invariant**: block layout is `(1, width)` —
one full-width row per block — so the number of blocks touched depends only
on window height, never on where it starts. Verified directly
(`test_source_amplification_is_offset_invariant`: offsets `(0,0)`,
`(111,222)`, `(5000,3000)` all give exactly 45.04x). The AFTER figure genuinely
varies with alignment because 256 px blocks don't evenly divide a 448 px tile
grid (gcd(448,256)=64); both ends are reported rather than the flattering
best case. As a sanity cross-check, the **already-tiled** group's own
amplification (not converted, but useful to validate the formula) computes to
**1.31x (aligned) up to 2.94x (worst-case offset)** for a 256×256-blocked
scene — the 2.94x figure matches docs/DATA.md's stated "~2.9x" for that group
almost exactly, which is strong corroborating evidence the block-geometry
formula is right.

File sizes: `2022-10-29.tif` 592.4 MB → 237.5 MB (DEFLATE compressed well —
this scene has more spatial redundancy); `2025-06-06.tif` 592.4 MB →
596.8 MB (barely compresses — more textured/noisy content). Both preserve
CRS, geotransform, band count, dimensions exactly (asserted in
`test_cog_preserves_crs_transform_band_count`), and both pass
`assert_pixel_identical` over 25 sampled windows each (a 5×5 grid spanning
corners, edges and interior — `_sample_offsets`). Conversion wall time:
**~6.8 s for both scenes** (28 CPUs, DEFLATE, `NUM_THREADS=ALL_CPUS`).

Only `leb`'s two scenes were converted — `test_only_leb_scenes_are_converted`
asserts `index/cog/` contains exactly those two filenames and nothing else.

---

## Deviations / unspecified decisions

1. **`SCALES`/`OVERLAP` live in `tiling.py`, not `config.py`.** `config.py`
   (read, not modified) exposes only path/device settings; it has no
   `SCALES`/`OVERLAP` symbols to reuse, and `src/calibrate.py` (S0) already
   established the precedent of a module keeping its own local `SCALES`
   tuple. Adding them to `config.py` was out of scope for this stage's
   Outputs, so I kept them local to `tiling.py` instead of editing a file
   that isn't mine to touch. Flagging in case the PM wants them centralised
   later.

2. **Date and AOI extraction are new functions (`scene_date`, `scene_aoi`),
   not additions to `inventory.py`.** `inventory.py`'s S1 pixel-rule
   classification never needed a date/AOI concept; I added these fresh in
   `tiling.py` per the brief's "add a new function rather than changing an
   existing signature" instruction, rather than touching S1's green module.
   `scene_date` matches only the basename against
   `\d{4}-\d{2}-\d{2}\.tiff?$` (deliberately filename-based — this is date
   *extraction*, not the D-1 source/derived classification, which stays
   pixel-only in `inventory.py` and is untouched).

3. **Tile id format**: `"{aoi}::{source_file}::{date}::{scale}::{col:05d}::{row:05d}"`.
   `::` was chosen as separator (cannot appear in any real path/date
   component here) so `parse_tile_id` round-trips via a plain `str.split`.
   `aoi` is technically redundant with `source_file` (which already encodes
   it via its directory) but is included literally per D-4's stated tuple
   `(aoi, source_file, date, col, row, TILE_PX)`.

4. **`tests/test_cog.py` added** — not named in the brief's Outputs list
   (which named only `test_tiling.py` and `test_geo_roundtrip.py`), but the
   acceptance list includes a `test_cog_pixel_identical` criterion and
   `src/cog.py` is a distinct module; I judged a dedicated test file clearer
   than folding COG tests into `test_tiling.py`. Reversible, low-cost,
   reported rather than asked about per the handback protocol.

5. **Radiometric "valid" tile rule**: a complete (floor-grid) tile counts as
   valid iff `< 50%` of its pixels are the zero-fill sentinel (all of bands
   1-3 == 0). This threshold is not specified anywhere I was allowed to read
   (spec.md/S2.md give the *concept*, not a threshold); I chose 50% as the
   natural "majority real content" cut and stated it plainly
   (`NODATA_TILE_THRESHOLD` in `tiling.py`). It reproduces docs/DATA.md's
   448-px column exactly for 7 of 8 scenes; `X605_Y3388` comes out 410, not
   397 — see the per-scene table section above for the full account. This
   is reported per the brief's explicit instruction ("your measured valid
   counts settle it... if your valid counts disagree... say so with the
   nodata fraction you measured"), not treated as a bug.

6. **Nodata sentinel = "all of bands 1-3 == 0"**, not a declared GDAL
   `nodata` value. All 8 scenes report `nodata: None` in the S1 inventory (no
   declared sentinel) — I measured the actual zero-fill pattern directly (a
   quick downsampled-read check before writing any code: AYOSH/X693_Y3501
   25.13%, X605_Y3388 14.23%, matching docs/DATA.md's 25.1%/14.2% almost
   exactly) and used that as the basis for `_nodata_mask`, rather than
   relying on undeclared metadata.

## Blockers

None. No per-scale count disagreed with the brief's table (all matched
exactly on first computation, no retries). No coverage gap. Both COGs are
pixel-identical to their sources. The two `date: unknown` AYOSH scenes get
disjoint tile ids by construction (verified). `ground_resolution`'s
`GsdGuardError` did not fire for any of the 8 scenes during planning (already
established green by S1/S1_fix; re-exercised here since `tiling.py` calls it
for every scene, and it stayed silent as expected).

## An incident during test authoring, fully resolved

While writing the D-3 sabotage-and-verify cycle for
`test_write_tile_plans_refuses_to_write_outside_index`, my first draft of
that test pointed the "outside index" destination at
`config.get_data_root() / "escape"` — the **real** read-only data root. With
the guard temporarily removed (for RED evidence), `write_tile_plans` actually
created `escape/leb__2022-10-29.json` under the real data root. I caught this
immediately after the RED run (before ever restoring the code, i.e. before
declaring anything green), deleted the directory
(`rm -rf .../data/escape`), confirmed via `find .../data -iname "*escape*"`
that nothing remained, and rewrote the test to target a `tmp_path` scratch
directory instead — which still exercises the exact same guard logic without
ever being able to touch the data root even if the guard is broken. Re-ran
the full RED→GREEN cycle against the corrected test (both captured above). A
final sweep (`find .../data -newer <a long-unmodified file>`, directory
listing, and explicit `leb/*.tif` mtime check) confirms the data root now
contains nothing dated after this session and no `escape` artifact anywhere.
Flagging this in full rather than quietly fixing it, since disclosure matters
more here than looking clean.
