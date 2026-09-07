"""D-1 / D-3 / D-4: source-vs-derived classification and the raster inventory.

D-1 is the signature correctness risk of this project. Only a handful of the
~266 rasters under the data root are real imagery; the rest are binary masks,
class rasters and fused change-detection output. **Filenames do not
distinguish them** -- `leb/cache/2022-10-29_ST_output.tiff` and
`leb/2022-10-29.tif` share a prefix and are not the same kind of thing at all.
So the decision is made on pixel value distributions, and the symlink
assertions below exist to prove that a filename fast-path cannot creep back in.

Run:  env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python -m pytest
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import inventory

DATA_ROOT = inventory.DATA_ROOT
LEB = "leb"

# The two scenes docs/DATA.md records as `leb`'s source imagery.
LEB_SOURCE_PAIR = {"leb/2022-10-29.tif", "leb/2025-06-06.tif"}

# A byte-identical 1024x1024 crop of leb/2022-10-29.tif that also lives under
# leb/. It passes the D-1 pixel rule (it *is* source pixels) but is not an
# indexable scene -- see test_d1_pixel_rule_over_leb_also_finds_a_contained_crop.
LEB_CONTAINED_CROP = "leb/leb_crop_x4096_y3072_1024.tif"

# A 357x233 4-band colour legend. Real pixels, many levels, no georeference.
LEB_LEGEND_PNG = "leb/tmp_results/cls_leb_legend.png"

# docs/DATA.md, "The index: 8 source scenes". path -> (W, H, bands, CRS, true GSD cm)
MANIFEST = {
    "leb/2022-10-29.tif": (20179, 9784, 3, "EPSG:3857", 10.47),
    "leb/2025-06-06.tif": (20179, 9784, 3, "EPSG:3857", 10.47),
    "AYOSH/X693_Y3500.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
    "AYOSH/X693_Y3501.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
    "gaza/X625_Y3404.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
    "gaza/X625_Y3405.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
    "gaza/X625_Y3406.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
    "X605_Y3388.tif": (10240, 10240, 3, "EPSG:32636", 10.00),
}


@pytest.fixture(scope="module")
def inv():
    return inventory.build_inventory()


def _rec(inv, path):
    for r in inv["rasters"]:
        if r["path"] == path:
            return r
    raise AssertionError(f"{path} not in inventory")


# --------------------------------------------------------------------------
# D-1
# --------------------------------------------------------------------------


def test_source_classifier(tmp_path, inv):
    """Exactly the two expected `leb` rasters are indexable source scenes, and
    the decision survives being addressed through deliberately misleading
    filenames."""
    assert set(inventory.indexable_paths(inv, subdir=LEB)) == LEB_SOURCE_PAIR

    # --- the same decision, made through renamed symlinks -----------------
    # Symlinks live under retrieval/index/, NEVER in the data dir (D-3).
    probe = inventory.INDEX_DIR / "symlink_probe"
    probe.mkdir(parents=True, exist_ok=True)

    source_under_derived_names = {
        "cls_binary_ST48_no_softmax.tif": "leb/2022-10-29.tif",
        "2019-01-01_output_Roads_fused_mask.tiff": "leb/2025-06-06.tif",
    }
    derived_under_source_names = {
        "2022-10-29.tif": "leb/cache/2022-10-29_ST_output.tiff",
        "2025-06-06.tif": "leb/cache/2025-06-06_debris_48_new_0706_binary.tiff",
    }

    for name, target in {**source_under_derived_names, **derived_under_source_names}.items():
        link = probe / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(DATA_ROOT / target)

    for name in source_under_derived_names:
        assert inventory.is_source_imagery(probe / name) is True, (
            f"{name} is real imagery reached through a derived-looking name; "
            "the classifier is reading the filename, not the pixels"
        )
    for name in derived_under_source_names:
        assert inventory.is_source_imagery(probe / name) is False, (
            f"{name} is a derived mask reached through a source-looking name; "
            "the classifier is reading the filename, not the pixels"
        )


def test_d1_pixel_rule_over_leb_finds_more_than_the_two_scenes(inv):
    """FINDING (S1): the D-1 pixel rule, applied on its own, returns FOUR
    rasters under leb/ -- not the two that spec.md D-1 and docs/DATA.md both
    record as `exactly 2 of 31`.

    Both extras are correctly source-by-pixels and correctly NOT indexable
    scenes, for two different reasons:

      leb_crop_x4096_y3072_1024.tif   3-band uint8, EPSG:3857, 236/246/255
                                      distinct -- a byte-identical 1024x1024
                                      window of leb/2022-10-29.tif at pixel
                                      offset (4096, 3072). Real imagery,
                                      duplicated ground.
      tmp_results/cls_leb_legend.png  357x233 4-band colour legend. Real
                                      pixels, many levels, no georeference.

    So D-1's *rule* is sound and the indexable set is still exactly the
    expected pair -- but the pixel rule alone is not a sufficient filter, and
    the "2 of 31" in the two documents is not what the rule returns. Pinned
    here so the PM's ruling is recorded rather than absorbed.
    """
    by_pixels = set(inventory.source_imagery_paths(inv, subdir=LEB))
    assert by_pixels == LEB_SOURCE_PAIR | {LEB_CONTAINED_CROP, LEB_LEGEND_PNG}

    crop = _rec(inv, LEB_CONTAINED_CROP)
    assert crop["is_source_imagery"] is True
    assert crop["is_indexable"] is False
    assert crop["reason"] == "duplicate_of"
    assert crop["duplicate_of"] == "leb/2022-10-29.tif"
    assert crop["duplicate_kind"] == "contained_crop"
    assert crop["crs"] == "EPSG:3857"
    assert min(crop["distinct_per_band"][:3]) > 64

    legend = _rec(inv, LEB_LEGEND_PNG)
    assert legend["is_source_imagery"] is True
    assert legend["is_indexable"] is False
    assert legend["reason"] == "no_georeference"
    assert legend["driver"] == "PNG"


def test_leb_denominator_is_reported_as_a_fact(inv):
    """spec.md D-1 and docs/DATA.md say `2 of 31`; notes.md#intake says `2 of
    leb's 50 TIFFs`. This pins what is actually on disk so the PM can correct
    whichever document is wrong.

        65 files under leb/
        31 of them .tif/.tiff        <- the 31 is right, the 50 is not
        32 readable rasters          (the 31 TIFFs + a 357x233 legend PNG)
         4 pass the D-1 pixel rule   (3 of them TIFFs -- see the test above)
         2 indexable source scenes
    """
    leb_rasters = [r for r in inv["rasters"] if r["path"].startswith("leb/")]
    leb_non = [p for p in inv["non_rasters"] if p.startswith("leb/")]
    leb_files = len(leb_rasters) + len(leb_non)
    leb_tiffs = [
        r["path"] for r in leb_rasters if r["path"].lower().endswith((".tif", ".tiff"))
    ]

    assert leb_files == 65, f"leb/ holds {leb_files} files"
    assert len(leb_tiffs) == 31, f"leb/ holds {len(leb_tiffs)} TIFFs, not 31 and not 50"
    assert len(leb_rasters) == 32, f"leb/ holds {len(leb_rasters)} readable rasters"
    assert inv["counts"]["leb_rasters"] == 32
    assert len(inventory.source_imagery_paths(inv, subdir=LEB)) == 4
    tiff_sources = [
        p for p in inventory.source_imagery_paths(inv, subdir=LEB)
        if p.lower().endswith((".tif", ".tiff"))
    ]
    assert len(tiff_sources) == 3
    assert len(inventory.indexable_paths(inv, subdir=LEB)) == 2


def test_classification_is_by_band_count_and_distinct_levels_only():
    """The pure rule, no I/O: >= 3 bands AND > 64 distinct levels in bands 1-3."""
    assert inventory.passes_pixel_rule(3, [238, 245, 248]) is True
    assert inventory.passes_pixel_rule(4, [256, 241, 215]) is True
    assert inventory.passes_pixel_rule(3, [2, 2, 2]) is False  # binary mask
    assert inventory.passes_pixel_rule(3, [6, 6, 6]) is False  # class raster
    assert inventory.passes_pixel_rule(1, [238]) is False  # too few bands
    assert inventory.passes_pixel_rule(3, [238, 245, 3]) is False  # one flat band
    assert inventory.passes_pixel_rule(3, [65, 65, 65]) is True  # just over
    assert inventory.passes_pixel_rule(3, [64, 65, 65]) is False  # exactly 64 fails


# --------------------------------------------------------------------------
# D-1 (breadth)
# --------------------------------------------------------------------------


def test_source_inventory_matches_manifest(inv):
    """The 8 indexed source scenes of docs/DATA.md, with its dimensions, band
    counts and CRS -- and the excluded-with-cause entries rejected for their
    own, distinguishable reasons."""
    assert set(inventory.indexable_paths(inv)) == set(MANIFEST)

    for path, (w, h, bands, crs, gsd_cm) in MANIFEST.items():
        r = _rec(inv, path)
        assert (r["width"], r["height"]) == (w, h), f"{path} dimensions"
        assert r["band_count"] == bands, f"{path} band count"
        assert r["crs"] == crs, f"{path} CRS"
        assert r["true_gsd_cm"] == pytest.approx(gsd_cm, abs=0.01), f"{path} true GSD"
        assert r["is_source_imagery"] is True
        assert r["is_indexable"] is True
        assert r["reason"] is None
        assert min(r["distinct_per_band"][:3]) > 64

    # gaza.tiff: real 3-band imagery, but no CRS and no geotransform. It fails
    # as an indexable scene for a DIFFERENT reason than a binary mask does.
    loose = _rec(inv, "gaza.tiff")
    assert loose["is_source_imagery"] is True
    assert loose["is_indexable"] is False
    assert loose["reason"] == "no_georeference"
    assert loose["crs"] is None
    assert loose["crs_kind"] == "none"
    assert loose["true_gsd_cm"] is None

    # sin: real imagery, georeferenced, but 415.7 cm true GSD -- a different
    # resolution regime, not a different kind of raster.
    sin = _rec(inv, "sin/Sini_Oct_Det_2025.tif")
    assert sin["is_source_imagery"] is True
    assert sin["reason"] == "resolution_regime_excluded"
    assert sin["true_gsd_cm"] == pytest.approx(415.7, abs=1.0)

    # Teheran / iran: real imagery at 50 cm -- again resolution regime.
    teh = _rec(inv, "Teheran/Ax00_y06.tif")
    assert teh["is_source_imagery"] is True
    assert teh["reason"] == "resolution_regime_excluded"

    # iran duplicates of Teheran's: byte-identical, so `duplicate_of`.
    for p in ("iran/Ax00_y06.tif", "iran/Ax01_y06.tif"):
        d = _rec(inv, p)
        assert d["reason"] == "duplicate_of", p
        assert d["duplicate_of"].startswith("Teheran/"), p

    # A 4-band binary mask: rejected as derived_raster, which must NOT be
    # conflated with no_georeference or resolution_regime_excluded. It has the
    # full three bands, so only the value distribution can reject it.
    mask = _rec(inv, "leb/tmp_results/2025-06-06_output_Trees_ST48_no_softmax.tif")
    assert mask["band_count"] == 4
    assert mask["is_source_imagery"] is False
    assert mask["reason"] == "derived_raster"
    assert max(mask["distinct_per_band"]) <= 6

    # FINDING (S1): docs/DATA.md says every derived raster has <= 6 distinct
    # values per band. Some do not -- the softmax/`ST_output` rasters reach
    # 13-16. The rule is unaffected (the threshold is 64 and source rasters
    # start at 187) but the manifest's "<= 6" is too strong.
    st = _rec(inv, "leb/cache/2022-10-29_ST_output.tiff")
    assert st["is_source_imagery"] is False
    assert st["reason"] == "derived_raster"
    assert 6 < max(st["distinct_per_band"][:3]) < 64

    # A 1-band mask: rejected on band count, a distinct reason again.
    thin = _rec(inv, "leb/cache/2022-10-29_Collis_k1_new_binary.tiff")
    assert thin["band_count"] == 1
    assert thin["reason"] == "too_few_bands"

    # The label raster is 1-band: too_few_bands, a third distinct reason.
    lbl = _rec(inv, "tile_cropped_x3308_y3674_z0.125.tif")
    assert lbl["reason"] == "too_few_bands"
    assert lbl["band_count"] == 1

    # All four rejection reasons are actually in use and distinguishable.
    reasons = {r["reason"] for r in inv["rasters"] if r["reason"]}
    assert {
        "derived_raster",
        "too_few_bands",
        "no_georeference",
        "resolution_regime_excluded",
        "duplicate_of",
    } <= reasons
    assert reasons <= inventory.REASONS


def test_summary_line_states_counts_per_reason(inv):
    line = inventory.summary_line(inv)
    assert line.startswith("8 indexable")
    for token in ("derived_raster", "no_georeference", "resolution_regime_excluded", "duplicate_of"):
        assert token in line


# --------------------------------------------------------------------------
# D-3
# --------------------------------------------------------------------------


def test_data_dir_readonly(tmp_path):
    before = inventory.tree_snapshot(DATA_ROOT)

    inv = inventory.build_inventory()
    out = inventory.write_inventory(inv)

    after = inventory.tree_snapshot(DATA_ROOT)
    assert after == before, "the data tree changed across an inventory run"
    assert inventory.tree_digest(DATA_ROOT) == inv["data_tree_digest"]

    # Every output path the module can produce resolves inside retrieval/index/
    index_dir = inventory.INDEX_DIR.resolve()
    produced = [Path(out).resolve(), *(Path(p).resolve() for p in inventory.output_paths())]
    for p in produced:
        assert p == index_dir or index_dir in p.parents, f"{p} escapes retrieval/index/"
        assert DATA_ROOT.resolve() not in p.parents, f"{p} is inside the data root"

    # write_inventory refuses a destination outside retrieval/index/.
    with pytest.raises(ValueError, match="refusing to write outside"):
        inventory.write_inventory(inv, tmp_path / "escaped.json")
    with pytest.raises(ValueError, match="refusing to write outside"):
        inventory.write_inventory(inv, DATA_ROOT / "inventory.json")
    assert not (DATA_ROOT / "inventory.json").exists()

    # No module in this stage may ever open a raster for writing.
    src = inventory.SRC_DIR
    for name in ("inventory.py", "geo.py"):
        text = (src / name).read_text()
        for bad in re.finditer(r"rasterio\.open\([^)]*", text):
            frag = bad.group(0)
            assert not re.search(r"""["'](r\+|w|w\+|a)["']""", frag), f"{name}: {frag}"


# --------------------------------------------------------------------------
# D-4
# --------------------------------------------------------------------------


def test_inventory_deterministic():
    a = inventory.build_inventory()
    b = inventory.build_inventory()
    ja = json.dumps(a, sort_keys=True, indent=2)
    jb = json.dumps(b, sort_keys=True, indent=2)
    assert ja == jb, "two inventory runs over unchanged inputs disagree"

    pa = inventory.write_inventory(a, inventory.INDEX_DIR / "inventory_run_a.json")
    pb = inventory.write_inventory(b, inventory.INDEX_DIR / "inventory_run_b.json")
    assert Path(pa).read_bytes() == Path(pb).read_bytes()

    # The recorded sampling strategy is what makes this reproducible, so it
    # must be in the artifact.
    assert a["sampling"]["strategy"] == "fixed_stride_window_grid"
    assert a["sampling"]["grid"] >= 2 and a["sampling"]["window_px"] >= 8
    assert a["sampling"]["seed"] is None  # no RNG at all: a fixed grid


def test_the_pixel_rule_has_a_wide_margin(inv):
    """The 64-level threshold must not be delicate. Report the real gap: the
    most-varied band of any non-source raster against the least-varied band of
    any source raster."""
    ceiling = inv["counts"]["max_distinct_among_non_source"]
    floor = inv["counts"]["min_distinct_among_source"]
    assert ceiling < 64 < floor, f"non-source tops out at {ceiling}, source starts at {floor}"
    assert floor - ceiling > 100, f"margin is only {floor - ceiling} levels"
