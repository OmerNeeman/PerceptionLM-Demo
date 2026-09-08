"""COG conversion -- `leb`'s two scenes only. See briefs/S2.md.

`leb`'s two source scenes are one-row-strip TIFFs with no overviews: a random
448 px RGB window costs ~45x read amplification. The other six indexed
scenes are already tiled 256x256 (~2.9x) and must NOT be converted. This
file's acceptance test is `test_cog_pixel_identical`; the surrounding tests
pin the amplification numbers that are this stage's entire justification and
the "leb only" scope.

Run:
    env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<data root> \
        /home/omer/anaconda3/envs/geo/bin/python -m pytest \
        retrieval/tests/test_cog.py -v

Building both COGs takes several seconds -- the module-scoped fixture below
builds them once and every test in this file reuses that build.
"""

from __future__ import annotations

import pytest
import rasterio

import cog
import config


@pytest.fixture(scope="module")
def built_cogs():
    return cog.convert_leb()


# --------------------------------------------------------------------------
# COG -- test_cog_pixel_identical
# --------------------------------------------------------------------------


def test_cog_pixel_identical(built_cogs):
    for rel in cog.LEB_SCENES:
        src = config.get_data_root() / rel
        dst = cog.cog_output_path(rel)
        assert dst.exists(), f"COG not written for {rel}"
        cog.assert_pixel_identical(src, dst)  # raises loudly on any mismatch


def test_assert_pixel_identical_detects_a_genuinely_different_copy(tmp_path):
    """The check itself, proven against a deliberately corrupted copy --
    entirely synthetic, scratch-only rasters (never the real leb scenes) so
    this runs in milliseconds and never risks the data root. A real COG
    build that silently resampled or reprojected would look exactly like
    this: same CRS/transform/band count/dimensions, differing pixels."""
    import numpy as np
    from rasterio.transform import from_origin

    width = height = 64
    data = (np.arange(width * height * 3, dtype="int64") % 256).astype("uint8").reshape(3, height, width)
    transform = from_origin(709632.0, 3585024.0, 1.0, 1.0)

    def _write(path, arr):
        with rasterio.open(
            path, "w", driver="GTiff", width=width, height=height, count=3,
            dtype="uint8", crs="EPSG:32636", transform=transform,
        ) as ds:
            ds.write(arr)

    src_path = tmp_path / "src.tif"
    good_path = tmp_path / "good_copy.tif"
    bad_path = tmp_path / "bad_copy.tif"
    _write(src_path, data)
    _write(good_path, data)  # identical pixels -- a legitimate "copy"
    corrupted = data.copy()
    corrupted[:, 10, 10] += 1  # one differing pixel is enough to matter
    _write(bad_path, corrupted)

    cog.assert_pixel_identical(src_path, good_path)  # must not raise
    with pytest.raises(AssertionError):
        cog.assert_pixel_identical(src_path, bad_path)  # must raise


def test_cog_preserves_crs_transform_band_count(built_cogs):
    for rel in cog.LEB_SCENES:
        src = config.get_data_root() / rel
        dst = cog.cog_output_path(rel)
        with rasterio.open(src) as s, rasterio.open(dst) as d:
            assert s.crs == d.crs
            assert s.transform == d.transform
            assert s.count == d.count
            assert (s.width, s.height) == (d.width, d.height)


def test_cog_is_actually_tiled_not_row_strips(built_cogs):
    """The whole point: block layout must change from (1, width) to a real
    tile grid, or nothing was gained."""
    for rel in cog.LEB_SCENES:
        dst = cog.cog_output_path(rel)
        with rasterio.open(dst) as d:
            by, bx = d.block_shapes[0]
            assert by > 1 and bx < d.width, f"{rel}: COG is not actually block-tiled ({bx}x{by})"


def test_only_leb_scenes_are_converted(built_cogs):
    cog_dir = config.get_index_root() / "cog"
    written = sorted(p.name for p in cog_dir.glob("*.tif"))
    expected = sorted(rel.split("/")[-1] for rel in cog.LEB_SCENES)
    assert written == expected, (
        f"only leb's two scenes may be converted; found {written} under {cog_dir}"
    )


# --------------------------------------------------------------------------
# The amplification numbers that justify this stage.
# --------------------------------------------------------------------------


def test_source_amplification_matches_docs_45x():
    for rel in cog.LEB_SCENES:
        src = config.get_data_root() / rel
        r = cog.measure_read_amplification(src)
        assert r.amplification == pytest.approx(45.0, rel=0.02), (
            f"{rel}: measured {r.amplification:.1f}x, docs/DATA.md states ~45x"
        )


def test_source_amplification_is_offset_invariant():
    """leb's block layout is one full-width row per block, so the number of
    blocks touched by a window read depends only on its height, never on
    where it starts -- verified directly rather than assumed."""
    src = config.get_data_root() / cog.LEB_SCENES[0]
    a = cog.measure_read_amplification(src, col_off=0, row_off=0)
    b = cog.measure_read_amplification(src, col_off=111, row_off=222)
    c = cog.measure_read_amplification(src, col_off=5000, row_off=3000)
    assert a.amplification == b.amplification == c.amplification


def test_cog_amplification_is_far_below_source(built_cogs):
    for rel in cog.LEB_SCENES:
        dst = cog.cog_output_path(rel)
        before = cog.measure_read_amplification(config.get_data_root() / rel)
        # A representative misaligned offset -- see cog.py's
        # _AFTER_SAMPLE_OFFSETS docstring for why this varies with alignment.
        after = cog.measure_read_amplification(dst, col_off=100, row_off=100)
        assert after.amplification < before.amplification / 10, (
            f"{rel}: COG amplification {after.amplification:.2f}x is not "
            f"far below source's {before.amplification:.1f}x"
        )
        assert after.amplification < 3.5  # docs/DATA.md's own tiled-group ceiling (~2.9x)


def test_cog_report_shape(built_cogs):
    for rel, r in built_cogs.items():
        assert r["after_amplification_max"] < r["before_amplification"]
        assert r["dst_size_bytes"] > 0
