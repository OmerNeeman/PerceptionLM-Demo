"""D-2: all reported ground distances are TRUE ground metres.

The trap (CLAUDE.md trap 3, spec D-2, docs/DATA.md):

  * `leb` is EPSG:3857. Mercator metres are inflated by 1/cos(latitude), so
    the 0.125002 m the geotransform reports is **not** a ground distance.
    True GSD is 10.47 cm and a 448 px tile frames 46.91 m, not 56.0 m.
  * The UTM scenes (EPSG:32636) are within 0.04% of unity. Applying the
    cosine correction there is exactly as wrong as omitting it for `leb`
    -- it would report ~8.4 cm instead of 10.00 cm.

So these tests pin both directions, and explicitly reject the two plausible
regressions (raw geotransform; correction applied globally).

Run:  env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python -m pytest
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

import geo

DATA_ROOT = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data")
LEB_MERCATOR = DATA_ROOT / "leb" / "2022-10-29.tif"
UTM_SCENE = DATA_ROOT / "X605_Y3388.tif"
TILE_PX = 448


# --------------------------------------------------------------------------
# The CRS classification must be derived from the CRS, not from the filename
# and not from a per-file table.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epsg,expected",
    [
        (3857, "mercator"),  # WGS 84 / Pseudo-Mercator -- needs the correction
        (3395, "mercator"),  # WGS 84 / World Mercator   -- also needs it
        (32636, "projected"),  # UTM 36N -- transverse Mercator, NO correction
        (32639, "projected"),  # UTM 39N -- ditto
        (4326, "geographic"),  # degrees, handled separately
    ],
)
def test_crs_kind_is_derived_from_the_crs(epsg, expected):
    assert geo.crs_kind(CRS.from_epsg(epsg)) == expected


def test_crs_kind_of_none_is_none():
    assert geo.crs_kind(None) == "none"


def test_tile_ground_extent_is_gsd_times_pixels():
    assert geo.tile_ground_extent_m(0.10471, 448) == pytest.approx(46.91, abs=0.05)
    assert geo.tile_ground_extent_m(0.10, 448) == pytest.approx(44.8, abs=0.05)


# --------------------------------------------------------------------------
# D-2 -- Mercator: the correction MUST be applied.
# --------------------------------------------------------------------------


def test_gsd_correction():
    res = geo.ground_resolution_for(LEB_MERCATOR)

    # The outcome first, so a regression names the wrong number out loud.
    gsd_cm = res.true_gsd_m * 100.0
    assert gsd_cm == pytest.approx(10.47, abs=0.01), f"true GSD is {gsd_cm:.2f} cm/px"

    extent = geo.tile_ground_extent_m(res.true_gsd_m, TILE_PX)
    assert extent == pytest.approx(46.91, abs=0.05), f"448 px tile is {extent:.2f} m"
    assert res.tile_extent_m(TILE_PX) == pytest.approx(46.91, abs=0.05)

    # Then the mechanism that produced it.
    assert res.crs_kind == "mercator"
    assert res.projected_px_m == pytest.approx(0.125002, abs=1e-5)
    assert res.centre_lat == pytest.approx(33.10707, abs=0.01)
    assert res.scale_correction == pytest.approx(0.837651, abs=1e-4)

    # Explicit rejection of the raw-geotransform regression. A regression here
    # must fail loudly, not look plausible.
    assert not math.isclose(gsd_cm, 12.5, abs_tol=0.05), "raw geotransform GSD returned"
    assert not math.isclose(extent, 56.0, abs_tol=0.5), "raw geotransform extent returned"
    assert gsd_cm < 11.0
    assert extent < 50.0


def test_mercator_correction_agrees_with_an_independent_geodesic_measurement():
    """Cross-check: measure one pixel on the WGS84 ellipsoid, no Mercator maths."""
    res = geo.ground_resolution_for(LEB_MERCATOR)
    assert res.geodesic_gsd_m is not None
    assert res.geodesic_gsd_m == pytest.approx(res.true_gsd_m, rel=0.005)
    assert res.geodesic_gsd_m * 100 == pytest.approx(10.47, abs=0.02)


# --------------------------------------------------------------------------
# D-2 (UTM) -- the correction MUST NOT be applied.
# --------------------------------------------------------------------------


def test_gsd_no_correction_for_utm():
    res = geo.ground_resolution_for(UTM_SCENE)

    # Outcome first.
    gsd_cm = res.true_gsd_m * 100.0
    assert gsd_cm == pytest.approx(10.00, abs=0.01), f"true GSD is {gsd_cm:.2f} cm/px"

    extent = geo.tile_ground_extent_m(res.true_gsd_m, TILE_PX)
    assert extent == pytest.approx(44.8, abs=0.05), f"448 px tile is {extent:.2f} m"

    # Then the mechanism: this CRS takes NO correction.
    assert res.crs_kind == "projected"
    assert res.scale_correction == 1.0
    assert res.projected_px_m == pytest.approx(0.100000, abs=1e-6)

    # A cosine correction wrongly applied at lat ~31.4 would give ~8.5 cm and
    # a ~38 m tile. Reject that explicitly.
    assert gsd_cm > 9.9, "cosine correction wrongly applied to a UTM scene"
    assert not math.isclose(gsd_cm, 8.4, abs_tol=0.5)
    assert not math.isclose(extent, 38.2, abs_tol=1.0)


def test_utm_no_correction_agrees_with_an_independent_geodesic_measurement():
    """The 0.04%-of-unity claim, measured rather than asserted."""
    res = geo.ground_resolution_for(UTM_SCENE)
    assert res.geodesic_gsd_m is not None
    assert res.geodesic_gsd_m == pytest.approx(res.projected_px_m, rel=0.0004)
    assert res.geodesic_gsd_m * 100 == pytest.approx(10.00, abs=0.01)


def test_the_two_regimes_frame_the_same_ground_area_within_5_percent():
    """docs/DATA.md: one 448 px tile means the same thing across all 8 scenes."""
    a = geo.ground_resolution_for(LEB_MERCATOR).tile_extent_m(TILE_PX)
    b = geo.ground_resolution_for(UTM_SCENE).tile_extent_m(TILE_PX)
    assert abs(a - b) / max(a, b) < 0.05


# --------------------------------------------------------------------------
# D-2 (geographic) -- Fix 2 (S1_fix.md): of 16 mutations the reviewer tried,
# exactly one survived all 21 tests: `gsd_x, gsd_y = gx, gy` (metres, correct)
# replaced by `px_x, px_y` (degrees, reported as if they were metres) at
# geo.py:193. Nothing ever called ground_resolution on a geographic raster,
# so a ~111,000x error was invisible. These two real EPSG:4326 rasters close
# that gap: one is spec E-1's primary ground-truth label raster.
# --------------------------------------------------------------------------

GEOGRAPHIC_LABEL_RASTER = DATA_ROOT / "tile_cropped_x3308_y3674_z0.125.tif"
GEOGRAPHIC_SIN_RASTER = DATA_ROOT / "sin" / "sin_min.tiff"


def test_ground_resolution_geographic():
    """A geographic (EPSG:4326) raster's true GSD must come back in METRES,
    not degrees. Under the surviving mutant (px_x, px_y in degrees returned
    as if they were the geodesic metres) this fails by a factor of ~111,000:
    0.00011 cm instead of 12.17 cm."""
    lbl = geo.ground_resolution_for(GEOGRAPHIC_LABEL_RASTER)
    assert lbl is not None
    assert lbl.crs_kind == "geographic"
    gsd_cm = lbl.true_gsd_m * 100.0
    assert gsd_cm == pytest.approx(12.17, abs=0.05), f"true GSD is {gsd_cm:.5f} cm/px"
    # Explicit rejection of the degrees-as-metres regression: that mutant
    # reports ~0.00011 cm, off by ~5 orders of magnitude.
    assert gsd_cm > 1.0, "degrees appear to have been returned as metres"

    sin = geo.ground_resolution_for(GEOGRAPHIC_SIN_RASTER)
    assert sin is not None
    assert sin.crs_kind == "geographic"
    sin_gsd_cm = sin.true_gsd_m * 100.0
    assert sin_gsd_cm == pytest.approx(408.33, abs=0.5), f"true GSD is {sin_gsd_cm:.5f} cm/px"
    assert sin_gsd_cm > 1.0, "degrees appear to have been returned as metres"


# --------------------------------------------------------------------------
# Fix 3 (S1_fix.md): geodesic_gsd_m is computed for every raster and then
# discarded, never used as a check. `_scale_correction` applies cos(lat) to
# any CRS classified "mercator", which is only right when the standard
# parallel IS the equator (lat_ts == 0). For a secant Mercator (lat_ts != 0,
# e.g. EPSG:3994 "Mercator 41") the true factor is cos(lat)/cos(lat_ts), so
# the naive rule silently mis-reports by 24-29%. ground_resolution must
# cross-check against geodesic_gsd_m and raise (never silently substitute)
# when they disagree beyond GSD_GUARD_TOLERANCE.
# --------------------------------------------------------------------------


def _make_memory_raster(crs, transform, width=32, height=32):
    """A tiny synthetic 3-band raster, built entirely in memory -- no write
    under the read-only data root, nothing left on disk."""
    data = (np.arange(width * height * 3, dtype="uint16") % 200).reshape(3, height, width)
    memfile = rasterio.io.MemoryFile()
    with memfile.open(
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint16",
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(data)
    return memfile


def test_gsd_guard_fires_on_secant_mercator():
    """EPSG:3994 (Mercator 41, lat_ts=-41) centred near the equator: the naive
    cos(lat) rule and the independent geodesic measurement disagree by ~24%,
    matching the brief's measured `rule 8.7780 cm vs geodesy 11.6232 cm`."""
    crs = CRS.from_epsg(3994)
    px = 0.08778  # metres; chosen to land near the brief's "rule 8.7780 cm"
    w = h = 32
    # lon_0=100, and y=0 is the equator for a Mercator projection regardless
    # of lat_ts -- so this places the raster's centre at lat ~= 0.
    transform = Affine(px, 0.0, -px * w / 2.0, 0.0, -px, px * h / 2.0)
    memfile = _make_memory_raster(crs, transform, w, h)
    try:
        with memfile.open() as ds:
            with pytest.raises(geo.GsdGuardError) as excinfo:
                geo.ground_resolution(ds.crs, ds.transform, ds.width, ds.height)
    finally:
        memfile.close()

    msg = str(excinfo.value)
    assert "3994" in msg
    assert "rule" in msg and "geodesy" in msg
    # Both figures actually appear, roughly matching the brief's measurement.
    assert "8.7" in msg or "8.8" in msg
    assert "11.6" in msg


EIGHT_INDEXABLE_SCENES = [
    DATA_ROOT / "leb" / "2022-10-29.tif",
    DATA_ROOT / "leb" / "2025-06-06.tif",
    DATA_ROOT / "AYOSH" / "X693_Y3500.tif",
    DATA_ROOT / "AYOSH" / "X693_Y3501.tif",
    DATA_ROOT / "gaza" / "X625_Y3404.tif",
    DATA_ROOT / "gaza" / "X625_Y3405.tif",
    DATA_ROOT / "gaza" / "X625_Y3406.tif",
    DATA_ROOT / "X605_Y3388.tif",
]


def test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes():
    """The guard must be silent on real data -- all eight scenes docs/DATA.md
    records as the index. If it fires here, the analytic rule is wrong for
    real imagery -- a finding for the PM, not a tolerance to widen
    (S1_fix.md)."""
    for path in EIGHT_INDEXABLE_SCENES:
        res = geo.ground_resolution_for(path)  # must not raise
        assert res is not None, path
