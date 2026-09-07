"""Ground-resolution and footprint maths -- spec D-2.

    Every ground distance this project reports is a TRUE ground distance.

The trap (CLAUDE.md trap 3, docs/DATA.md "The GSD trap"): a projected pixel
size is not a ground distance. For a **Mercator** CRS the projection inflates
distances by `1/cos(latitude)`, so `leb`'s 0.125002 m pixel is 10.47 cm on the
ground and a 448 px tile frames 46.91 m -- not 56.0 m, a 19.4% error.

The correction is **conditional on the CRS**, and the condition is read off
the CRS itself, never from a filename or a per-file table:

    EPSG:3857  Pseudo-Mercator   -> mercator   -> x cos(lat)
    EPSG:3395  World Mercator    -> mercator   -> x cos(lat)
    EPSG:32636 UTM 36N           -> projected  -> x 1.0   (within 0.04% of unity)
    EPSG:4326  lon/lat degrees   -> geographic -> measured geodesically

Applying the cosine to a UTM scene is exactly as wrong as omitting it for
`leb`: it would report ~8.5 cm instead of 10.00 cm.

Every raster is opened READ-ONLY (spec D-3). `rasterio.open(path)` defaults to
mode "r"; no call in this module passes any other mode.

Environment (CLAUDE.md): every invocation must be prefixed inline --

    env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python ...
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import pyproj
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform as warp_transform

log = logging.getLogger(__name__)

# pyproj cannot open the PROJ database in this env (it warns at import and
# CRS construction raises), but pyproj.Geod is pure ellipsoidal geodesy and
# needs no database. All CRS work therefore goes through rasterio/GDAL, all
# distance work through Geod. Verified in S1.
_GEOD = pyproj.Geod(ellps="WGS84")

WGS84 = "EPSG:4326"

# PROJ operation names, lowercased, that denote a Mercator-family projection.
# "utm"/"tmerc" are transverse Mercator and are deliberately NOT in this set.
_MERCATOR_PROJ_NAMES = frozenset({"merc", "webmerc"})

CRS_KINDS = ("mercator", "projected", "geographic", "none")


def crs_kind(crs) -> str:
    """Classify a CRS into the regime that decides the ground-distance rule.

    Derived from the CRS definition itself -- no filename, no lookup table.
    """
    if crs is None:
        return "none"
    c = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
    if c.is_geographic:
        return "geographic"
    name = None
    try:
        name = c.to_dict().get("proj")
    except Exception as exc:  # pragma: no cover - a CRS GDAL cannot export
        log.warning("crs_kind: could not export CRS to a PROJ dict: %s", exc)
    if name and name.lower() in _MERCATOR_PROJ_NAMES:
        return "mercator"
    # Secondary signal for CRSs whose PROJ export is lossy: the WKT
    # conversion method. "Transverse Mercator" must not match.
    try:
        wkt = c.to_wkt().lower()
    except Exception:  # pragma: no cover
        wkt = ""
    if "mercator" in wkt and "transverse" not in wkt:
        return "mercator"
    return "projected"


def is_georeferenced(crs, transform) -> bool:
    """True iff the raster carries both a CRS and a real geotransform.

    rasterio hands back the identity matrix for a raster with no geotransform,
    which is why the identity is treated as absent rather than as a 1 m pixel.
    """
    if crs is None or transform is None:
        return False
    return transform != Affine.identity()


def tile_ground_extent_m(gsd_m: float, tile_px: int) -> float:
    """Ground side length of a square tile of `tile_px` pixels, in TRUE metres."""
    return gsd_m * tile_px


@dataclass(frozen=True)
class GroundResolution:
    """One raster's resolution, in projected units and in true ground metres."""

    crs: str | None
    crs_kind: str
    projected_px_x_m: float
    projected_px_y_m: float
    centre_lon: float | None
    centre_lat: float | None
    scale_correction: float | None  # None where a cosine factor is meaningless
    true_gsd_x_m: float
    true_gsd_y_m: float
    geodesic_gsd_m: float | None  # independent cross-check, not the source

    @property
    def projected_px_m(self) -> float:
        return (self.projected_px_x_m + self.projected_px_y_m) / 2.0

    @property
    def true_gsd_m(self) -> float:
        return (self.true_gsd_x_m + self.true_gsd_y_m) / 2.0

    def tile_extent_m(self, tile_px: int) -> float:
        return tile_ground_extent_m(self.true_gsd_m, tile_px)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["projected_px_m"] = self.projected_px_m
        d["true_gsd_m"] = self.true_gsd_m
        return d


def _scale_correction(kind: str, centre_lat: float | None) -> float | None:
    """The dimensionless projected->ground factor for this CRS regime.

    Mercator is conformal, so x and y are inflated identically by 1/cos(lat);
    the correction is therefore cos(lat) on both axes. Every other projected
    CRS in this project's regime is within 0.04% of unity and takes no
    correction at all.
    """
    if kind == "mercator":
        if centre_lat is None:
            return None
        return math.cos(math.radians(centre_lat))
    if kind == "projected":
        return 1.0
    return None  # geographic / none: a cosine factor is not the right model


def _centre_lonlat(crs, transform, width: int, height: int):
    x, y = transform * (width / 2.0, height / 2.0)
    lon, lat = warp_transform(crs, WGS84, [x], [y])
    return lon[0], lat[0], x, y


def _geodesic_pixel_m(crs, x: float, y: float, px_x: float, px_y: float):
    """Measure one pixel on the WGS84 ellipsoid. Independent of any GSD rule."""
    xs = [x, x + px_x, x]
    ys = [y, y, y + px_y]
    lons, lats = warp_transform(crs, WGS84, xs, ys)
    _, _, dx = _GEOD.inv(lons[0], lats[0], lons[1], lats[1])
    _, _, dy = _GEOD.inv(lons[0], lats[0], lons[2], lats[2])
    return dx, dy


def ground_resolution(crs, transform, width: int, height: int) -> GroundResolution | None:
    """True ground resolution, or None if the raster is not georeferenced."""
    if not is_georeferenced(crs, transform):
        return None

    px_x, px_y = abs(transform.a), abs(transform.e)
    lon, lat, x, y = _centre_lonlat(crs, transform, width, height)
    kind = crs_kind(crs)

    try:
        gx, gy = _geodesic_pixel_m(crs, x, y, px_x, px_y)
        geodesic = (gx + gy) / 2.0
    except Exception as exc:  # reported, never swallowed
        log.warning("geodesic cross-check unavailable for CRS %s: %s", crs, exc)
        geodesic = None

    factor = _scale_correction(kind, lat)
    if kind == "geographic":
        # Degrees are not metres; the geodesic measurement *is* the answer.
        if geodesic is None:
            log.warning("geographic CRS %s with no geodesic measurement", crs)
            return None
        gsd_x, gsd_y = gx, gy
    else:
        gsd_x, gsd_y = px_x * factor, px_y * factor

    return GroundResolution(
        crs=str(crs),
        crs_kind=kind,
        projected_px_x_m=px_x,
        projected_px_y_m=px_y,
        centre_lon=lon,
        centre_lat=lat,
        scale_correction=factor,
        true_gsd_x_m=gsd_x,
        true_gsd_y_m=gsd_y,
        geodesic_gsd_m=geodesic,
    )


def ground_resolution_for(path: str | Path) -> GroundResolution | None:
    """Open a raster READ-ONLY and report its true ground resolution."""
    with warnings.catch_warnings():
        # A missing geotransform is a fact we record, not noise to print.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:  # mode defaults to "r"
            return ground_resolution(ds.crs, ds.transform, ds.width, ds.height)
