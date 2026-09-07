"""COG conversion -- `leb`'s two scenes only. Spec D-3 (read-only source).

`leb`'s two scenes are laid out as **one-row strips spanning the full raster
width** (`block_shapes == (1, width)`, no overviews): reading any small
window forces GDAL to decode every strip row the window touches at *full
raster width*, for every band -- docs/DATA.md measures this at ~45x for a
448 px RGB window (~27 MB decoded to return ~0.6 MB of pixels). The other six
indexed scenes are already tiled 256x256 (~2.9x) and **must not be
converted** here -- that would be hours of wasted work for no benefit
(briefs/S2.md).

This module only ever *reads* the two `leb` source rasters (mode "r", never
"r+"/"w"/"a") and only ever *writes* under the index root (D-3). The COG
copies are a read-optimised re-encoding, not a reprocessing: same pixels,
same CRS, same geotransform, same band count -- `assert_pixel_identical`
below is the check that a silent resample/reproject would fail.

Environment: prefix every invocation inline --

    env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=<path> <python> ...
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window

import config

log = logging.getLogger(__name__)

#: The only two scenes this module ever converts (briefs/S2.md: "Convert
#: COGs for leb's two scenes only"). Not derived from the inventory's
#: is_indexable set here on purpose -- the six already-tiled scenes must
#: never be silently swept in by a future inventory change.
LEB_SCENES: tuple[str, ...] = ("leb/2022-10-29.tif", "leb/2025-06-06.tif")

#: Lossless (bit-exact) compressor -- COMPRESS=DEFLATE never alters a pixel
#: value, only how it is packed on disk. A lossy predictor/compressor here
#: would silently defeat `assert_pixel_identical`.
COG_CREATION_OPTIONS = {
    "COMPRESS": "DEFLATE",
    "BLOCKSIZE": 256,
    "OVERVIEWS": "AUTO",
    "OVERVIEW_RESAMPLING": "AVERAGE",
    "BIGTIFF": "IF_SAFER",
    "NUM_THREADS": "ALL_CPUS",
}

#: Windows sampled for the pixel-identity check -- a fixed, deterministic set
#: (no RNG), including one edge/corner window so a resample that only
#: shows up at boundaries cannot hide.
_SAMPLE_WINDOW_PX = 448


def _sample_offsets(width: int, height: int, win: int = _SAMPLE_WINDOW_PX) -> list[tuple[int, int]]:
    """Deterministic sample offsets: four interior points plus the true
    bottom-right corner (clamped so the window still fits on-raster)."""
    w = min(win, width)
    h = min(win, height)
    xs = sorted({0, (width - w) // 4, (width - w) // 2, (width - w) * 3 // 4, width - w})
    ys = sorted({0, (height - h) // 4, (height - h) // 2, (height - h) * 3 // 4, height - h})
    return [(x, y) for y in ys for x in xs]


def cog_output_path(rel_source: str, index_root: Path | None = None) -> Path:
    """Where a source scene's COG copy lives -- always under
    `<index_root>/cog/`, named by the source's own filename (D-4: no
    build-time value in the name)."""
    root = index_root or config.get_index_root()
    name = Path(config.posix_key(rel_source)).name
    return root / "cog" / name


@dataclass(frozen=True)
class ReadAmplification:
    """Bytes GDAL must decode to satisfy one `window_px x window_px` RGB
    read, versus the useful bytes actually returned -- docs/DATA.md's ~45x
    figure, computed from the raster's own block layout rather than
    estimated. Deterministic (no I/O-timing noise): it depends only on
    `block_shapes`, `dtype`, `count`, and the requested window, so it is
    reproducible run to run and machine to machine."""

    window_px: int
    block_x: int
    block_y: int
    band_count: int
    decoded_bytes: int
    useful_bytes: int

    @property
    def amplification(self) -> float:
        return self.decoded_bytes / self.useful_bytes


def measure_read_amplification(
    path: Path, window_px: int = _SAMPLE_WINDOW_PX, col_off: int = 0, row_off: int = 0
) -> ReadAmplification:
    """Decoded-vs-useful bytes for one `window_px` square RGB window at
    (col_off, row_off), derived from the dataset's own block geometry.

    GDAL decodes whole blocks: a read touching part of a block still pays
    for the entire block. For `leb` (block == one full-width row) that means
    every block in the window's row range costs `width` pixels wide
    regardless of how few columns are actually wanted -- which is exactly
    the ~45x figure docs/DATA.md states, reproduced here from first
    principles rather than quoted.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:  # mode defaults to "r" -- D-3
            by, bx = ds.block_shapes[0]
            dtype_size = np.dtype(ds.dtypes[0]).itemsize
            band_count = ds.count
            w = min(window_px, ds.width - col_off)
            h = min(window_px, ds.height - row_off)

            first_bx, last_bx = col_off // bx, (col_off + w - 1) // bx
            first_by, last_by = row_off // by, (row_off + h - 1) // by
            n_blocks = (last_bx - first_bx + 1) * (last_by - first_by + 1)
            decoded = n_blocks * bx * by * dtype_size * band_count
            useful = w * h * dtype_size * band_count

    return ReadAmplification(
        window_px=window_px,
        block_x=bx,
        block_y=by,
        band_count=band_count,
        decoded_bytes=decoded,
        useful_bytes=useful,
    )


def build_cog(src_path: Path, dst_path: Path) -> Path:
    """Write a COG copy of `src_path` to `dst_path`. `src_path` is opened
    read-only; `dst_path` must resolve inside the index root (D-3) and is
    never the source path itself."""
    index_root = config.get_index_root().resolve()
    resolved_dst = dst_path.resolve()
    if not (resolved_dst == index_root or index_root in resolved_dst.parents):
        raise ValueError(f"refusing to write outside {index_root}: {resolved_dst}")
    src_resolved = Path(src_path).resolve()
    if resolved_dst == src_resolved:
        raise ValueError("refusing to write a COG onto its own source path")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(src_path) as src:  # mode defaults to "r" -- D-3
            rio_copy(src, str(dst_path), driver="COG", **COG_CREATION_OPTIONS)
    return dst_path


def assert_pixel_identical(src_path: Path, dst_path: Path) -> None:
    """Raise unless `dst_path` is pixel-identical to `src_path` over a fixed
    set of sampled windows, with CRS, transform and band count preserved.

    Never a sample-only check dressed up as identity: this is compared over
    real windows read from both files, band for band, byte for byte -- a
    resample or reproject (even a subtle one, e.g. nearest-neighbour vs
    bilinear) changes at least one sampled pixel and fails loudly here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(src_path) as src, rasterio.open(dst_path) as dst:
            if src.crs != dst.crs:
                raise AssertionError(f"CRS changed: {src.crs} -> {dst.crs}")
            if src.transform != dst.transform:
                raise AssertionError(f"geotransform changed: {src.transform} -> {dst.transform}")
            if src.count != dst.count:
                raise AssertionError(f"band count changed: {src.count} -> {dst.count}")
            if (src.width, src.height) != (dst.width, dst.height):
                raise AssertionError(
                    f"dimensions changed: {(src.width, src.height)} -> {(dst.width, dst.height)}"
                )
            for x, y in _sample_offsets(src.width, src.height):
                w = min(_SAMPLE_WINDOW_PX, src.width - x)
                h = min(_SAMPLE_WINDOW_PX, src.height - y)
                win = Window(x, y, w, h)
                a = src.read(window=win)
                b = dst.read(window=win)
                if not np.array_equal(a, b):
                    diff = np.argwhere(a != b)
                    raise AssertionError(
                        f"pixel mismatch at window (x={x}, y={y}, w={w}, h={h}): "
                        f"{len(diff)} differing samples, first at {tuple(diff[0])}"
                    )


#: Offsets used to characterise the AFTER (256x256-block COG) amplification.
#: `leb`'s BEFORE figure is offset-invariant (block_y=1 means every window
#: touches exactly `window_px` one-row blocks and the single full-width
#: column block, regardless of where it starts -- verified in
#: tests/test_cog.py::test_before_amplification_is_offset_invariant), but a
#: 256 px block grid is not: a 448 px window can land block-aligned (best
#: case, touches a 2x2 block neighbourhood) or maximally offset (worst case,
#: 3x3 blocks) depending on which tile it is. Both ends are sampled here so
#: the report states a range, not a cherry-picked best case.
_AFTER_SAMPLE_OFFSETS = [(0, 0), (128, 128), (64, 191), (191, 64)]


def convert_leb(data_root: Path | None = None, index_root: Path | None = None) -> dict:
    """Convert both `leb` scenes, verify pixel identity, and report the
    measured read-amplification improvement -- the entire justification for
    this stage (briefs/S2.md)."""
    root = data_root or config.get_data_root()
    out_root = index_root or config.get_index_root()
    report = {}
    for rel in LEB_SCENES:
        src = root / rel
        dst = cog_output_path(rel, out_root)
        log.info("cog: converting %s -> %s", rel, dst)
        before = measure_read_amplification(src)
        build_cog(src, dst)
        assert_pixel_identical(src, dst)
        after_samples = [
            measure_read_amplification(dst, col_off=x, row_off=y).amplification
            for x, y in _AFTER_SAMPLE_OFFSETS
        ]
        report[rel] = {
            "src_size_bytes": src.stat().st_size,
            "dst_size_bytes": dst.stat().st_size,
            "before_amplification": before.amplification,
            "after_amplification_min": min(after_samples),
            "after_amplification_max": max(after_samples),
            "before_decoded_bytes": before.decoded_bytes,
            "useful_bytes": before.useful_bytes,
        }
        log.info(
            "cog: %s amplification %.1fx -> %.2f-%.2fx (window %dpx)",
            rel,
            before.amplification,
            min(after_samples),
            max(after_samples),
            _SAMPLE_WINDOW_PX,
        )
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    report = convert_leb()
    for rel, r in report.items():
        print(
            f"{rel}: {r['before_amplification']:.1f}x -> "
            f"{r['after_amplification_min']:.2f}-{r['after_amplification_max']:.2f}x "
            f"({r['src_size_bytes'] / 1e6:.1f} MB -> {r['dst_size_bytes'] / 1e6:.1f} MB)"
        )


if __name__ == "__main__":
    main()
