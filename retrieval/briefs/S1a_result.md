# S1a result — config + device abstraction, setup half of INSTRUCTIONS.md

**Status: DONE**

Proves spec **N-8** in full and the setup half of **N-9**. All 44 tests
(39 pre-existing + 5 new) pass; N-3's `float16` assertion is unchanged and
still green. No behaviour change found or made.

## What changed

- **New `retrieval/src/config.py`** — `get_data_root()` (no machine default;
  raises `ConfigError` naming `AERIAL_DATA_ROOT` and both bash/PowerShell
  syntax if unset), `get_index_root()` (defaults to `<repo>/retrieval/index`,
  overridable via `AERIAL_INDEX_ROOT`), `get_model_cache_root()` (optional,
  `AERIAL_MODEL_CACHE_ROOT`), `get_forced_device()` (`AERIAL_DEVICE`), and
  `posix_key()` — canonicalises a path/string to `/`-separated form via
  `PureWindowsPath(...).as_posix()`, which accepts both separators as input.
- **New `retrieval/src/device.py`** — `resolve_device(force=None)` and
  `select_dtype(device, capability=None)`. `select_dtype` is a pure function;
  `capability` is an optional `(major, minor)` override so the Ampere branch
  is testable without Ampere hardware. Rule: CUDA cc<8.0 → `float16`, CUDA
  cc>=8.0 → `bfloat16`, CPU/MPS → `float32`.
- **`src/embedders.py`** — removed the module-level `DTYPE = torch.float16`
  constant. `load_embedder(candidate_id, device=None)`: with no `device` arg
  and no `AERIAL_DEVICE` set, behaviour is **byte-for-byte the same as
  before** — CUDA is required, the same trap-1 `RuntimeError` fires if it's
  unavailable, and `select_dtype(torch.device("cuda"))` on this cc-7.5
  machine resolves to `float16`, same as the old hardcoded constant.
  Passing `device="cpu"` (or `AERIAL_DEVICE=cpu`) is the new, additive path
  for N-8. `Embedder.embed_images` now casts to `self.dtype` (the model's
  actual loaded dtype) instead of the removed global — `self.dtype` was
  already defined and already equalled `DTYPE` in every prior case, so this
  is a no-op on existing behaviour. Also wired `AERIAL_MODEL_CACHE_ROOT`
  through to `hf_hub_download(..., cache_dir=...)`; `None` (unset) is a
  no-op, identical to the prior unconditional call.
- **`src/inventory.py`** — `DATA_ROOT` and `INDEX_DIR` now come from
  `config.py` instead of a hardcoded `Path("/home/omer/...")`. The
  per-file `rel` key (written into `inventory.json` as `"path"`, and used
  as the manifest key elsewhere) now goes through `config.posix_key(...)`
  instead of `str(...)` — identical output on this (POSIX) dev machine,
  because `Path.relative_to(...).as_posix() == str(...)` whenever there are
  no backslashes, which is always true on Linux — but portable if this ever
  ran on Windows. `tree_snapshot()`'s keys were deliberately left as
  `str(...)`: they never leave the single process/run that produces them
  (used only for the before/after read-only check), so there is no
  cross-OS boundary to protect and no reason to touch it.
- **`src/calibrate.py`** — the demo scene's absolute path is now
  `config.get_data_root() / "X605_Y3388.tif"` (filename only is
  project-specific; the folder comes from config).
- **`src/geo.py`** — no code path had a machine path; only its module
  docstring's example invocation line did. Scrubbed.
- **Docstrings** in all four retrofitted files had their example
  `/home/omer/anaconda3/envs/geo/bin/python` invocation lines replaced with
  a `<python>` placeholder plus a pointer to `INSTRUCTIONS.md` — the grep
  test scans every `.py` file in `src/` including docstrings, so this was
  necessary, not cosmetic.
- **`tests/conftest.py`** — added
  `os.environ.setdefault("AERIAL_DATA_ROOT", "/home/omer/PycharmProjects/Dynamic-Terrain/data")`.
  This is *test* infrastructure (outside `src/`, not scanned by
  `test_no_machine_paths`), analogous to a developer's own shell profile: on
  a fresh machine nothing sets this for you, which is exactly what
  `test_config_fails_loudly_without_data_root` checks (it explicitly
  `monkeypatch.delenv`s the var for the duration of that one test, so the
  conftest default never masks it).
- **New `tests/test_portability.py`** — the 5 acceptance tests.
- **New `INSTRUCTIONS.md` at the repository root** — setup half only, per
  brief. Every command in it was actually run during this stage; §9 (index
  build / query / export) is an explicit, clearly-marked placeholder since
  those stages don't exist yet.

## Acceptance criteria — RED then GREEN

### `test_no_machine_paths`

**RED**, captured against the unmodified codebase (before any retrofit),
confirming the trap actually existed and this test actually detects it:

```
$ env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/omer/anaconda3/envs/geo/bin/python -m pytest tests/test_portability.py::test_no_machine_paths -v
...
>       assert not hits, "machine-specific literal(s) found in src/:\n" + "\n".join(hits)
E       AssertionError: machine-specific literal(s) found in src/:
E         src/calibrate.py:8: '/home/' in '/home/omer/anaconda3/envs/geo/bin/python retrieval/src/calibrate.py'
E         src/calibrate.py:34: '/home/' in 'SRC = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data/X605_Y3388.tif")'
E         src/calibrate.py:8: 'anaconda3' in '/home/omer/anaconda3/envs/geo/bin/python retrieval/src/calibrate.py'
E         src/calibrate.py:34: 'Dynamic-Terrain' in 'SRC = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data/X605_Y3388.tif")'
E         src/embedders.py:16: '/home/' in '/home/omer/anaconda3/envs/geo/bin/python ...'
E         src/embedders.py:16: 'anaconda3' in '/home/omer/anaconda3/envs/geo/bin/python ...'
E         src/geo.py:26: '/home/' in 'env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python ...'
E         src/geo.py:26: 'anaconda3' in 'env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python ...'
E         src/inventory.py:53: '/home/' in 'env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python \\\\'
E         src/inventory.py:78: '/home/' in 'DATA_ROOT = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data")'
E         src/inventory.py:53: 'anaconda3' in 'env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python \\\\'
E         src/inventory.py:78: 'Dynamic-Terrain' in 'DATA_ROOT = Path("/home/omer/PycharmProjects/Dynamic-Terrain/data")'
FAILED tests/test_portability.py::test_no_machine_paths - AssertionError: mac...
1 failed in 1.16s
```

(One extra RED cycle along the way, not the initial one: my first attempt at
`config.py`'s error message included a PowerShell example
`'C:\path\to\data'`, which the grep correctly flagged since `C:\` is exactly
the literal the test watches for — even inside a helpful error string.
Fixed by using `'C:/path/to/data'` in the example, which PowerShell and
Windows both accept.)

**GREEN**, after the full retrofit:

```
tests/test_portability.py::test_no_machine_paths PASSED
```//confirmed as part of the full-suite run below.

### `test_dtype_for_device`

No meaningful RED to paste beyond "the module does not exist yet"
(`src/device.py` did not exist before this stage, so this test could not
even be collected). **GREEN**, all three branches on fabricated capability,
plus MPS for good measure:

```
$ env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/omer/anaconda3/envs/geo/bin/python -m pytest tests/test_portability.py -v
tests/test_portability.py::test_dtype_for_device PASSED
```

Directly asserts (fabricated capability, no matching hardware required):
`select_dtype(cuda, capability=(7,5)) == float16`,
`select_dtype(cuda, capability=(8,6)) == bfloat16`,
`select_dtype(cpu) == float32`, `select_dtype(mps) == float32`.

### `test_cpu_only_build`

Same story — `device="cpu"` support did not exist in `load_embedder` before
this stage (it required CUDA unconditionally). **GREEN**, an actual
CPU-forced run, not inspection:

```
tests/test_portability.py::test_cpu_only_build
CPU-only throughput (RemoteCLIP-ViT-L-14, 6 tiles @224px): 0.97s = 6.15 tiles/sec (reference: 265 tiles/sec on GPU 0)
PASSED
```

A separate, larger, warm-up-excluded measurement for the throughput figure
quoted in `INSTRUCTIONS.md` (32 tiles, single-threaded-vs-multi handled by
torch's own default CPU thread pool, no tuning applied):

```
$ env PYTHONNOUSERSITE=1 AERIAL_DATA_ROOT=/home/omer/PycharmProjects/Dynamic-Terrain/data /home/omer/anaconda3/envs/geo/bin/python - <<'EOF'
... (load_embedder("RemoteCLIP-ViT-L-14", device="cpu"), 4-tile warm-up excluded, then 32 tiles)
EOF
device: cpu dtype: torch.float32
32 tiles in 6.46s = 4.95 tiles/sec (CPU, RemoteCLIP-ViT-L-14, 224px)
```

**Measured CPU-only throughput: ~5 tiles/sec**, against the reference 265
tiles/sec on GPU 0 (Quadro RTX 6000) recorded in `CLAUDE.md` — roughly 50x
slower. Both figures are in `INSTRUCTIONS.md` §4.

### `test_manifest_path_portable`

**GREEN** (new module, no meaningful RED beyond non-existence):

```
tests/test_portability.py::test_manifest_path_portable PASSED
```

Asserts `config.posix_key("leb\\2022-10-29.tif") == config.posix_key("leb/2022-10-29.tif") == "leb/2022-10-29.tif"`,
a `PureWindowsPath` input normalises the same way, and a value round-tripped
through an actual written-and-reread JSON file keeps the canonical form.
No real Windows machine needed — `pathlib.PureWindowsPath` parses
Windows-style strings on any host OS.

Also fixed the underlying bug this test protects against:
`inventory.py`'s per-raster `"path"` field used to be `str(p.relative_to(root))`,
which is OS-native-separator on whichever OS runs it — i.e., would emit
backslash-separated keys if ever run on Windows, silently breaking any
`startswith("leb/")`-style prefix check downstream (`indexable_paths()`,
`source_imagery_paths()`). Now `config.posix_key(...)`. Verified
byte-identical to the old output on this Linux machine (see "no behaviour
change" below).

### `test_config_fails_loudly_without_data_root`

**GREEN** (new module):

```
tests/test_portability.py::test_config_fails_loudly_without_data_root PASSED
```

`monkeypatch.delenv("AERIAL_DATA_ROOT")` then asserts `config.get_data_root()`
raises `config.ConfigError` whose message names `AERIAL_DATA_ROOT`, shows
`PowerShell` syntax, and contains neither `Dynamic-Terrain` nor
`/home/omer` — i.e., it never falls back to this machine's own path.
Manually re-verified outside pytest too (both `config.get_data_root()`
directly and `import inventory` with the env var absent):

```
$ env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python -c "
import config
try:
    config.get_data_root()
except config.ConfigError as e:
    print('RAISED OK:', e)
"
RAISED OK: AERIAL_DATA_ROOT is not set. Point it at the folder that contains the source imagery (the .tif/.tiff rasters) before running anything under retrieval/src, e.g.
  bash/zsh   : export AERIAL_DATA_ROOT=/path/to/data
  PowerShell : $env:AERIAL_DATA_ROOT = 'C:/path/to/data'
See INSTRUCTIONS.md.

$ env PYTHONNOUSERSITE=1 /home/omer/anaconda3/envs/geo/bin/python -c "import sys; sys.path.insert(0,'.'); import inventory"
Traceback (most recent call last):
  ...
  File ".../src/inventory.py", line 81, in <module>
    DATA_ROOT = config.get_data_root()
config.ConfigError: AERIAL_DATA_ROOT is not set. ...
```

## Full final test run — 39 existing + 5 new = 44, all green

Run twice (once right after the retrofit, once again as a final check before
handback) to confirm determinism — both runs: 44 passed, same wall time
class, no flakes.

```
$ env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/omer/anaconda3/envs/geo/bin/python -m pytest tests -v
collected 44 items
tests/test_embedders.py::test_dtype_and_cuda[PE-Core-G14-448] PASSED
tests/test_embedders.py::test_embed_unit_norm[PE-Core-G14-448] PASSED
tests/test_embedders.py::test_text_embed_deterministic[PE-Core-G14-448] PASSED
tests/test_embedders.py::test_self_similarity[PE-Core-G14-448] PASSED
tests/test_embedders.py::test_dtype_and_cuda[PE-Core-L14-336] PASSED
tests/test_embedders.py::test_embed_unit_norm[PE-Core-L14-336] PASSED
tests/test_embedders.py::test_text_embed_deterministic[PE-Core-L14-336] PASSED
tests/test_embedders.py::test_self_similarity[PE-Core-L14-336] PASSED
tests/test_embedders.py::test_dtype_and_cuda[RemoteCLIP-ViT-L-14] PASSED
tests/test_embedders.py::test_embed_unit_norm[RemoteCLIP-ViT-L-14] PASSED
tests/test_embedders.py::test_text_embed_deterministic[RemoteCLIP-ViT-L-14] PASSED
tests/test_embedders.py::test_self_similarity[RemoteCLIP-ViT-L-14] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3857-mercator] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[3395-mercator] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32636-projected] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[32639-projected] PASSED
tests/test_geo.py::test_crs_kind_is_derived_from_the_crs[4326-geographic] PASSED
tests/test_geo.py::test_crs_kind_of_none_is_none PASSED
tests/test_geo.py::test_tile_ground_extent_is_gsd_times_pixels PASSED
tests/test_geo.py::test_gsd_correction PASSED
tests/test_geo.py::test_mercator_correction_agrees_with_an_independent_geodesic_measurement PASSED
tests/test_geo.py::test_gsd_no_correction_for_utm PASSED
tests/test_geo.py::test_utm_no_correction_agrees_with_an_independent_geodesic_measurement PASSED
tests/test_geo.py::test_the_two_regimes_frame_the_same_ground_area_within_5_percent PASSED
tests/test_geo.py::test_ground_resolution_geographic PASSED
tests/test_geo.py::test_gsd_guard_fires_on_secant_mercator PASSED
tests/test_geo.py::test_gsd_guard_does_not_fire_on_the_eight_indexable_scenes PASSED
tests/test_inventory.py::test_source_classifier PASSED
tests/test_inventory.py::test_d1_pixel_rule_over_leb_finds_more_than_the_two_scenes PASSED
tests/test_inventory.py::test_leb_denominator_is_reported_as_a_fact PASSED
tests/test_inventory.py::test_classification_is_by_band_count_and_distinct_levels_only PASSED
tests/test_inventory.py::test_source_inventory_matches_manifest PASSED
tests/test_inventory.py::test_summary_line_states_counts_per_reason PASSED
tests/test_inventory.py::test_adversarial_dedup_grid_pair_is_genuinely_adversarial PASSED
tests/test_inventory.py::test_adversarial_full_sample_grid_pair_is_genuinely_adversarial PASSED
tests/test_inventory.py::test_same_pixels_rejects_adversarial_agreement PASSED
tests/test_inventory.py::test_data_dir_readonly PASSED
tests/test_inventory.py::test_inventory_deterministic PASSED
tests/test_inventory.py::test_the_pixel_rule_has_a_wide_margin PASSED
tests/test_portability.py::test_no_machine_paths PASSED
tests/test_portability.py::test_dtype_for_device PASSED
tests/test_portability.py::test_cpu_only_build PASSED
tests/test_portability.py::test_manifest_path_portable PASSED
tests/test_portability.py::test_config_fails_loudly_without_data_root PASSED
44 passed, 1 warning in 113.48s (0:01:53)
```

`test_dtype_and_cuda` (N-3) for all three candidates still asserts and
confirms `torch.float16` on this cc-7.5 machine — unchanged by the
retrofit, as required.

## Measured CPU-only throughput (for `INSTRUCTIONS.md`)

**~5 tiles/sec** (RemoteCLIP-ViT-L-14, 224 px tiles, 32-tile batch, 4-tile
warm-up excluded, single CPU, default torch thread count) versus the
reference **265 tiles/sec on GPU 0** (Quadro RTX 6000, from `CLAUDE.md`) —
roughly 50x slower. Recorded in `INSTRUCTIONS.md` §4 so a CPU-only reader
can plan wall time honestly rather than discovering the ratio the hard way.

## Deviations / unspecified decisions

- **Env var names**: brief gave `AERIAL_DATA_ROOT`/`AERIAL_INDEX_ROOT` as
  examples ("…"); I named the third `AERIAL_MODEL_CACHE_ROOT` and added
  `AERIAL_DEVICE` (not explicitly named in the brief, but required to make
  "forcing device cpu" configurable via environment as well as by parameter,
  matching the pattern of the other two). All four are documented together
  in `config.py`'s module docstring and in `INSTRUCTIONS.md` §6.
- **`load_embedder`'s new `device` parameter**: the brief didn't specify the
  exact call shape for forcing CPU. I chose `load_embedder(candidate_id,
  device=None)` with `None` preserving the exact old behaviour (CUDA
  required, same error) and an explicit value or `AERIAL_DEVICE` opting into
  the new path — this was the smallest change that satisfies "forcing device
  cpu completes an embed and a query" while leaving every existing call site
  (`calibrate.py`, `test_embedders.py`) untouched and behaviourally identical.
- **`tests/conftest.py`'s `AERIAL_DATA_ROOT` default**: not mentioned in the
  brief. Without it, every one of the 39 pre-existing tests (which hardcode
  the real data root themselves, e.g. `test_geo.py`, `test_inventory.py`)
  would still pass on their own terms, but `inventory.py`'s new
  `DATA_ROOT = config.get_data_root()` module-level line would raise
  `ConfigError` at *import* time the instant any test imports `inventory`
  without the env var set — which would break collection of
  `test_inventory.py` entirely. Setting a `setdefault` in `conftest.py` (test
  infrastructure, outside `src/`, not scanned by the grep test) was the
  option that satisfies both "fail loudly with no default in `src/`" and
  "don't break existing tests." Flagging this as a decision rather than
  silently doing it, since it's the one place a hardcoded path still exists
  in the tree — deliberately confined to test setup.
- **`tree_snapshot()`'s keys** were left as `str(...)` rather than also
  switched to `posix_key`, since they never cross an OS boundary (used only
  within one process's own before/after read-only check) — see "What
  changed" above. Flagging in case the PM wants uniformity anyway; I judged
  it out of scope for this stage since it isn't a manifest key and touching
  it adds risk for no portability benefit.
- **PowerShell example path syntax**: `config.py`'s error message uses
  `'C:/path/to/data'` (forward slashes) rather than the more familiar
  `'C:\path\to\data'`, specifically so the example itself doesn't trip
  `test_no_machine_paths`'s literal `C:\` check. Windows and PowerShell both
  accept forward slashes in paths, so this loses no correctness, but it's a
  slightly unusual-looking example — flagging in case a human reviewer
  wants it phrased differently later (e.g. escaping the backslash in a way
  that's readable but doesn't literally contain `C:\`).

## Blockers

None. Specifically, neither of the brief's named blocker conditions
occurred: `select_dtype` maps this machine's real cc 7.5 to `float16`
(verified both directly in `test_dtype_for_device` and indirectly via N-3's
`test_dtype_and_cuda` staying green on the real hardware), and no retrofit
changed any existing test's result — all 39 pre-existing tests pass
unchanged, run twice for determinism.
