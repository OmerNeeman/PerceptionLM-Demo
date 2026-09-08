# INSTRUCTIONS

How to set up the Aerial Tile Retrieval subsystem (`retrieval/`) on a machine
that is not the one it was developed on — Linux or Windows, any CUDA GPU
generation, or no GPU at all.

**Scope of this document right now:** everything below is the *setup* half —
install, environment, and how to verify the setup works. It was written and
actually run on the project's own dev machine as of the commit that added it
(stage S1a). **The *run* half — building a real tile index, querying it, and
producing an export — does not exist yet** (that lands in later stages) and
is deliberately left as a placeholder near the bottom rather than invented.
Nothing in this document describes a command that was not actually executed
while writing it.

---

## 1. Prerequisites, and how to check them

| Requirement | Check |
|---|---|
| Python 3.12 | `python --version` |
| ~2 GB free disk for a conda env + model weights, more for any imagery you index | `df -h .` (Linux/macOS) / `Get-PSDrive` (PowerShell) |
| RAM: 16 GB is comfortable; large single rasters (this project has seen 6+ gigapixel TIFFs) are read in small windows, never in full, so working-set RAM stays modest regardless of file size | — |
| CUDA GPU (optional) | `nvidia-smi` — if this fails or is absent, you have no GPU and the system runs CPU-only (see §4). If it succeeds, note the GPU name; you will need it in §5 to know which dtype rule branch applies to you |

A conda (or mamba) environment is strongly recommended over plain `venv` for
one reason: `rasterio`/GDAL. See §2.

---

## 2. Install — Linux and Windows side by side

Do this from a fresh conda environment on both platforms.

```bash
conda create -n aerial-retrieval python=3.12 -y
conda activate aerial-retrieval
```

(PowerShell: `conda activate` works identically once conda is initialized
for PowerShell — `conda init powershell` once, then reopen the shell.)

### rasterio / GDAL — install this from conda-forge, not pip

This is the single most common failure point on Windows. GDAL has a large
native dependency tree (PROJ, GEOS, a dozen format drivers); conda-forge
ships prebuilt binaries for it on both platforms, pip's wheels do not
reliably work the same way on Windows.

**Expected to work, both platforms:**

```bash
conda install -c conda-forge rasterio gdal -y
```

**Commonly fails on Windows:** `pip install rasterio` / `pip install gdal`
alone — missing or mismatched native GDAL libraries, DLL load failures at
import time. If you must use pip, install GDAL's own wheel index first and
pin `rasterio` to a version built against the same GDAL — but conda-forge is
the path this project actually expects to work; nothing else has been tried.

### PyTorch — CUDA build vs CPU build

**With an NVIDIA GPU** (any generation — see §5 for what changes per
generation): install the CUDA build matching a driver you have. Pick the
CUDA version from [pytorch.org](https://pytorch.org)'s install matrix, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**With no GPU, or on a machine you're only using for CPU-only development:**

```bash
pip install torch
```

(the default PyPI wheel is CPU-only on Linux/Windows unless you asked for a
CUDA index URL above).

**The trap this project hit on its own dev machine (see §6):** a CPU-only
`torch` installed into a *user* site-packages directory silently shadows a
correctly-installed CUDA build in a conda env, on Linux. Verify with the
check in §4 before assuming a GPU install "didn't take."

### The rest

```bash
pip install open_clip_torch==3.3.0 pytest transformers huggingface_hub \
    opencv-python-headless scikit-learn pillow numpy einops
```

`open_clip_torch` 3.3.0 pulls `torch`, `torchvision`, `regex`, `ftfy`, `tqdm`,
`huggingface-hub`, `safetensors` and `timm>=1.0.17` transitively, so those need
no separate line. **`einops` does not come with anything above** and is listed
explicitly because the optional `perception_models` fallback imports it — see
below.

Do **not** install `faiss` — this project's index sizes (thousands of tiles
× 768-d) do exact top-k in numpy in well under a millisecond; faiss is
unnecessary weight, not a missing dependency.

### The fallback embedder's own package (`perception_models`)

Only needed if you ever revert off the chosen embedder (RemoteCLIP). If you
do:

```bash
git clone https://github.com/facebookresearch/perception_models
pip install -e ./perception_models --no-deps
```

**`--no-deps` is mandatory.** Its own `requirements.txt` pins older
`numpy`/`pillow`/`timm`/`scikit-learn`/`opencv-python` than the versions
above and can silently replace a CUDA `torch` build with a CPU-only one. Its
real imports are `numpy`, `torch`, `torchvision`, `einops`, `timm`,
`huggingface_hub`, `ftfy` and `regex`. All are satisfied by the install above —
`einops` only because it is named explicitly there; the rest arrive
transitively with `open_clip_torch`. Verified against `open_clip_torch`
3.3.0's declared requirements, not assumed.

Note also that `perception_models` pins `timm==1.0.15` while `open_clip_torch`
3.3.0 requires `timm>=1.0.17`. `--no-deps` is what keeps those from fighting;
it is not merely a tidiness measure.

---

## 3. Verify the install

From `retrieval/`, with your environment active:

```bash
python -c "import rasterio, torch, open_clip; print(rasterio.__version__, torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` should read `True` if you have a GPU and
followed the CUDA install above, `False` if you deliberately installed
CPU-only. If it reads `False` on a machine you *know* has a GPU, see the
shadowed-torch entry in §6 before concluding the GPU install failed.

---

## 4. CUDA and CPU-only, both as first-class paths

The embedder loader (`retrieval/src/embedders.py`) autodetects CUDA by
default and requires it unless you say otherwise — because on this
project's own dev machine, `torch.cuda.is_available() is False` has always
meant a shadowed CPU-only `torch` (§6), not "there is no GPU," and failing
loudly on that surprises nobody who actually has a GPU.

**On a machine that genuinely has no GPU**, force CPU explicitly — either:

```bash
export AERIAL_DEVICE=cpu        # bash/zsh
$env:AERIAL_DEVICE = "cpu"      # PowerShell
```

or pass `device="cpu"` directly to `load_embedder(...)` in code. This has
been run for real on this project's own (GPU-equipped, but CPU-forced)
machine: a small batch of 224 px tiles embedded end-to-end on CPU, and a
text query embedded alongside them, both completed. **Measured throughput:
~5 tiles/sec** (32 tiles, RemoteCLIP-ViT-L-14, warm-up excluded) — against
**265 tiles/sec on this project's GPU 0** (a Quadro RTX 6000). Expect CPU-only
embedding to be on the order of 50x slower than a mid-range CUDA GPU; plan
index-build wall time accordingly, and prefer a GPU for anything beyond a
small demo AOI if one is available at all, even an older-generation one.

MPS (Apple Silicon) follows the same `AERIAL_DEVICE=mps` path but has not
been exercised on real Apple hardware by this project — treat it as
untested, not unsupported.

---

## 5. The dtype rule — and why it is *not* "always fp16"

This project's own dev notes say, correctly, "use fp16, never bf16." That
rule is right **only for the specific GPUs this project was developed on**
(Quadro RTX 6000, NVIDIA Turing architecture, compute capability 7.5), and
is wrong as a universal statement. Do not copy "always fp16" onto different
hardware.

The actual rule (`retrieval/src/device.py:select_dtype`), by device and GPU
generation:

| Device | Compute capability | dtype used | Why |
|---|---|---|---|
| CUDA | **< 8.0** (Turing, Volta — e.g. RTX 20-series, Quadro RTX 6000, V100) | `float16` | bf16 is *emulated* in software on these cards: measured 7.0 TFLOP/s vs 38.9 TFLOP/s for fp16 on this project's hardware — 5.6x slower, and slower than plain fp32 |
| CUDA | **>= 8.0** (Ampere and later — e.g. RTX 30/40-series, A100, H100) | `bfloat16` | native hardware bf16 on this generation and later; the better choice |
| CPU or MPS | — | `float32` | fp16 on CPU is unsupported by most kernels or pathologically slow where it exists at all |

**If you have an RTX 4090, an A100, an H100, or anything else Ampere or
newer: you should be on bf16, not fp16.** The dtype is selected
automatically by `select_dtype()` from `torch.cuda.get_device_capability()`
— you do not need to set anything for this. It is documented here so that
nobody reads this project's own trap notes (written for Turing hardware) and
force-configures fp16 on a GPU where that is the wrong answer.

---

## 6. Where imagery is expected, and how to point elsewhere

The system never assumes imagery lives at a particular path. Three
environment variables, all read by `retrieval/src/config.py`:

| Variable | Required? | Default | Meaning |
|---|---|---|---|
| `AERIAL_DATA_ROOT` | **Yes** | none — fails loudly if unset | Folder containing the source imagery (the raw `.tif`/`.tiff` rasters). Read-only; nothing under this path is ever written to. |
| `AERIAL_INDEX_ROOT` | No | `<repo>/retrieval/index` | Where derived artifacts (inventory, tiles, embeddings, exports) are written. The default is relative to the repository, so it needs no per-machine configuration. |
| `AERIAL_MODEL_CACHE_ROOT` | No | `huggingface_hub`'s own default (`HF_HOME`, or the platform cache dir) | Where downloaded model checkpoints are cached, if you want them somewhere specific. |

Set at minimum `AERIAL_DATA_ROOT`:

```bash
# bash/zsh
export AERIAL_DATA_ROOT=/path/to/your/imagery
```

```powershell
# PowerShell
$env:AERIAL_DATA_ROOT = "C:/path/to/your/imagery"
```

If it is not set, every entry point fails immediately with a message naming
the variable and showing both syntaxes above — it never silently falls back
to a path from this project's own development machine.

---

## 7. Run the acceptance tests

This is the real, run-today way to confirm your setup works — not a stand-in
for building an index (§9), just proof the environment is sound:

```bash
export AERIAL_DATA_ROOT=/path/to/your/imagery   # PowerShell: $env:AERIAL_DATA_ROOT = "..."
cd retrieval
python -m pytest tests/test_portability.py -v
```

All five tests should pass regardless of platform or GPU generation — that
is the entire point of this stage. On this project's own dev machine, the
**full** suite (portability tests plus every earlier stage's tests) is 44
tests and took **113 s**; the portability tests alone took under 10 s.
Running the full suite requires actual GPU access to this project's chosen
and fallback embedders and will need real imagery under `AERIAL_DATA_ROOT`
matching this project's own dev dataset — it is not expected to pass
unmodified against arbitrary imagery. `test_portability.py` alone has no
such dependency beyond a small amount of scratch disk and, for
`test_cpu_only_build`, downloading the RemoteCLIP checkpoint once.

---

## 8. Troubleshooting

**`torch.cuda.is_available()` is `False` on a machine that has a GPU.**
Almost always a CPU-only `torch` in a *user* site-packages directory
(`~/.local/lib/pythonX.Y/site-packages` on Linux) shadowing the correctly
installed CUDA build in your active environment — this is exactly what
happened during this project's own development. Fix:

```bash
export PYTHONNOUSERSITE=1   # bash/zsh, before running python
```

```powershell
$env:PYTHONNOUSERSITE = "1"   # PowerShell
```

Confirm with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
— a `+cpu` build number is the tell.

**bf16 looks "supported" (`torch.cuda.is_bf16_supported()` returns `True`)
but training/inference is unexpectedly slow.** `is_bf16_supported()` answers
"can the hardware execute a bf16 instruction at all," not "does it do so
natively." On pre-Ampere cards (compute capability < 8.0) bf16 is emulated
in software — see §5. This is not a bug to work around; it is the reason
`select_dtype()` picks fp16 on that hardware.

**`pyproj` warns "unable to set PROJ database path" at import, and
`pyproj.CRS(...)` / `pyproj.Transformer.from_crs(...)` raise.** Known,
environment-specific: `pyproj`'s bundled PROJ database is not reliably found
in every install. This project's own code never depends on `pyproj.CRS` or
`pyproj.Transformer` for exactly this reason — all CRS classification and
reprojection goes through `rasterio`/GDAL instead (`retrieval/src/geo.py`),
which does not hit this problem. `pyproj.Geod` (pure ellipsoidal geodesy, no
database needed) is used for distance measurement and works regardless. If
you write new code, follow the same split: CRS work through rasterio/GDAL,
distance work through `pyproj.Geod`.

**GDAL/rasterio import fails, or fails only on Windows.** See §2 — install
both from `conda-forge`, not pip, in a conda/mamba environment. A pip-only
install is the single most common way to reach this failure on Windows.

**A ground-distance figure looks ~19% too large (or too small).** This is
the Mercator GSD trap. A raster's *projected* pixel size (from its
geotransform) is not automatically a *true ground* distance: in a Mercator
projection (EPSG:3857 "Pseudo-Mercator," EPSG:3395 "World Mercator"), the
projection inflates distances by `1 / cos(latitude)` — at this project's own
demo latitude (33.1°N) that is a **19.4%** overstatement if left
uncorrected. The correction is *conditional on the CRS actually being
Mercator* — a UTM scene (e.g. EPSG:32636) is already within 0.04% of unity
and applying the cosine correction there is exactly as wrong as omitting it
for a Mercator scene. `retrieval/src/geo.py` derives the correct regime from
the CRS itself (never a filename or a per-file table) and applies it — or
doesn't — accordingly. If you compute a ground distance yourself from a raw
geotransform anywhere outside that module, re-derive it from `geo.py`
instead of re-deriving the correction by hand.

---

## 9. Building an index, querying it, and exporting

Tiling, embedding and indexing (§§F-1–F-2a) are earlier stages and already
ran once against the real imagery; their output lives under
`retrieval/index/` (gitignored, never committed) and is **not rebuilt** by
anything below. The standalone-HTML exporter (F-8) is a separate, later
stage (S5b) and does not exist yet. What *does* exist as of this stage (S5)
is the local query app: type a description, get back ranked, located tiles
with a confidence band and a resolution caveat.

### Launch the local app

```bash
export PYTHONNOUSERSITE=1          # mandatory -- see §8/trap 1
export CUDA_VISIBLE_DEVICES=0      # GPU 1 drives the desktop on the dev machine
export AERIAL_DATA_ROOT=/path/to/your/imagery
cd retrieval/src
python app.py
```

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:AERIAL_DATA_ROOT = "C:/path/to/your/imagery"
cd retrieval/src
python app.py
```

This starts a local web server at `http://127.0.0.1:8420/` — open that URL
in a browser. Everything is inline (no external stylesheet/script, no CDN,
no network call at runtime beyond the page talking to its own local
server), so it works with no internet connection.

Equivalently, via uvicorn's own CLI — note the `--factory` flag: the app
target is a function that *builds* the app (so importing the module never
forces a model load as a side effect), not a ready-made app object, so a
bare `app:app` target will not work here:

```bash
python -m uvicorn app:create_app --factory --host 127.0.0.1 --port 8420
```

**What loads when.** The corpus (vectors + tile locations) loads at process
start and is fast (< 5 s, no GPU — F-5's own budget). The embedding model
loads lazily, on the *first* query — that call takes about **9 s** longer
than every query after it, once per process, and the page says so
("first search can take ~10s while the model loads"). This is
`open_clip`'s `RemoteCLIP-ViT-L-14`; you may see it print
`No pretrained weights loaded ... initialized randomly` in the *terminal*
running the model-download/build stages elsewhere in this project — the
app's own console output suppresses that specific line during its own load,
because on first read it looks like a failure and is not one (a random
skeleton is built before the real RemoteCLIP checkpoint loads over it).

**Which AOI.** The app serves whichever AOI's index it finds under
`retrieval/index/emb/` (default `X605_Y3388`, the project's demo AOI — see
`docs/DATA.md`). Only presence/coarse-class queries are meaningful on it;
damage vocabulary (rubble, collapsed roof) has no support on this AOI by
design (`CLAUDE.md`'s ratified problem statement) — the app does not offer
those as examples for that reason, not by omission.

**Measured, this machine:** cold process start to a fully warmed, answered
first query: corpus load ~0.3 s + model load ~9 s + first search
< 1 ms (N-1's own budget, unaffected by the model). Every subsequent query
in the same process: well under 300 ms end-to-end.

### Tests

```bash
export AERIAL_DATA_ROOT=/path/to/your/imagery
cd retrieval
python -m pytest tests/test_app.py -v
```

Uses a small synthetic index built under `tmp_path` with the real embedder
(mirrors every other stage's own test fixtures) — it does not touch the
shipped `index/` and does not require the demo AOI's real imagery.
