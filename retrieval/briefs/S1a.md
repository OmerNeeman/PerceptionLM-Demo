# Brief — S1a: config + device abstraction, and the setup half of INSTRUCTIONS.md

**Dispatch:** subagent. **Proves:** spec **N-8**, and part of **N-9**.
**Depends on:** S1 (green). **Blocks:** S2, S3, S4, S5.

## Goal

Make the system runnable on a machine that is **not this one** — Linux or
Windows, any CUDA GPU generation, or no GPU at all — and write the half of
`INSTRUCTIONS.md` that is true as soon as that lands.

## Why this stage exists, and why now rather than at the end

The owner asked for instructions a fresh session could follow on any computer.
An audit found that is **not a documentation problem**:

| Location | Hardcoded |
|---|---|
| `src/embedders.py:35` | `float16` |
| `src/inventory.py` | the data root `/home/omer/PycharmProjects/Dynamic-Terrain/data` |
| `src/calibrate.py` | the demo scene's absolute path |
| several docstrings | `/home/omer/anaconda3/envs/geo/bin/python` |

No document can make that run elsewhere. And the dtype case is sharper than a
path: **`CLAUDE.md` trap 2 says "fp16, never bf16" and it is right — for
*these* cards.** They are Quadro RTX 6000 (Turing, cc 7.5), where bf16 is
emulated at 7.0 TFLOP/s against 38.9 for fp16. On **Ampere and later bf16 is
native and the better choice**, and on CPU fp16 is unsupported or
pathologically slow. **A portable implementation must reach the opposite
conclusion on the owner's next GPU.**

This runs **before S2** because S2, S3, S4 and S5 would each copy the pattern.
Fixing two modules now is strictly cheaper than retrofitting five later.

## Inputs — the whole reading list

- `CLAUDE.md` — environment, traps, standards.
- `spec.md` items **N-8** and **N-9** only. They are written as checkable
  assertions; they are your acceptance criteria.
- `src/embedders.py`, `src/inventory.py`, `src/geo.py`, `src/calibrate.py` —
  what you are retrofitting.

Do not read `plan.md`, `notes.md`, or other briefs.

## Outputs

- `src/config.py` — paths and settings, env-overridable.
- `src/device.py` — device selection and the dtype rule.
- Retrofits of `src/embedders.py`, `src/inventory.py`, `src/calibrate.py`
  (and `src/geo.py` if it carries any machine path).
- `tests/test_portability.py`.
- **`INSTRUCTIONS.md` at the repository root** — setup half only (see below).

## What must be true

### 1. No machine-specific value in `src/`

- Data root, index root and model-cache location come from `config.py`, each
  overridable by an environment variable (`AERIAL_DATA_ROOT`,
  `AERIAL_INDEX_ROOT`, …). Name them consistently and document them.
- **No default that only exists on one machine.** If the data root is not
  configured, fail with a message naming the env var and how to set it on both
  Linux and Windows — do not silently fall back to this machine's path.
- Index root defaults sensibly **relative to the repository**, since that is
  machine-independent.
- A specific interpreter path may appear in *documentation*. It may not appear
  in importable code.

### 2. dtype is a pure function of device capability

This is the heart of the stage. A function — call it `select_dtype(device)` —
mapping:

| device | dtype | why |
|---|---|---|
| CUDA, compute capability **< 8.0** | `float16` | Turing/Volta: bf16 emulated, 5.6x slower than fp16 and slower than fp32 |
| CUDA, compute capability **>= 8.0** | `bfloat16` permitted | Ampere and later: native bf16 |
| CPU or MPS | `float32` | fp16 on CPU is unsupported or pathologically slow |

**Test the function directly at all three cases**, not just the one this
machine happens to be. Fabricate the capability input — do not require an
Ampere card to test the Ampere branch.

**This machine is cc 7.5, so it must still resolve to `float16`.** The existing
N-3 test asserting the loaded model dtype is `torch.float16` must stay green.
The current behaviour becomes a *case* of the rule, not the rule.

### 3. It runs with no GPU

Forcing device `cpu` completes an embed of a small tile set and a query.
Assert with an actual CPU-forced run, not by inspection. Record the observed
throughput so `INSTRUCTIONS.md` can state honest expectations — for reference,
RemoteCLIP does 265 tiles/sec on GPU 0 here.

### 4. Paths are OS-agnostic

`pathlib` throughout; no hand-built separators, no POSIX-only assumptions.
Tile ids and manifest keys use **one canonical separator** (use `/`) so an
index built on Linux loads on Windows and vice versa. Test that a manifest
written from Windows-style inputs reads back identically.

## `INSTRUCTIONS.md` — the setup half only

At the **repository root**. Write only what is true once this stage lands;
the run commands come at M1 and verification at S9. Cover:

- **Prerequisites and how to check them** — Python version, CUDA or its
  absence, disk, RAM.
- **Linux and Windows side by side.** Do not write Linux and add a Windows
  footnote. `rasterio`/`GDAL` in particular are the usual Windows failure, so
  say plainly which install route is expected to work (conda-forge) and which
  commonly does not.
- **CUDA and CPU-only**, both as first-class paths.
- **The dtype rule and why it differs by GPU generation** — state the Turing
  and Ampere cases explicitly. A reader with an RTX 4090 must not conclude
  they should force fp16 because this project's notes said so.
- **Where imagery is expected and how to point elsewhere** — the env vars, with
  syntax for `bash` and for PowerShell.
- **Troubleshooting**, for failure modes actually hit here, not imagined ones:
  a shadowed CPU-only torch in user site-packages (`PYTHONNOUSERSITE=1`); bf16
  on pre-Ampere; `pyproj` unable to open its PROJ database; GDAL/rasterio
  install; and the Mercator GSD correction, which silently yields
  ground distances 19.4% too large.

Write it for someone who has never seen this project and cannot ask a question.
Anything that would force them to ask is a defect.

**Do not describe commands you have not run.** If the run half isn't buildable
yet, say so and leave a clearly-marked placeholder rather than inventing a CLI.

## Acceptance criteria — test-first, paste RED then GREEN

- `test_no_machine_paths` — greps every file in `src/` for `/home/`, `C:\`,
  `anaconda3` and the data-root literal; fails on any hit. **This will be RED
  immediately** against the current code; that is the point.
- `test_dtype_for_device` — all three cases asserted with fabricated
  capability; cc 7.5 -> `float16`, cc 8.6 -> `bfloat16`, cpu -> `float32`.
- `test_cpu_only_build` — a forced-CPU embed and query completes.
- `test_manifest_path_portable` — Windows-style input round-trips identically.
- `test_config_fails_loudly_without_data_root` — a missing data root raises a
  message naming the env var, and does **not** fall back to a hardcoded path.
- **All 39 existing tests stay green**, including N-3's `float16` assertion.

## Constraints

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/omer/anaconda3/envs/geo/bin/python
```

**Your Bash tool does not persist env vars between calls — prefix every python
and pytest invocation inline, every time.**

- **The data directory is READ-ONLY.** All output under `retrieval/index/`,
  except `INSTRUCTIONS.md` at the repo root.
- Do not change any **behaviour** while retrofitting. This stage moves values
  into config and generalises dtype selection; it does not alter what the
  classifier decides, what the geodesy computes, or what the embedder returns.
  Every existing test staying green is how you demonstrate that.
- `pyproj.CRS`/`Transformer` remain broken here; route CRS work through
  rasterio/GDAL.
- Do not commit. The PM handles git.

## Report back

Status, QA checklist with **pasted RED-then-GREEN**, the full test run, the
measured CPU-only throughput, deviations, unspecified decisions, blockers.

**Flag as a blocker rather than working around it if:** making dtype
capability-dependent breaks N-3's `float16` assertion on this machine (it
should not — cc 7.5 maps to fp16), or a retrofit changes any existing test's
result. A behaviour change here is a finding, not a fix.
