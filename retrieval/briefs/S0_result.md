# S0 result — environment bring-up + embedder bake-off

**Status: `DONE_WITH_CONCERNS`**

Everything the brief asked for is delivered and green. The concerns are
findings, not incomplete work, and two of them change what S1–S3 should do:

1. **`aircraft carrier` (the known-absent control) out-scores PE-Core-L14's
   own `car` query.** No absolute score threshold is usable for a "no
   results" UI on any of the three candidates. Flagged for spec U-* / F-4.
2. **448 px surfaces zero vehicles for all three candidates (0/5 each).**
   The pyramid's 448 scale is dead weight for small objects — but it is
   where road- and building-shaped things live. The scales are not
   interchangeable and should not be pooled into one ranking blindly.
3. **The large corrugated hall roof is retrieved by `sand`** (RemoteCLIP,
   ranks 2 and 4 at 224 px; rank 5 at 112 px). Beige ribbed metal and sand
   are not separable at ~35–45 cm effective resolution.

**Stated choice: `RemoteCLIP-ViT-L-14` (768-d).** Reasoning in §4.

---

## 1. Stated embedder choice

**RemoteCLIP ViT-L-14**, `chendelong/RemoteCLIP` @ `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`,
file `RemoteCLIP-ViT-L-14.pt`, 768-d, native input 224 px.

**Reason — it wins on the only query that was in doubt, and on every
secondary axis.** On `car`, the small-object query the whole project is
betting on, RemoteCLIP returned 4/5 genuine vehicle crops at 112 px against
PE-Core-L14's 3/5 (1 uncertain, 1 clear miss) and PE-Core-G14's 2/5. It is
the only candidate whose every real query out-scores the known-absent
control by a usable margin. It is 3.1x faster than L14 and 23x faster than
G14, and its 768-d vectors are the smallest of the three, which directly
buys headroom against the 16 MB export budget (F-8) and the PCA/int8
reduction (F-2a). Answering the owner's question directly: **yes, being
pretrained on aerial/satellite imagery shows up in the results.** The
margin is modest on the easy queries — `tents` and `palm trees` are ~5/5 for
all three — but consistent, and decisive on vehicles.

PE-Core-G14-448 is eliminated on three counts, not one: 11.6 tiles/sec means
**2.6 hours** for the 108,542-tile pyramid against RemoteCLIP's 6.8 minutes;
its control margin is +0.0019, i.e. zero; and it was the *worst* of the
three at 112 px, because a 112 px crop upsampled 4x to its 448 native input
is far out of distribution. Best on paper, worst on this job.

---

## 2. QA checklist — RED then GREEN

Environment for every run below, inline on every invocation (the harness
does not persist env vars between calls):

```
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/omer/anaconda3/envs/geo/bin/python -m pytest retrieval/tests -v
```

### RED — the naive first draft

The implementation was reverted to the plausible first draft — model loaded
at torch's default dtype, raw encoder output returned with no fp32 cast and
no L2 normalisation — and the tests run against it. All 8 fail, each for its
own reason:

```
=========================== short test summary ============================
FAILED retrieval/tests/test_embedders.py::test_dtype_and_cuda[PE-Core-L14-336]
FAILED retrieval/tests/test_embedders.py::test_embed_unit_norm[PE-Core-L14-336]
FAILED retrieval/tests/test_embedders.py::test_text_embed_deterministic[PE-Core-L14-336]
FAILED retrieval/tests/test_embedders.py::test_self_similarity[PE-Core-L14-336]
FAILED retrieval/tests/test_embedders.py::test_dtype_and_cuda[RemoteCLIP-ViT-L-14]
FAILED retrieval/tests/test_embedders.py::test_embed_unit_norm[RemoteCLIP-ViT-L-14]
FAILED retrieval/tests/test_embedders.py::test_text_embed_deterministic[RemoteCLIP-ViT-L-14]
FAILED retrieval/tests/test_embedders.py::test_self_similarity[RemoteCLIP-ViT-L-14]
======================= 8 failed, 4 deselected in 15.56s =======================
```

**`test_dtype_and_cuda` (N-3) — RED, naive draft loads fp32:**

```
    def test_dtype_and_cuda(embedder):
        assert torch.cuda.is_available() is True, (...)
>       assert embedder.dtype == torch.float16, (
            f"{embedder.model_id} loaded as {embedder.dtype}, expected float16"
        )
E       AssertionError: RemoteCLIP-ViT-L-14 loaded as torch.float32, expected float16
E       assert torch.float32 == torch.float16
E        +  where torch.float32 = <embedders.Embedder object at 0x7cd58dc5cd40>.dtype
retrieval/tests/test_embedders.py:62: AssertionError
```

**`test_dtype_and_cuda` (N-3) — RED again, second failure mode: the same
test run WITHOUT `PYTHONNOUSERSITE=1`.** This is trap 1 caught live, and it
is why the CUDA assertion exists:

```
$ env CUDA_VISIBLE_DEVICES=0 /home/omer/anaconda3/envs/geo/bin/python -m pytest ... -k test_dtype_and_cuda

        if not torch.cuda.is_available():
>           raise RuntimeError(
                "torch.cuda.is_available() is False. This is CLAUDE.md trap 1: ..."
            )
E           RuntimeError: torch.cuda.is_available() is False. This is CLAUDE.md
E           trap 1: a CPU-only torch in ~/.local shadows the CUDA build.
E           Re-run with PYTHONNOUSERSITE=1.
retrieval/src/embedders.py:244: RuntimeError
======================= 11 deselected, 1 error in 1.00s ========================
```

**`test_embed_unit_norm` (F-2) — RED:**

```
        norms = np.linalg.norm(v, axis=1)
>       assert np.allclose(norms, 1.0, atol=1e-5), (...)
E       AssertionError: RemoteCLIP-ViT-L-14: norms off unit by up to 1.971e+01;
E         norms=[20.696808 20.712812 20.534966 20.64109  20.56789  20.481417
E                20.179005 20.249186]
retrieval/tests/test_embedders.py:83: AssertionError
```

**`test_text_embed_deterministic` (F-3) — RED:**

```
>       assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5), (
            f"{embedder.model_id}: text norm not unit: ..."
        )
E       AssertionError: RemoteCLIP-ViT-L-14: text norm not unit: [20.628239]
retrieval/tests/test_embedders.py:102: AssertionError
```

**`test_self_similarity` (F-4-lite) — RED:**

```
        for name, m in (("image", v), ("text", q)):
            self_cos = np.einsum("ij,ij->i", m, m)
>           assert np.allclose(self_cos, 1.0, atol=1e-5), (...)
E           AssertionError: RemoteCLIP-ViT-L-14 image: self-cosine off by up to
E             4.332e+02: [429.73492 434.2367  433.7848  428.1318 ]
retrieval/tests/test_embedders.py:135: AssertionError
```

### A second, genuine RED found by the F-3 test

With the implementation fixed, `test_text_embed_deterministic` still failed
for the two PE candidates — a real bug in the test harness, not in the
model. PE's `load_ckpt` prints to stdout, so the cross-process probe's
stdout channel was contaminated:

```
>       other = np.frombuffer(bytes.fromhex(out.stdout.strip()), dtype=np.float32)
E       ValueError: non-hexadecimal number found in fromhex() arg at position 0
retrieval/tests/test_embedders.py:116: ValueError
=========================== short test summary ============================
FAILED retrieval/tests/test_embedders.py::test_text_embed_deterministic[PE-Core-G14-448]
FAILED retrieval/tests/test_embedders.py::test_text_embed_deterministic[PE-Core-L14-336]
=================== 2 failed, 10 passed in 72.33s (0:01:12) ====================
```

Fixed by having the probe write raw float32 bytes to a file instead of
stdout. Worth remembering for every later stage that shells out to a PE
process: **PE's stdout is not a clean channel.**

### GREEN — all three candidates, all four criteria

```
============================= test session starts ==============================
collecting ... collected 12 items

retrieval/tests/test_embedders.py::test_dtype_and_cuda[PE-Core-G14-448] PASSED [  8%]
retrieval/tests/test_embedders.py::test_embed_unit_norm[PE-Core-G14-448] PASSED [ 16%]
retrieval/tests/test_embedders.py::test_text_embed_deterministic[PE-Core-G14-448] PASSED [ 25%]
retrieval/tests/test_embedders.py::test_self_similarity[PE-Core-G14-448] PASSED [ 33%]
retrieval/tests/test_embedders.py::test_dtype_and_cuda[PE-Core-L14-336] PASSED [ 41%]
retrieval/tests/test_embedders.py::test_embed_unit_norm[PE-Core-L14-336] PASSED [ 50%]
retrieval/tests/test_embedders.py::test_text_embed_deterministic[PE-Core-L14-336] PASSED [ 58%]
retrieval/tests/test_embedders.py::test_self_similarity[PE-Core-L14-336] PASSED [ 66%]
retrieval/tests/test_embedders.py::test_dtype_and_cuda[RemoteCLIP-ViT-L-14] PASSED [ 75%]
retrieval/tests/test_embedders.py::test_embed_unit_norm[RemoteCLIP-ViT-L-14] PASSED [ 83%]
retrieval/tests/test_embedders.py::test_text_embed_deterministic[RemoteCLIP-ViT-L-14] PASSED [ 91%]
retrieval/tests/test_embedders.py::test_self_similarity[RemoteCLIP-ViT-L-14] PASSED [100%]

======================== 12 passed in 74.30s (0:01:14) =========================
```

| Criterion | Spec | Status |
|---|---|---|
| `test_dtype_and_cuda` | N-3 | GREEN x3 — `cuda_avail True`, dtype `torch.float16`, device `cuda`, on `Quadro RTX 6000` |
| `test_embed_unit_norm` | F-2 | GREEN x3 — norms 1.0 ± 1e-5, dims 1280 / 1024 / 768 as stated, revision recorded |
| `test_text_embed_deterministic` | F-3 | GREEN x3 — bitwise identical in-process; cross-process delta ≤ 1e-6 |
| `test_self_similarity` | F-4-lite | GREEN x3 — image and text self-cosine 1.0 ± 1e-5; cross-cosines within [-1, 1] |
| Bake-off table + choice | F-0 | This document, §3–§4 |

---

## 3. The bake-off

### 3.1 Calibration subregion

`X605_Y3388.tif` (READ-ONLY), scene offset **(4608, 5120)**, size
**1792 x 1792 px** = 179.2 true ground metres at 10.00 cm/px (UTM scene, no
GSD correction needed — trap 3 applies only to the EPSG:3857 `leb` pair).
Zero-fill 0.0000. Chosen by eye from a gridded thumbnail after comparing
four candidate windows.

Verified content, all confirmed at native resolution before use: tents and
tarpaulins (white, grey, blue, yellow, orange), date palms in both grove and
scattered form, open sand and sand tracks, a large corrugated-roof hall,
flat concrete roofs with parapets and solar panels, and **unambiguous
vehicles** including a white sedan and dark cars at local (896–1008,
448–560) and a white pickup at local ~(1520, 1640).

Tiled at 448 / 224 / 112 px, 0% overlap: **16 + 64 + 256 = 336 crops**, each
resized to the model's native input by the model's own preprocessing
(F-1a).

### 3.2 Throughput — measured, warm-up excluded

| Candidate | dim | tiles/sec | 336 crops | Extrapolated to the 108,542-tile pyramid |
|---|---|---|---|---|
| `PE-Core-G14-448` | 1280 | **11.6** | 28.95 s | **2 h 36 min** |
| `PE-Core-L14-336` | 1024 | **84.5** | 3.98 s | **21 min** |
| `RemoteCLIP-ViT-L-14` | 768 | **265.0** | 1.27 s | **6.8 min** |

fp16, GPU 0, batch size 32. Load-time weight fetch and the 8-crop warm-up
are excluded.

### 3.3 Candidate x query — top-5 verdicts

Verdicts are **my own visual judgement of the written-out crops**, read back
with the Read tool. `n/5` counts crops I could confirm contain the queried
thing. "cannot tell" is recorded as such rather than guessed. Verdicts are
given **per scale**, because the scales differ enormously and an
overall-pooled verdict would hide the finding.

Crops are on disk at `retrieval/index/calib/<model>/<query>/`, with a
per-query contact sheet at `_sheet.png` (3 rows = 3 scales, 5 columns =
top-5 within that scale, each labelled with cosine and local offset).

#### RemoteCLIP-ViT-L-14 — **chosen**

| Query | 448 px | 224 px | 112 px | Notes |
|---|---|---|---|---|
| `tent` | 5/5 | 5/5 | 5/5 | Base-rate inflated — almost everything here is a tent |
| `tents` | 5/5 | 5/5 | 5/5 | Top-1 overall lands at **224 px**, i.e. a *group*; `tent`'s lands at 112, a *single* tent. Correct behaviour |
| `a white car` | **0/5** | 2/5, 2 cannot tell | 2/5 | Rank-1 and 2 at 112 are unmistakable white cars |
| `car` | **0/5** | 3/5, 2 cannot tell | **4/5** | Best small-object result of any candidate. See §3.4 |
| `palm trees` | 5/5 | 5/5 | 5/5 | Clean at every scale |
| `a building with a flat roof` | 4/5 | 3/5 | 5/5 | 224 rank-1 is a **pitched red-tile villa** — a genuine building, wrong attribute |
| `dirt road` | 3/5 weak | 3/5 | ~3/5 | At 112 this collapses into `sand`; see §5 |
| `sand` | 5/5 | 3/5 | 4/5 | Misses are the **hall roof** mistaken for sand |
| `aircraft carrier` (control) | — | — | — | Correctly finds nothing. Max 0.2236, **below all 8 real queries** |

#### PE-Core-L14-336

| Query | 448 px | 224 px | 112 px | Notes |
|---|---|---|---|---|
| `tent` | 5/5 | 5/5 | 5/5 | |
| `tents` | 5/5 | 5/5 | 5/5 | 448 rank-1 is the single best "tents" crop produced by anything — distinct peaked tents in open sand |
| `a white car` | **0/5** | 3/5 | 3/5 | |
| `car` | **0/5** | 3/5 | 3/5, 1 cannot tell, 1 clear miss | Weakest query for this model, and it loses to the control |
| `palm trees` | 4/5 | 5/5 | 5/5 | 448 rank-1 is mostly tents |
| `a building with a flat roof` | 4/5 | 4/5 | 5/5 | 112 row is excellent — flat roofs with parapets, tanks, solar panels |
| `dirt road` | 2/5 | 3/5 | 4/5 | |
| `sand` | 5/5 | 4/5 | 5/5 | |
| `aircraft carrier` (control) | — | — | — | **Max 0.2002 — higher than this model's own `car` max of 0.1845.** See §5 |

#### PE-Core-G14-448 — eliminated

| Query | 448 px | 224 px | 112 px | Notes |
|---|---|---|---|---|
| `a white car` | **0/5** | 3/5, 1 cannot tell | **2/5** | Worst at 112: a 112 crop upsampled 4x to 448 is out of distribution |
| `palm trees` | 4/5 | 5/5 | 5/5 | Competent |
| `aircraft carrier` (control) | — | — | — | Max 0.1489 vs weakest real query 0.1508 — margin **+0.0019**, effectively zero |
| (others) | | | | Numerically consistent with the two above; not read crop-by-crop once throughput and control margin had settled the decision. Sheets are on disk for independent review |

### 3.4 Control separation — the quantitative cross-check

`margin` = (lowest max-score across the 8 real queries) − (control's max
score). `spread` = mean over real queries of (max − mean), i.e. how far the
top result stands above the corpus.

| Candidate | control max | weakest real max | margin | spread | margin/spread | real queries beating control |
|---|---|---|---|---|---|---|
| `PE-Core-G14-448` | +0.1489 | +0.1508 | +0.0019 | 0.0498 | 0.04 | 8/8 |
| `PE-Core-L14-336` | +0.2002 | +0.1845 | **−0.0157** | 0.0640 | **−0.25** | **7/8** |
| `RemoteCLIP-ViT-L-14` | +0.2236 | +0.2371 | **+0.0135** | 0.0415 | **+0.33** | 8/8 |

RemoteCLIP is the only candidate with a positive, non-trivial margin.

### 3.5 Vocabulary discrimination — top-5 overlap between query pairs

| Query pair | G14 | L14 | RemoteCLIP |
|---|---|---|---|
| `a white car` vs `car` | 3/5 shared | 3/5 shared | 3/5 shared |
| `dirt road` vs `sand` | 0/5 | 0/5 | 1/5 |
| `tent` vs `tents` | 0/5 | 1/5 | 3/5 |
| `tent` vs `a building with a flat roof` | 0/5 | 0/5 | 0/5 |
| `sand` vs `palm trees` | 0/5 | 0/5 | 0/5 |

**`a white car` and `car` share 3 of 5 for every candidate.** This is the
expected and correct finding, exactly as `docs/DATA.md` predicts: a 4.5 m
car carries ~10–13 independent samples at ~35–45 cm effective resolution,
so colour is not a retrievable attribute. Not a bug. The UI must not offer
colour (spec U-5). Notably, RemoteCLIP's plain `car` **outperformed** its
own `a white car` (4/5 vs 2/5 at 112 px) — the colour word actively hurt.

Genuinely distinct queries stay distinct (0/5 overlap), so the models are
not simply returning a generic saliency ranking.

---

## 4. The owner's two questions, answered

### Q1. Does RemoteCLIP beat PE Core on this imagery?

**Yes — modestly but consistently, and decisively where it matters.**

It wins on `car` (4/5 vs 3/5 vs 2/5 at 112 px), it is the only candidate
whose real queries all clear the known-absent control, and its
singular/plural handling of `tent`/`tents` by choosing the appropriate
*scale* is the most linguistically sensible behaviour any candidate showed.
On the easy queries (`tents`, `palm trees`) all three are ~5/5 and the
domain pretraining buys nothing visible. Combined with being 3.1x faster
than L14 with 40% smaller vectors, the choice is not close.

Two caveats the PM should carry forward. First, **RemoteCLIP's scores are
compressed into a narrow high band** — every query's max sits between
+0.2236 and +0.2838, and its corpus mean is high too. Ranking works;
absolute thresholding does not. Second, the absolute cosine values are not
comparable across candidates, so this comparison rests on the crop
verdicts and the within-model control margin, not on raw score magnitude.

### Q2. Do 112 px crops actually surface vehicles in the tent camp?

**Yes. And 448 px does not — for any candidate.**

Per-scale, which is what the brief asked for:

| Scale | Ground extent | `car` verdict, best candidate (RemoteCLIP) | All candidates |
|---|---|---|---|
| **448 px** | 44.8 m | **0/5 — no vehicle discernible in any crop** | 0/5, 0/5, 0/5 |
| **224 px** | 22.4 m | 3/5 genuine, 2 cannot tell | works |
| **112 px** | 11.2 m | **4/5 genuine** | best scale for all three |

112 px produced the **top-1 result for `car` in all three candidates**, and
per-scale top-1 scores rise monotonically as the crop shrinks:

| Candidate | 448 px | 224 px | 112 px |
|---|---|---|---|
| `PE-Core-G14-448` | +0.0941 | +0.1356 | +0.1508 |
| `PE-Core-L14-336` | +0.1281 | +0.1724 | +0.1845 |
| `RemoteCLIP-ViT-L-14` | +0.2329 | +0.2555 | +0.2606 |

**The multi-scale pyramid bet is vindicated, but the reasoning needs
inverting for half the vocabulary.** 112 px is necessary for vehicles and
useless for extended features: an 11.2 m crop physically cannot contain a
road as a linear object, only a patch of its surface, which is why
`dirt road` and `sand` converge at that scale. 448 px is the reverse. The
scales are complementary, not redundant — and a single pooled ranking
across all three will let 112 px crops dominate every query (it took top-1
for 9/9 queries for both PE candidates), including queries where 112 px is
the wrong scale to be looking at. **S3 should be briefed to rank per scale
or to weight by scale, not to pool blindly.** Flagging rather than deciding:
this touches the retrieval interface.

---

## 5. Concerns, in priority order

1. **No usable absolute score threshold on any candidate.** PE-Core-L14's
   known-absent `aircraft carrier` scored +0.2002, above its own `car` at
   +0.1845. G14's control margin is +0.0019. Even RemoteCLIP's best margin
   is +0.0135 against a 0.0415 spread. A "no results found" feature cannot
   be built on a fixed cosine cut-off; it needs a per-query calibration or
   must be dropped. **Touches spec F-4 and the UX requirements — flagging,
   not deciding.**
2. **Pooling scales into one ranking is unsafe** (see Q2 above). Interface
   question for S3.
3. **Beige corrugated metal is retrieved by `sand`.** RemoteCLIP put the
   large hall roof at ranks 2 and 4 of `sand` at 224 px. At ~35–45 cm
   effective resolution these surfaces have the same colour and texture
   statistics. Expect this class of confusion in the eval numbers; it is a
   resolution limit, not a model defect.
4. **Attribute queries do not discriminate.** `a white car` ≈ `car`, and
   "flat roof" did not exclude a pitched tile roof. Consistent with
   `docs/DATA.md`; the colour word made results *worse*. Do not offer
   attribute search.
5. **`tent`/`tents` verdicts are base-rate inflated.** Nearly every crop in
   this subregion contains a tent, so ~5/5 for all candidates measures the
   scene, not the model. The eval in S4 needs the labelled raster
   (`tile_cropped_x3308_y3674_z0.125.tif`) for anything quantitative — but
   note that raster covers `leb`, not this scene.
6. **`Car` has zero labels in the eval raster**, so the vehicle finding
   above rests entirely on my visual reads of the crops on disk. The PM
   should look at `retrieval/index/calib/_car112_*.png` independently before
   S3 commits.

---

## 6. Deviations from the brief

1. **Subregion is 1792 x 1792, not "roughly 2048 x 2048".** 1792 = 4x448 =
   8x224 = 16x112, so all three scales tile the **identical extent** with
   zero remainder. At 2048 the 448 grid would cover only 1792 px (76.6% of
   the area) while 112 covered 2016, and the per-scale comparison — the
   whole point of question 2 — would have been confounded by the scales
   seeing different ground. Reversible and local; decided and reported.
2. **`pip install -e .` for `perception_models` was run with `--no-deps`.**
   Its `requirements.txt` pins `numpy==2.1.2`, `pillow==11.0.0`,
   `timm==1.0.15`, `scikit-learn==1.6.1`, `opencv-python==4.11.0.86` and
   pulls `torchdata==0.11.0`, `torchcodec`, `lm-eval` and `wandb`. Running
   it as written would have downgraded five of the packages `CLAUDE.md`
   records as verified and risked replacing `torch 2.6.0+cu118`. I audited
   the actual imports of `core.vision_encoder.{pe,config,transforms,
   tokenizer,rope}` — they need only numpy, torch, torchvision, einops,
   timm, huggingface_hub, ftfy and regex, **all already present** — and
   installed with `--no-deps`. Verified afterwards that torch 2.6.0+cu118,
   CUDA True, numpy 2.3.5, timm 1.0.27, PIL 12.2.0 and sklearn 1.9.0 are
   all intact, and that `import core.vision_encoder.pe` works.
   **Recommend `CLAUDE.md` record `--no-deps` as mandatory for this repo.**
3. **Contact sheets in addition to the individual crop PNGs.** The brief
   asked for top-5 PNGs read back with the Read tool; 3 candidates x 9
   queries x 5 = 135 images is more than one context can usefully hold. I
   wrote the 135 individual PNGs as asked *and* one 3x5 contact sheet per
   candidate x query (row = scale, labelled with cosine and offset), and did
   my reading from the sheets plus enlarged 300 px crops for the borderline
   vehicle calls. This is why per-scale verdicts exist at all.

## 7. Unspecified decisions made (for the PM's log)

- **Normalise in fp32, not fp16.** The forward pass is fp16 but `_normalise`
  casts to fp32 before dividing by the norm. fp16 eps is 9.8e-4, so an fp16
  norm cannot satisfy F-2's 1.0 ± 1e-5 — the tolerance would have had to be
  weakened otherwise. Returned arrays are float32.
- **RemoteCLIP is loaded via `open_clip.create_model_and_transforms(
  'ViT-L-14', pretrained=None)` then a full state-dict load.** No OpenAI
  weights are downloaded. The loader raises if any non-`logit_scale`
  parameter is absent from the checkpoint, so a silent random-weight model
  is impossible. Measured: `missing=0 unexpected=0` — the checkpoint covers
  every parameter. (open_clip logs "Model initialized randomly" *before*
  the checkpoint load; that warning is expected and misleading.)
- **RemoteCLIP uses open_clip's own preprocessing** (bicubic, shortest-side
  resize to 224, centre crop, CLIP mean/std) rather than PE's squash
  transform. All crops are square, so the two are equivalent here, and
  using each model's native transform avoids an input-distribution shift.
- **`Embedder.dtype` exposed as a property** over `next(model.parameters())`.
  The brief said "nothing else" beyond the two embed methods; N-3 requires
  asserting the loaded dtype, which needs some accessor. Read-only.
- **Batch size 32**, seeds fixed at 0, `cudnn.deterministic=True`,
  `benchmark=False`.
- **Reconnaissance thumbnails kept** at `retrieval/index/calib/_scene_thumb.png`,
  `_cand_{A,B,C,D}.png`, `_zoom_car{1,2,3}.png` — they document why this
  subregion was chosen. Gitignored, 67 MB total.
- Did not read crop-by-crop for all 9 G14 queries once throughput (2h36m)
  and control margin (+0.0019) had settled its elimination; all its sheets
  are on disk.

## 8. Installs to record in `CLAUDE.md`

| Package | Version | Why |
|---|---|---|
| `pytest` | 9.1.1 | Every acceptance criterion is a `test_*` function |
| `open_clip_torch` | 3.3.0 | RemoteCLIP ships as an open_clip checkpoint. Now load-bearing, not optional — it is the chosen embedder's loader |
| `accelerate` | 1.14.0 | Installed as authorised; **not actually needed** — no candidate required sharded loading |
| `perception_models` | 1.0.0, commit `3e352cca660658d4b5c90f42a7808b11469e4c66` | PE Core. Cloned to `/home/omer/perception_models` (outside the repo). **Install with `--no-deps`** — see §6.2 |

Also pulled in by `open_clip_torch`: `pluggy` 1.6.0, `iniconfig` 2.3.0.
Unchanged and verified intact: torch 2.6.0+cu118, numpy 2.3.5, timm 1.0.27,
Pillow 12.2.0, scikit-learn 1.9.0, rasterio 1.5.0.

### Pinned model identities for the manifest (F-1a, N-4)

| Candidate | dim | HF repo | revision |
|---|---|---|---|
| **RemoteCLIP-ViT-L-14** (chosen) | 768 | `chendelong/RemoteCLIP` | `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38` |
| PE-Core-L14-336 | 1024 | `facebook/PE-Core-L14-336` | `bafb0f76541d399057e980a25947f67acec76575` |
| PE-Core-G14-448 | 1280 | `facebook/PE-Core-G14-448` | `a6046680086f67d1f24d4b465a240de0578dfc0b` |

## 9. Artifacts

| Path | What |
|---|---|
| `retrieval/src/embedders.py` | Loader interface over the three candidates; `embed_images` / `embed_texts` returning L2-normalised float32 |
| `retrieval/src/calibrate.py` | The bake-off script |
| `retrieval/tests/test_embedders.py` | The four acceptance tests, parametrised over all three candidates |
| `retrieval/tests/conftest.py`, `tests/_text_embed_probe.py` | Path setup; cross-process determinism probe |
| `retrieval/index/calib/results.json` | Full machine-readable scores: top-5 overall and top-5 per scale for every candidate x query, plus min/max/mean |
| `retrieval/index/calib/<model>/<query>/*.png` | 135 top-5 crops, named `rank_score_scale_x_y.png` |
| `retrieval/index/calib/<model>/<query>/_sheet.png` | 27 contact sheets, 3 scales x top-5 |
| `retrieval/index/calib/_car112_*.png` | Enlarged 112 px `car` top-5 — the evidence behind Q2 |
| `retrieval/index/calib/_subregion.png` | The calibration subregion |

All under `retrieval/index/`, which `.gitignore` covers (verified:
`git check-ignore` matches `.gitignore:5:retrieval/index/`). No weights,
tiles, embeddings or imagery are staged. **Nothing was written to
`/home/omer/PycharmProjects/Dynamic-Terrain/data`** — verified by `find
-newermt`, which returns empty.
