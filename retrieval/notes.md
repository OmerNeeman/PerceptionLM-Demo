# Decision log

Newest entries at the bottom. Every entry: what, why, downstream impact.

---

## intake — 2026-09-07 — project start

### Stated by owner

- Build a **real** demo on **real imagery**, starting from the GeoTIFFs in
  `/home/omer/PycharmProjects/Dynamic-Terrain/data/leb`.
- Mechanism, specified precisely: tile the raster, embed tiles, embed a text
  query, rank by **cosine similarity** of the embeddings, return the nearest
  neighbour tiles.
- The user should be able to **define an AOI and dates of interest**, then
  retrieve from all imagery relevant to the query.
- **Index everything** under `/home/omer/PycharmProjects/Dynamic-Terrain/data`,
  not just `leb`.
- Methodology: multi-agent, PM role, subagents do the implementing.

### Decided at intake (owner ratified, 4 questions)

1. **Embedder: PE Core for retrieval + PLM for tile Q&A.**
   *Why:* the mission as written named PerceptionLM, but PLM is a generative
   VLM — PE encoder into MLP into Llama decoder, emitting text tokens. It has
   no text tower aligned to its image features, so `cos(text, tile)` is not
   defined for it. Cosine retrieval requires a dual-tower contrastive model.
   Perception Encoder **Core** is that model, in the same paper family, with
   aligned towers, and Apache 2.0 rather than PLM's gated research licence.
   *Impact:* PE Core is the retrieval engine; PLM's role narrows to answering
   questions about tiles already retrieved.

2. **Interaction: query -> ranked tiles -> ask about a chosen tile.**
   *Impact:* the expensive VLM pass runs on a handful of tiles, never the
   whole index, so cost stays bounded.

3. **Full local index; the HTML viewer is a per-AOI export.**
   *Why:* a standalone artifact caps at 16 MB. `leb` alone just fits (int8
   embeddings 3.2 MB + 1,890 thumbnails at 5 KB = 12.6 MB), but 5,000 tiles
   of thumbnails alone is 15-25 MB, so a single file cannot hold all six
   AOIs. Resolved by separating the index from the artifact.
   *Impact:* two components - an on-disk index, and an exporter that writes a
   shareable subset. Embedding precision and thumbnail budget become spec
   items, not implementation details.

4. **All six AOIs indexed from the start.**
   *Why:* owner's explicit choice. PM flagged the risk first - only `leb` has
   ground truth, and heterogeneous CRS/GSD across AOIs may mean a tile covers
   a different ground area per site. Owner chose breadth anyway.
   *Mitigation, PM:* index all six; anchor **evaluation** on `leb`, the only
   AOI where recall@k is measurable rather than eyeballed.

### Established by recon (measured, not assumed)

- **True ground GSD is 10.47 cm/px, not the 12.5 cm/px `gdalinfo` prints.**
  EPSG:3857 metres are inflated by 1/cos(lat); at lat 33.107 the factor is
  0.8377. A 448 px tile therefore covers **46.9 real metres**, not 56.0.
  Verified independently by the PM.
- **The imagery is oversampled ~3-4x.** Gradient-growth ratios and
  downsample/re-upsample residuals (2.9% variance lost at 4x for 2022, 5.3%
  for 2025) both indicate true optical resolution near **35-45 cm** - this is
  sub-metre satellite imagery resampled onto a fixed web-map grid. 2022 is
  measurably softer than 2025.
  *Impact - this constrains the query vocabulary.* A vehicle carries only
  ~10-13 independent samples along its length. Presence and coarse class
  ("car", "truck", "debris", "damaged building") are supportable; fine
  attributes ("white pickup", "red sedan") are not, and must not be promised
  in the demo.
- `leb` is **one 2.11 x 1.02 km footprint at two dates**, all 31 rasters
  pixel-co-registered. 945 full 448 px tiles per date, **1,890 total**.
- **Only 2 of `leb`'s 50 TIFFs are source RGB.** The other 48 are derived
  model outputs - binary masks, segmentation, change-detection rasters.
  Filenames alone do not distinguish them; band count and value distribution
  do.
- Source scenes are **strip-organised (20179x1) with no overviews**, so a
  random 448 px window costs ~27 MB of I/O to return 0.6 MB of pixels -
  roughly 45x amplification. Sequential tiling is unaffected; interactive
  AOI access is not. COG conversion is therefore a stage, not an optimisation.
- Hardware is ample: 2x RTX 6000 24 GiB, GPU 0 idle. Three environment traps
  recorded in `CLAUDE.md` (user-site CPU torch shadowing, bf16 on Turing,
  the GSD correction).
- **PE Core is downloadable now; PLM returns a hard 403** (`gated: manual`,
  hand-reviewed). Owner is requesting access; the Q&A layer is built behind a
  swappable interface with an ungated VLM as default so nothing blocks.
- **Free evaluation signal exists.** `ai_segmentation.gpkg`, the `cdinf`
  positive-detection shapefiles, and a 10-class legend that includes `car`
  give retrieval ground truth without manual labelling: query a class, check
  the returned tiles against the segmentation. This resolves the "nothing can
  be measured" problem that stalled the earlier tracking work.

### Open - blocking final acceptance criteria

The four problem anchors below were **not** covered at intake; the problem
statement in `CLAUDE.md` is the PM's inference and is marked pending.

1. Who is the primary user, and what is their current workaround?
2. What does that workaround cost today - time per search, or searches not
   attempted at all?
3. Why now?
4. What retrieval accuracy would count as good enough to proceed?

Per methodology section 3, measurable acceptance criteria cannot be finalised
without (4). `spec.md` will carry provisional criteria, flagged as such.

### Deferred

- Change-aware retrieval ("buildings that became debris") was offered and not
  chosen. The `cdinf` outputs already detect exactly this, so it remains the
  highest-value follow-on once single-epoch retrieval is proven.

---

## recon-2 — 2026-09-07 — the rest of /data characterised

All figures measured unless marked inferred.

### Scope is much smaller than 17 GB suggests

**Only 12 of ~232 rasters outside `leb` are source imagery; 74 unique source
scenes in total** after dedup (`iran/Ax00_y06.tif` and `Ax01_y06.tif` are
md5-identical to `Teheran/`'s — do not double count).

**D-1 becomes a hard measurable rule:** every derived raster in this tree has
**<= 6 distinct values per band**; every source raster has **187-256**. The
source classifier no longer needs a heuristic.

### Four scale regimes, not one

| Group | CRS | true GSD | scenes | 448 px tile = | area/tile |
|---|---|---|---|---|---|
| A | UTM 36N (32636) | 10.00 cm | 6 (AYOSH x2, gaza x3, X605) | 44.8 m | 0.20 ha |
| C | EPSG:3857 | 10.47 cm | 2 (leb) | 46.9 m | 0.22 ha |
| B | UTM 39N (32639) | 50.00 cm | 67 (Teheran 66, iran 1) | 224 m | 5.0 ha |
| D | EPSG:3857 | 415.7 cm | 1 (sin) | 1862 m | 347 ha |

- **A and C are one scheme** (4.7% apart). Different CRSs, matching true
  scale — only the geo-lookup layer needs to handle both.
- **B is 5x coarser = 25x the ground area.** Vehicles are not resolvable at
  50 cm (confirmed visually). "tents", "a white car" are answerable in A/C and
  unanswerable in B.
- **D is `sin`** — 65536 x 98304, 6.44 gigapixels, the whole Sinai Peninsula
  (272 x 408 km ground). **86% of the entire nominal tile budget**, and
  nothing man-made resolvable. Shares EPSG:3857 with `leb`, which is a trap:
  identical CRS, 40x different scale. Also has no overviews, so there is no
  cheap path to a coarse index over it.

**Decision (PM):** index **A + C only** — 8 scenes, ~4,146 valid 448 px tiles.
B is admitted later as a separate coarse tier if wanted; D is excluded.
Folding all four into one embedding space would ask the same text query of
tiles whose footprints differ by three orders of magnitude.
*Impact:* the index is far smaller than budgeted, which makes a multi-scale
pyramid affordable. Supersedes the "all six AOIs" sizing assumption — the
AOIs are still all indexed, but `sin` is out on resolution grounds, not
preference.

### F-7 (date filter) is unsatisfiable as written — amendment required

Only `leb` is a true multi-date pair over one footprint. **Five of the eight
non-`leb` source footprints carry no acquisition date in the file at all**
(AYOSH x2, gaza, X605, sin). The two that do carry *acquisition windows*, not
dates: Teheran 2024-05-22 -> 2025-01-07, and `iran/x1_y7` **2015-09-15 ->
2024-04-14** — an 8.6-year window, so that mosaic may internally mix epochs.
Date filtering requires an **external manifest supplied by the owner**; it
cannot be derived from the files. Flagged to owner; `spec.md` F-7 pending
amendment.

### Two significant finds

- **This is the second attempt.** `/data/vlm_logs/` holds four records from
  2026-02-16: a bbox + text-prompt UI over a 10240^2 scene, operator prompts
  `"tents"` and `"find all tents in the frame"`. Every record has
  `"mock": true` and a byte-identical analysis paragraph — a stub endpoint, no
  real inference ever ran. Confirms the intended workload is exactly
  small-object text retrieval on the 10 cm imagery.
- **More ground truth than recorded at intake.**
  `tile_cropped_x3308_y3674_z0.125.tif` is a **45-class semantic label
  raster over the `leb` AOI** (`Car`, `House`, `DirtRoad`, `PavedRoad`,
  `Pavement`, `Water`, plus rock/soil/vegetation classes), 31 classes present,
  EPSG:4326 at ~12.1 cm. Companion to `All_Valid_Labels_45.txt`. Richer eval
  signal than the 10-class legend — E-1 should use this.

### Planned work that evaporates

The 0.1 m AOIs are **already tiled 256x256** (~2.9x read amplification for a
448^2 window). `leb`'s strip layout at ~45x is the anomaly, not the norm. S2's
COG conversion narrows to `leb` alone.

### Excluded from the index

- `sin/Sini_Oct_Det_2025.tif` — resolution regime D, see above.
- `/data/gaza.tiff` — **not aerial imagery.** 1600x800, no CRS, no
  geotransform; a ground-level oblique press photograph of armour with the
  Gaza skyline. A web/UI asset.
- `iran/Ax0*_y06.tif` — byte-duplicates of `Teheran/`'s.
- All 146 Teheran + 7 iran + 3 gaza derived masks, and the `leb` derived stack.

### Open with owner

1. Scale strategy for small objects — research in flight (owner asked for SOTA
   and what the PLM/PE papers recommend).
2. F-7 date manifest — owner must supply dates, or F-7 narrows to `leb` only.
3. Confirm `sin` exclusion and whether B (50 cm Teheran/iran) is wanted as a
   separate coarse tier.

---

## research-1 — 2026-09-07 — scale strategy resolved; patch-token approach rejected

Owner asked for SOTA options and what the PLM / PE papers recommend. One
research agent died on a spend limit (the remote-sensing-SOTA and
open-vocabulary-detection half is **still missing** — see Open below). The
late-interaction / dense-CLIP survey completed.

### REJECTED: late interaction over PE-Core patch tokens

The PM floated this last turn as the cheap way to find small objects without a
pyramid. **The literature refutes it directly.**

- **ColPali's own ablation is the exact experiment.** `ColSigLIP` — MaxSim over
  a CLIP-family dual tower's patch tokens — scored **2.5 nDCG@5** vs 81.3 for
  ColPali and **51.4 for plain pooled SigLIP**. Worse than doing nothing, and
  that was *with* contrastive fine-tuning. Authors' reason: "only a pooled
  latent representation is used in the contrastive loss, which does not
  optimize the representations of individual patch and token embeddings."
  https://arxiv.org/html/2407.01449v2
- **Every working multi-vector visual retriever routes patch tokens through an
  LLM decoder and fine-tunes** (ColPali, ColQwen2, ColNomic, Nemotron
  ColEmbed). None works off a frozen CLIP tower.
- **Patch-text alignment degrades as the tower grows.** PACL patch
  classification on VOC: 52.49% (ViT-B/16) vs **27.91% (ViT-L/14)**. We
  proposed PE-Core-**G**/14, larger still. https://arxiv.org/abs/2212.04994
- **Naive patch-text cosine is quantitatively terrible**: 3.1% mIoU ADE20k,
  5.7% COCO-Stuff. https://arxiv.org/html/2312.01597v3 Raw CLIP patch maps are
  anti-correlated — "opposite visualization", background over foreground.
  https://arxiv.org/abs/2304.05653
- **PE-Core architecture blocks it anyway.** Patch tokens are **1536-d**; the
  text-aligned 1280-d space exists only after an 8-head attention-pool block
  then `proj`. PE-Core-G has **no class token**. Patch vectors never entered
  the contrastive objective. PE's own paper says final layers "overwrite
  spatial features"; Meta shipped **PE-Spatial** separately for dense tasks —
  but that is distilled from SAM 2 correspondence with **no language
  supervision**, so its patches are not text-comparable either. There is no
  off-the-shelf text-aligned dense variant of PE.
  https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/pe.py
- **Our query shape is MaxSim's worst case.** With 1-3 token nouns, MaxSim
  collapses to max-over-patches of a single dot product — exactly the naive
  cosine above, with no summation over query terms to suppress noise. ColPali's
  queries are long multi-concept questions; that is the regime where it helps.
- **Storage ends it regardless:** 4,146 tiles x 1024 patches x 1280-d fp16 =
  **10.87 GB**, vs 10.6 MB pooled. A 1024x index for a quality regression.

### ADOPTED: multi-scale single-vector pyramid

Use the only part of PE that is actually text-aligned — the pooled output — and
solve scale with **crops instead of patches**. Published precedent for exactly
this case: **DetailCLIP**, which attacks small objects in remote sensing by
tiling at multiple scales and fusing into a single indexable vector.
https://arxiv.org/pdf/2208.14649

Scales **448 / 224 / 112**, 0% overlap initially, matched to the query classes:
"dirt road" is scene-level (448 = 46.9 m), "mosque" building-level (224 =
23.5 m), "tents" / "car" object-level (112 = 11.7 m, where a car is 5.9% of
tile area rather than 0.37%).

Measured cost over the 8 group-A+C scenes:

| | tiles | local fp16 1280d | export int8 128d |
|---|---|---|---|
| 448/224/112, 0% ov | 108,542 | 278 MB | 13.9 MB |
| same, 50% ov @112 | 349,926 | 896 MB | 44.8 MB |
| multi-vector @448 only | 4.25M vectors | **10.87 GB** | impossible |

Single-AOI static export (X605_Y3388, 10240^2, the tent camp): 11,109 tiles at
128-d int8 + 1.5 MB basemap = **3.40 MB**. The 16 MB cap stops being a design
constraint.

*Impact:* F-1 `TILE_PX` becomes a scale **list**, not a scalar. F-2 stores one
vector per tile per scale. F-8's budget is comfortable. S2/S3 unblocked.

### Constraint: one embedder for all scales

Different models are different spaces, so a query vector cannot be compared
across them. The embedder choice is therefore single and global. Every tile
costs one full forward at the model's input resolution regardless of source
crop size, so 108,542 forwards is the dominant build cost. G14-448 (1280-d,
1.88B vision tower) vs L14-336 (1024-d, ~6x faster) is a live decision for the
calibration stage.

### Held as a side experiment, off the critical path

**MaskCLIP-style surgery on PE-Core** — bypass `attn_pool`, push each patch
through the pool block's value/output projections and `proj` to land in the
1280-d text space, then score per patch. Training-free surgery (MaskCLIP,
CLIP Surgery, SCLIP, GEM, ClearCLIP, TraceCLIP) demonstrably *does* recover
usable dense alignment from a frozen CLIP — 14.1% -> 38.2% mIoU for SCLIP. And
the research found a **genuine gap**: nobody has evaluated MaxSim retrieval
over surgically-densified patch tokens, and nobody has evaluated any of it on
PE. Interesting, publishable, and not what a product-scoping demo should bet
on. Revisit only if the pyramid underperforms.

### Open

1. **Missing research — spend limit.** Remote-sensing-specific retrieval
   (RemoteCLIP, GeoRSCLIP/RS5M, SkyScript, SegEarth-OV) and **open-vocabulary
   detection** (OWLv2, GroundingDINO, YOLO-World) were not covered. Both
   matter: an RS-pretrained CLIP may beat PE Core on aerial imagery, and the
   `vlm_logs` prompt "find all tents **in the frame**" suggests the owner may
   actually want *boxes*, not ranked tiles — a different deliverable. Re-dispatch
   when the limit resets.
2. **GRAFT precedent worth noting**: in this exact domain, getting text-aligned
   patch features required training a new encoder against co-located
   ground-level photos. https://arxiv.org/abs/2312.06960

---

## e1-scoped — 2026-09-07 — E-1 measured and cut back; PM over-sold it

The PM described the 45-class label raster as a free retrieval eval set. On
measurement it is far weaker than advertised, and the spec has been corrected
rather than left optimistic.

Measured (`tile_cropped_x3308_y3674_z0.125.tif`, 8192x8192, EPSG:4326):

- Covers **993 x 500 m = 0.497 km2 = 23% of the `leb` footprint** only.
  210 tiles at 448 px.
- **49.8% `Unclassified`** -> ~0.25 km2 of usable label.
- **`Car`: zero pixels.** The single most important small-object query class
  has no ground truth at all. 33 of 45 classes present, `Car` absent.
- Top three real classes are `Batha` 13.9%, `DryGrassland` 12.8%,
  `Garigue` 6.5% — Mediterranean succession stages, near-zero CLIP signal.
- **33 of 45 classes are lithology/pedology** (Limestone/Dolomite/Nari/Basalt/
  Chalk/Maral variants, Rendzina, TerraRosa, ClayeySoil…). Not visual
  categories; the distinction is field geology. Including them would produce a
  misleadingly bad headline number while hiding that the system works on
  House/DirtRoad/Pavement.
- **EPSG:4326 vs the imagery's EPSG:3857** — needs warping onto the image grid
  before scoring. Now explicitly part of the stage.

**Owner asked whether 10 classes would be better than 45.** Answer: the
10-class list (`cls_leb_legend.txt`: road, sidewalk, building, rut, rock, wall,
fence, tree/hedge, grass+soil+sand, car) **has no ground-truth raster** — it is
a segmentation model's *output* legend, not labels. So it is not an
alternative source, only an alternative vocabulary.

**Decision:** keep the 45-class raster as the label source; evaluate the
**eight** classes that are both CLIP-nameable and supported (DirtRoad, Shadow,
Pavement, House, DirtRoadB, Maquis, UnirrigatedOrchard, PavedRoad), each
reported with its tile support; exclude the geology/pedology classes with a
stated reason; judge **vehicles by eye** (E-1a), optionally with the
segmentation output as clearly-labelled pseudo-ground-truth.

*Impact:* E-1 stays non-gating and becomes honest. It can tell us whether
scene-level and building-level retrieval works. It **cannot** tell us whether
vehicle retrieval works — that remains a by-eye judgement, which is what the
owner's qualitative gate was going to be anyway.

## owner-answers — 2026-09-07 — late interaction confirmed as ColQwen2.5

- **Start with the naive single-vector method** (owner: "A5 - ok let's start
  with this"). Tile, embed pooled, L2-normalise, cosine, top-k. The multi-scale
  pyramid exists only because a car is 0.37% of a 448 px tile.
- **ColQwen2.5 is the chosen late-interaction implementation** when F-11 is
  reached — off-the-shelf, no training by us, ungated (Qwen2-VL, Apache 2.0)
  unlike ColPali's gated PaliGemma. Still local-only: ~256 KB/tile cannot be
  exported.
- **Dense-CLIP surgery stays out of the loop** (owner: "keep it outside the
  loop for now"). Recorded as future work only.

---

## arch-confirm — 2026-09-07 — encoder-only confirmed; the decoder/MaxSim trap

Owner asked directly whether we are aligned on using **only the Perception
Encoder** and not PerceptionLM's generative half, and supplied context arguing
for (1) contrastive vision encoders rather than LLM decoders, (2) multi-vector
indexing, ColPali style, and (3) late interaction / MaxSim over patch tokens.

**Aligned on (1).** PE Core's two towers only; no Llama decoder in the
pipeline. Additional reason beyond compute waste: PLM has no text tower
aligned to its image features, so `cos(text, tile)` is undefined for it.
Retrieval was never something PLM could do.

**Aligned on (2), and it is already the design.** The 448/224/112 pyramid *is*
N vectors per image — one per tile per scale, 11,109 for a single 10240^2
scene. "One for the global thumbnail and one for each dynamic tile" is what is
specced.

**(1) and (3) are mutually incompatible — this is the key point.** ColPali's
MaxSim does not run on the *vision encoder's* patch tokens. It runs on the
**LLM decoder's output token embeddings**: PaliGemma projects SigLIP patch
embeddings into Gemma-2B's text vector space, and the authors state this is
the load-bearing step — "One benefit of inputting image patch embeddings
through a language model is that they are natively mapped to a latent space
similar to the textual input (query). This enables leveraging the ColBERT
strategy." The control experiment that bypasses the decoder is **ColSigLIP:
2.5 nDCG@5** vs 81.3 for ColPali and 51.4 for pooling.

So "pure vision encoder + patch-token MaxSim" is precisely the 2.5
configuration. Note also that ColPali-style retrieval is **not** cheap: it
requires running a 2B decoder over 1,024 patch tokens per tile at index time —
the very cost (1) seeks to avoid.

**Measured cost comparison (PM computed):**

| | one AOI, 11,109 tiles | all 8 scenes, 108,542 |
|---|---|---|
| PE Core pooled | **28 MB**, ~7 min | **278 MB**, ~72 min |
| ColQwen2.5 patch tokens | 2.91 GB, ~1.2 h | 28.45 GB, ~11.8 h |
| + ColBERTv2 20 B/vec | 0.23 GB | 2.22 GB |

102x the storage. Even compressed, one AOI is 14x over the 16 MB export cap,
so the shareable HTML deliverable would be lost.

**Same diagnosis, different remedy.** The owner's diagnosis — a tiny car's
representation is diluted by the surrounding tile — is correct. Two fixes
exist: un-dilute the representation (patch tokens, needs the decoder), or
reduce what dilutes it (shrink the tile). At 112 px a car is **5.9% of the
tile instead of 0.37%** — a 16x gain in signal fraction, achieved with the
only text-aligned representation PE Core exposes.

**Unchanged decision:** encoder-only pooled pyramid for the build; ColQwen2.5
late interaction stays queued as **F-11**, an A/B on one AOI after retrieval
works, with the static export knowingly traded away if it wins. The owner is
right that multi-vector late interaction is SOTA for micro-detail; the
reservations are that it needs the decoder, costs 102x, and its only
off-the-shelf form was tuned on 127,460 *document* query-page pairs, so aerial
transfer is unmeasured.
