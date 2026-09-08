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

---

## owner-ratify-2 — 2026-09-07 — problem anchor closed, three decisions taken

Asked at the start of implementation, answered by the owner.

**1. Primary user: the owner, scoping a product.** The demo exists to inform a
build/don't-build decision. Success is *enough signal to judge feasibility*,
not time-saved for a third-party analyst.
*Impact — this is load-bearing and it simplifies things.* The
"RATIFICATION PENDING" block in `CLAUDE.md` is **removed**; the problem
statement is now stated fact. It also retires the three anchor questions
`notes.md#intake` left open (workaround cost, why-now, accuracy bar): there is
no external workaround to cost, and the accuracy bar is *whatever is enough to
decide*, which is exactly why the owner set a qualitative gate. It further
confirms **U-5 is the most important UX requirement in the spec** — a demo
whose purpose is a go/no-go decision must not overstate what the imagery
supports, or it corrupts the decision it exists to serve.

**2. Licence fork: take the measured winner, document the licence.** If the
best embedder on this imagery carries a restrictive licence, use it and record
the caveat rather than shipping worse retrieval. Rationale accepted: the index
is rebuildable from source imagery, so the choice costs one re-embed to
reverse — unlike PLM, whose licence problem was structural.
*Impact:* F-0 is decided on retrieval quality; licence is a recorded property
of the manifest, not a veto. **Open:** RemoteCLIP's exact licence terms are not
yet verified (no LICENSE file ships in the HF snapshot) — see open item below.

**3. Demo query vocabulary: all four categories.** Small objects (tents, cars),
structures (buildings, flat roofs), terrain (dirt road, sand, palm trees), and
damage (rubble, debris, collapsed roof).
*Impact:* the owner will query at **all three scales**, which materially
raises the stakes on the cross-scale ranking decision below — a single pooled
ranking dominated by 112 px would break the terrain queries specifically.
Damage queries are **not answerable on `X605_Y3388`** (a tent camp); they are
what `leb`'s 2022->2025 pair and `gaza` contain. So M2's fan-out is worth more
than `plan.md` currently ranks it, and the deferred change-aware retrieval
gains value. Not pulled into M1 — the milestone stays a running demo — but
recorded so S6 is not treated as optional breadth.

---

## s0-bakeoff — 2026-09-07 — RemoteCLIP chosen by measurement; PM-verified

**Stage S0 gated GREEN.** Worker status was `DONE_WITH_CONCERNS`; both
concerns that touch the spec were put to the owner and decided (below), so the
stage advances rather than being accepted as a soft pass.

### The decision

**Embedder: `RemoteCLIP-ViT-L-14`, 768-d**, `chendelong/RemoteCLIP` pinned at
revision `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`, loaded via `open_clip`.

*Why:* it wins the one query that was actually in doubt, and it is the only
candidate whose known-absent control is beaten by all 8 real queries.

| | control max | weakest real max | margin | spread | margin/spread | real > control |
|---|---|---|---|---|---|---|
| PE-Core-G14-448 | +0.1489 | +0.1508 | +0.0019 | 0.0498 | 0.04 | 8/8 |
| PE-Core-L14-336 | +0.2002 | +0.1845 | **-0.0157** | 0.0640 | **-0.25** | **7/8** |
| RemoteCLIP-ViT-L-14 | +0.2236 | +0.2371 | **+0.0135** | 0.0415 | **+0.33** | 8/8 |

Vehicle retrieval at 112 px, by eye: RemoteCLIP **4/5**, PE-Core-L14 3/5,
PE-Core-G14 2/5. Throughput: RemoteCLIP **265 tiles/sec** (6.8 min for the
108,542-tile pyramid) vs L14 84.5 (21 min) vs G14 11.6 (**2h36m**). Its 768-d
vectors are also the smallest, which is direct headroom against F-8.

*Impact:* **F-2 dimensionality becomes 768**, not 1280. `open_clip_torch`
becomes a load-bearing runtime dependency, not a bake-off convenience. PE Core
is retained as a documented fallback — the loader interface in
`src/embedders.py` covers all three candidates, so reverting is a config
change plus a re-embed.

### Both owner questions from the handoff, answered

**Does RemoteCLIP beat PE Core on this imagery? Yes — modestly, consistently,
and decisively on vehicles.** On easy queries (`tents`, `palm trees`) all three
score ~5/5 and aerial pretraining buys nothing visible; the gain is
concentrated exactly where the domain gap should show. RemoteCLIP also handled
`tent` vs `tents` most sensibly, by selecting the appropriate **scale** (112 px
for one tent, 224 px for a group) rather than by returning different crops at
one scale. This settles the question the killed RS-CLIP survey was owed —
empirically, on our own imagery, for ten minutes of compute.

**Do 112 px crops surface vehicles? Yes — and 448 px surfaces none at all**
(0/5, 0/5, 0/5 across all three candidates). 112 px took top-1 for `car` in
every candidate; per-scale top-1 scores rise monotonically as the crop shrinks;
224 px also works (3/5). **The multi-scale pyramid is vindicated**: without the
112 px level this system would not find a vehicle at all. That is the strongest
single result in S0 and it validates `notes.md#arch-confirm`'s core claim — a
16x gain in signal fraction, obtained from the only text-aligned representation
available.

### PM verification — not taken on trust

- Re-ran the suite independently: **12 passed in 72.83s**, all three candidates.
- **Read the vehicle crops myself.** `RemoteCLIP` top-5 at 112 px: #1 dark car
  on a track plus a light car, #2 white sedan and a dark car, #3 vehicles among
  palms, #4 silver car and a white vehicle, #5 tarps and sand — a miss. **4/5,
  confirming the worker's read.** `PE-Core-L14`: #1, #2, #5 hits; #3 and #4
  palms and tarps. **3/5, also confirmed.**
- Cross-checked the control claim two independent ways: the table says
  PE-Core-L14's control max is +0.2002, and its top `car` crop is labelled
  +0.1845 in the contact sheet I read. **The known-absent query really does
  outscore the real one.**
- Data tree untouched (`find -newermt`, empty). Nothing large tracked (92 KB
  total; `.gitignore:5` covers `retrieval/index/`).
- RED-then-GREEN present and genuine: two independent REDs for N-3 (naive
  draft loading fp32, then trap 1 reproduced live by dropping
  `PYTHONNOUSERSITE`), plus a **third real bug the F-3 test caught on its own**
  — PE's `load_ckpt` prints to stdout and contaminated a cross-process probe
  channel. Not born-green.

### Two interface decisions — flagged by the worker, taken by the owner

**1. Ranking is per scale, not pooled.** *Owner decision.* The finding:
112 px took top-1 for **9/9** queries on both PE candidates, so it dominates
any single pooled ranking — including queries where it is the wrong scale to
look at. An 11.2 m crop physically cannot contain a road as a linear object,
which is exactly why `dirt road` and `sand` converge there.

Results are returned as **three rankings, one per scale, each labelled by true
ground extent (11 m / 22 m / 47 m)**. Chosen over score normalisation because
the fusion weights would be a tuning knob with no ground truth to tune against
(`Car` has zero labels), and over a scale selector because U-6 already has
filters to keep legible. It is also self-documenting: every query the owner
types re-answers "does the 112 px level earn its place".

*Why a clarification and not an amendment:* `spec.md` F-4 specifies cosine,
sorted descending, bounded, exact — and says nothing about how scales combine.
This was **unspecified, not weakened**; per methodology §8, unspecified is a
blocker, and the worker correctly stopped rather than inventing a fusion rule.
No criterion was loosened. F-4 and U-2 gain the per-scale requirement.

**2. The empty state uses a relative gap, not an absolute threshold.**
*Owner decision.* No fixed cosine cut-off is usable: PE-Core-L14's
`aircraft carrier` scored **+0.2002 against its own best `car` at +0.1845**,
G14's margin is +0.0019, and even RemoteCLIP's +0.0135 sits against a 0.0415
spread. Absolute cosine magnitude carries no cross-query meaning.

U-3 is therefore satisfied by a **per-query relative gap** — a weak match is
one whose top score fails to stand out from the corpus mean by a stated
multiple of spread. **No spec amendment: U-3 never required an absolute
threshold**, only that the empty state exist and guide. E-2's known-negative
query supplies the calibration data, and the multiple must be stated in the
UI, not buried.

### Four findings accepted as measured limits, not defects

These are the resolution reality that `spec.md` D-5 and U-5 already anticipate.
Logged so later stages and the eval report expect them:

- **Beige corrugated metal is retrieved by `sand`** — RemoteCLIP put a large
  hall roof at ranks 2 and 4 of `sand` at 224 px. At ~35-45 cm effective
  resolution these surfaces share colour and texture statistics. Expect this
  class of confusion in E-1; it is a resolution limit.
- **Attribute queries do not discriminate.** `a white car` ~ `car` (3/5 shared
  crops on every candidate), and "flat roof" did not exclude a pitched tile
  roof. The colour word made RemoteCLIP measurably **worse** (2/5 vs 4/5).
  Exactly the `docs/DATA.md` prediction. **Do not offer attribute search, and
  do not use attribute phrasings in the demo's example queries.**
- **`tent`/`tents` verdicts are base-rate inflated** — nearly every crop in the
  calibration subregion contains a tent, so ~5/5 measures the scene, not the
  model. Anything quantitative needs the labelled raster, which covers `leb`,
  not this scene.
- **The vehicle result rests on visual reads**, because `Car` has zero pixels in
  the eval raster. Mitigated by the PM independently reading the same crops and
  agreeing; it cannot be mitigated further, and E-1a already says so.

### Deviations, all accepted

1. **Calibration subregion 1792x1792, not ~2048x2048.** 1792 = 4x448 = 8x224 =
   16x112, so all three scales tile the **identical extent** with zero
   remainder. At 2048 the 448 grid would have covered 76.6% of the area the
   112 grid saw, confounding the per-scale comparison that was the whole point.
   **Better than the brief.** The brief was wrong; the worker was right.
2. **`perception_models` installed with `--no-deps`.** Its `requirements.txt`
   pins `numpy==2.1.2`, `pillow==11.0.0`, `timm==1.0.15`,
   `scikit-learn==1.6.1`, `opencv-python==4.11.0.86` and pulls `torchdata`,
   `torchcodec`, `lm-eval`, `wandb`. Running the brief's command as written
   would have downgraded five packages `CLAUDE.md` records as verified and
   risked replacing `torch 2.6.0+cu118` — i.e. **the brief as written would
   have broken the environment it was written to protect.** The worker audited
   the actual imports (numpy, torch, torchvision, einops, timm,
   huggingface_hub, ftfy, regex — all present), installed with `--no-deps`, and
   verified the env intact afterwards. **`--no-deps` is now mandatory in
   `CLAUDE.md`.** Best catch of the stage.
3. **Contact sheets in addition to the 135 individual crop PNGs** — 3
   candidates x 9 queries x 5 crops does not fit usefully in one context. The
   sheets (row = scale, labelled with cosine and pixel offset) are why
   per-scale verdicts exist at all, and they are what made the PM's independent
   re-read cheap. Adopt this pattern for later visual-judgement stages.

### New environment trap, discovered

**PE's `load_ckpt` prints to stdout**, so a subprocess that returns data on
stdout gets a contaminated channel. Recorded as trap 4 in `CLAUDE.md`. Only
bites stages that shell out to a PE process — which, now that RemoteCLIP is
chosen, is a narrower risk than it was.

### Dependencies added — recorded for N-4 / §14

`pytest` 9.1.1 · `open_clip_torch` 3.3.0 (**load-bearing** — loads the chosen
embedder) · `accelerate` 1.14.0 (installed, **not needed**; no candidate
required sharded loading — kept, harmless) · `perception_models` 1.0.0 @ commit
`3e352cca660658d4b5c90f42a7808b11469e4c66`, cloned to `/home/omer/perception_models`,
**outside** the repo, installed `--no-deps`.

### Licence — resolved the same day

**RemoteCLIP is Apache 2.0**, verified at `github.com/ChenDelong1999/RemoteCLIP`.

The HF repo `chendelong/RemoteCLIP` has **no model card and no LICENSE file** —
only the `.pt` blob — so the terms come from the upstream source repo, and the
manifest records that provenance rather than implying the weights shipped with
a licence of their own.

*So the owner's licence fork never had to be paid.* The measured winner carries
the **same Apache 2.0 as PE Core**, and unlike PerceptionLM (FAIR
non-commercial) it does not disqualify a future product. The debt was
discharged the day it was opened.

*Residual, carried as debt rather than closed:* RemoteCLIP's training corpora
(RET-3 / SEG-4 / DET-10, aggregated from third-party remote-sensing datasets)
do not state their own terms. That is a **dataset-lineage** question for a
commercial launch, not a weights-licence question, and it is legal work rather
than engineering — so it must not stall the demo whose whole purpose is to
inform whether a launch happens at all.

---

## owner-portability — 2026-09-07 — portability becomes a spec requirement, not a doc task

**Owner asked for three things:** (1) to explore and sample the system
themselves once the PM has verified it works, (2) everything committed and
pushed, (3) an `INSTRUCTIONS.md` a fresh session can follow to set up and run
the system **on any computer — Linux or Windows, with a different GPU**.

### (3) is a design change, and it arrived at the right time

The PM audited what S0 and S1 actually produced:

| Location | Hardcoded |
|---|---|
| `src/embedders.py:35` | `float16` |
| `src/inventory.py:78` | `/home/omer/PycharmProjects/Dynamic-Terrain/data` |
| `src/calibrate.py:34` | the demo scene's absolute path |
| several docstrings | `/home/omer/anaconda3/envs/geo/bin/python` |

**No document can fix that.** And the fp16 case is worse than a mere path
problem: `CLAUDE.md` trap 2 says "fp16, never bf16" and calls model cards
recommending bf16 *wrong for this machine* — which is true, and true **only**
for this machine. These are Turing cards (cc 7.5) where bf16 is emulated at
7.0 TFLOP/s against 38.9 for fp16. On **Ampere or later, bf16 is native and
the better choice**. So a correct portable implementation must reach the
*opposite* conclusion on the owner's next GPU, and a CPU-only machine needs
fp32 because fp16 on CPU is unsupported or pathologically slow.

*Impact:* two new requirements — **N-8** (portable across OS, GPU generation
and CPU-only; nothing machine-specific in `src/`; dtype selected by device
capability as an asserted pure function) and **N-9** (`INSTRUCTIONS.md`,
verified **by execution on a foreign machine**, not by review — anything the
fresh session had to ask counts as a defect in the document).

*Trap 2 is not repealed, it is generalised.* fp16 remains correct here, as a
**case** of the capability rule rather than as the rule. Any worker that reads
`CLAUDE.md` and hardcodes fp16 is still doing the right thing on this
hardware and the wrong thing in the repository.

### New stage S1a, before S2 — and why not at S9

Portability lands as **S1a: config + device abstraction**, sequenced *after*
S1 and *before* S2. It is the only stage permitted to edit S0's and S1's
files.

The reason it is not deferred to S9 (the existing cold-start stage) is
arithmetic: S2, S3, S4 and S5 would each copy the hardcoded pattern, so
retrofitting five modules costs strictly more than fixing two. Deferring a
cross-cutting property until the end is how it stops being achievable.

Cheap to do now, so it is done now. The index itself is not the constraint —
re-embedding one AOI is ~42 s at 265 tiles/sec — the *code pattern* is.

### (1) and (2)

**(1) Hand-over point is M1, after S5.** That is the first moment anything is
explorable: one AOI indexed at three scales, queryable, exported as a
standalone HTML. The PM verifies it works first — re-running the tests and
using the export as a confused user — then hands it over. `INSTRUCTIONS.md` is
written when there are real, tested commands to document; written earlier it
would be fiction.

**(2) Commit and push authorised by the owner**, 2026-09-07. Branch-per-stage
still holds; `main` stays protected and is reached by merge, not by direct
commit.

---

## s1-review — 2026-09-07 — S1 sent back; D-1 amended, D-2 strengthened

Stage S1 returned `BLOCKED` on its brief's own named condition — correctly. It
did **not** tune a test to hide a mismatch, which is the behaviour the handback
protocol exists to produce.

### PM verification, before any adjudication

- Re-ran the suite: **21 passed**.
- `leb` recursive: **65 files, 31 `.tif`/`.tiff`** — denominator confirmed as
  **31**.
- The contested crop: **byte-identical** to the (4096, 3072) 1024x1024 window
  of `2022-10-29.tif`, md5 `e3b380b8e5c3` both, and **not** matching
  `2025-06-06.tif`. Distinct counts `[236,246,255]`, so it genuinely passes
  `> 64`.
- Derived-raster distinct-value ceiling: 16 among `derived_raster` records on
  the sampled grid, 5 records exceeding 6. So "<= 6" is wrong.

### The independent reviewer

Dispatched because methodology §3 requires independent confirmation for an
amendment to a **measurable** criterion — owner-alone is explicitly weak, and
this is the most gameable rule in the methodology.

**Verdicts:** spec compliance **PASS** on D-1 (rule), D-2, D-3, D-4; code
quality **PASS with findings**. It **mutated the implementation 16 ways and 15
were caught by the semantically correct test** — no born-green, tautological or
self-asserting test found. It attacked the classifier with 20 constructed
fixtures — real imagery behind three derived-looking names, masks behind date
names, both sides of the `> 64` boundary, 2-band real imagery, 4-band with
constant alpha — and **could not make it read a filename**. That is the
strongest part of S1 and it is now explicitly out of scope for changes.

Two observations worth keeping:
- The mutation `all()` -> `any()` was caught **only** by a one-line pure-rule
  unit test (`passes_pixel_rule(3, [238, 245, 3]) is False`). Every data-driven
  test still passed, because on real rasters a band that passes usually means
  all bands pass. **The most trivial-looking line in the file is the sole guard
  against a real semantic inversion.** Argument for keeping cheap unit tests
  next to data-driven ones.
- The reviewer named where it might have been led: the phrase "conflates two
  predicates" is rhetorical, and it did not accept it on that basis. What
  convinced it were two things the framing did not supply — the crop's
  **mtime** (pre-existing, so the number was always wrong) and the **brief's
  own** requirement that `gaza.tiff` be source-but-not-indexable.

### AMENDMENT — D-1's *Expected* clause. Owner signed.

Ruled **genuinely wrong**, not merely unmet. Reasoning in `spec.md`'s amendment
block. The key point: the clause contradicts **D-1's own first sentence** on
real data, so no faithful implementation satisfies both.

**Nothing was weakened.** The spec now pins **two** exact sets (the 4-member
pixel-rule set and the 2-member indexable set) where it previously pinned one —
strictly more checkable. The "never by filename" requirement stays attached to
the pixel rule with the renamed-symlink assertion named, both conditions the
reviewer set for accepting.

### STRENGTHENED — D-2 gains a geodesy cross-check. Owner signed.

The analytic rule is over-general: `cos(lat)` is right only for Mercator with
standard parallel at the equator. EPSG:3994 is **24.5% wrong**,
`+proj=merc +lat_ts=45` **29.3%**, and Hotine Oblique Mercator is misclassified
and given a cosine it should not get. `spec.md` D-2 states the same
over-general rule, so the code faithfully implements the spec — a latent trap
of exactly the class trap 3 exists to prevent.

The fix is nearly free because `geo.py` **already computes a geodesic GSD for
every raster and throws it away**. The reviewer found it agreed with its own
independent ellipsoidal measurement in **100% of probes** across Web Mercator,
three UTM zones and geographic, to all printed digits. Real agreement is within
**0.17%**; failures diverge **24-29%**. One assertion catches every bad
projection, including ones nobody enumerated.

Chosen over generalising the formula because failing loudly is the honest
outcome for an unhandled projection, and the formula can be extended later if
such imagery ever arrives.

### SEND-BACK — two findings closed before later stages depend on them

Neither is a spec failure; both are latent defects sitting directly under
planned work.

1. **The duplicate content check verified 0.0005% of the raster.**
   `_same_pixels` compared 16 windows of 8x8 px = 1,024 px/band. The reviewer
   built two rasters differing in **99.5% of values** that agree at exactly
   those windows, and got `duplicate_of` — **a genuinely distinct scene
   silently dropped from the index.** Trigger is not exotic: same-footprint
   scenes with large nodata regions covering the sample points, and this tree
   holds scenes 25.1% and 14.2% zero-fill. **F-7 depends on this logic.**
   *The live case was verified safe:* all 16 windows differ between `leb`'s two
   dated scenes, so the multi-date pair survives — and the hazard the
   implementer described is real, since the pair does share identical bounds
   and a bounds-only rule would collapse it.
2. **The geographic-CRS branch had no test at all** — the **one mutant of 16
   that survived**. Replacing metres with degrees at `geo.py:193` still gave
   `21 passed`. Two EPSG:4326 rasters already flow through it, one of them
   **E-1's primary label raster**, whose GSD would read `0.00011 cm` instead of
   `12.17 cm` — a factor of ~111,000 — unnoticed. **E-1 depends on this path.**

Also in the send-back: use the discarded geodesy as D-2's guard, and three
trivia — `_bounds` is dead **and** wrong (anchors at `(0,0)` not the transform
origin), a docstring says 32,768 where the real value is 262,144, and
`S1_result.md` quotes full-raster distinct counts as if they were the recorded
sampled ones.

### The first fix attempt died, and its work was worse than nothing

Killed mid-Fix-1 by a spend limit. It had written **132 lines of test code and
no source change**, and **its tests passed against the unfixed code** — so its
"adversarial" fixtures were not adversarial. It never validated that its
fixture defeated the current implementation.

PM reverted them and verified the defect is live:
`_same_pixels(G2, G1) -> True` on rasters differing in 99.5% of values.

*Lesson, now written into the brief as a mandatory step:* **validate the
fixture before trusting the test.** Generate the pair, assert `_same_pixels`
returns `True` against the **unfixed** code, paste it — *that* is the RED — and
only then fix. Encoded permanently as a fixture self-check so the pair cannot
silently stop being adversarial when sampling constants change.

Its one good idea was kept: **two** adversarial pairs at different scales, one
agreeing on the entire primary sampling grid, so the test cannot be satisfied
by merely enlarging the sample.

Re-dispatched on a **cheaper model** to stay under the spend limit, with the
order changed to Fix 2 -> 3 -> 4 -> 1 so an interruption cannot cost
everything again.

### Manifest corrections (documentation, not spec)

`docs/DATA.md` and `CLAUDE.md` corrected on three measured claims, using the
reviewer's **dense** figures rather than sampled ones. Superseded text left
visible so nobody re-derives from it. Details in `docs/DATA.md`.

---

## s1-green — 2026-09-07 — S1 gated after the send-back

**S1 is GREEN.** All four fixes closed; **39 tests pass on the PM's own re-run.**

### PM verification — against the reviewer's fixtures, not the worker's

This distinction mattered: the *first* fix attempt produced tests that passed
against unfixed code, so worker-authored fixtures had already proved
untrustworthy once. Every check below uses either the adversarial reviewer's
original files or the real corpus.

| Check | Result |
|---|---|
| Reviewer's `G1`/`G2` (differ in 99.5% of values) | `_same_pixels -> False` — **defect fixed** |
| `leb_crop` vs `2022-10-29.tif` (true contained duplicate) | `True` — still caught |
| `leb/2022-10-29` vs `leb/2025-06-06` (the multi-date pair) | `False` — **not merged**, F-7 safe |
| `tile_cropped_…` (EPSG:4326, E-1's label raster) | 0.12167 m — **metres, not degrees** |
| `sin/sin_min.tiff` (EPSG:4326) | 4.08326 m — matches the manifest |
| Geodesy guard on EPSG:3994 | raises `GsdGuardError`, 24.6% > 1.5%, naming both figures and the CRS |
| Guard on the 8 indexable scenes | silent; all 8 present with correct GSD (0.1047 Mercator / 0.1000 UTM) |
| Inventory determinism | byte-identical across two fresh rebuilds |
| Data tree | untouched |

### What the fix actually did

`_same_pixels` now streams a **full** comparison of the overlap in 512-row
strips with early exit, keeping footprint containment as the cheap pre-filter.
**Cost was the open question and it is a non-issue:** full inventory 8.98 s ->
9.08 s, because genuinely different scenes diverge almost immediately — the
early exit does the work.

Two adversarial pairs guard it at different scales, so the test cannot be
satisfied by merely enlarging the sample. Notably the worker **caught its own
second pair being non-adversarial** (it matched the primary grid but not the
dedup grid the deployed code actually samples) and rebuilt it by taking the
union. That is precisely the discipline the previous attempt lacked, and it is
the reason to keep the fixture-validity step in every future brief.

Guard tolerance **1.5%**, justified from measurement rather than taste: real
agreement <= 0.17%, known failures 24-29%. Nothing lives in between.

### Cost of the review, and whether it was worth it

The adversarial review plus a send-back cost roughly one extra stage of
wall time and two dispatches. It bought: a silently-dropped-scene bug that
would have surfaced as a corrupted date filter at F-7; a 111,000x GSD error
latent under E-1; three false "measured fact" claims in the manifest; and a
spec criterion that contradicted its own rule. **Keep the mandatory reviewer
for S3 and S4** — `CLAUDE.md` already marks them as the high-risk stages, and
this is the evidence for why.

### Schema note for later stages

`index/inventory.json` is **schema_version 2**. Dedup metadata changed from
`dedup_grid`/`dedup_window_px` to `dedup_strategy`/`dedup_strip_rows`, and
records now carry `geodesic_gsd_cm`. Size 235,707 -> 235,747 B. Anything
reading the inventory should check `schema_version`.

---

## s1a-green — 2026-09-07 — portability landed; the fp16 rule generalised

**S1a is GREEN.** N-8 met, **44 tests green on the PM's own re-run**, and the
setup half of `INSTRUCTIONS.md` exists.

### The substantive change

`src/embedders.py` had `DTYPE = torch.float16` as a module constant. It is now
`select_dtype(device, capability=None)` in `src/device.py`:

| device | capability | dtype |
|---|---|---|
| CUDA | cc < 8.0 (Turing, Volta) | `float16` |
| CUDA | cc >= 8.0 (Ampere and later) | `bfloat16` |
| CPU / MPS | — | `float32` |

PM verified all six branches with **fabricated** capability tuples, so the
Ampere path is tested without Ampere hardware — and confirmed this machine
(cc 7.5) still resolves to `float16`, keeping N-3 green.

**This is the important conceptual move in the stage.** `CLAUDE.md` trap 2
says "fp16, never bf16" and calls model cards recommending bf16 *wrong*. That
was true and is still true — **for Turing**. It was never a fact about the
project; it was a fact about the hardware. Stated as a universal it would have
shipped the *wrong* dtype to the owner's next GPU, where bf16 is native and
better. The trap is not repealed, it is demoted to a case.

Paths likewise: `AERIAL_DATA_ROOT` (required, **fails loudly with no
machine-specific fallback**), `AERIAL_INDEX_ROOT` (defaults relative to the
repo), `AERIAL_MODEL_CACHE_ROOT`, `AERIAL_DEVICE`.

### A latent Windows bug fixed in passing

`str(Path.relative_to(...))` emits **backslash-separated** keys on Windows, so
an index built there would have had non-portable manifest keys and tile ids
that silently differ from a Linux-built index. Now routed through
`posix_key()`. Nobody was looking for this; it fell out of taking N-8
seriously rather than writing a document about it. **Vindicates doing S1a as a
stage instead of as prose** — this class of bug is invisible to a
documentation pass.

### CPU-only is real, and honestly slow

Measured **~5 tiles/sec** on CPU against **265 tiles/sec** on GPU 0 — ~50x.
Recorded in `INSTRUCTIONS.md` so nobody plans a full index build on CPU by
accident: the 108,542-tile pyramid would be ~6 hours instead of 6.8 minutes.

### PM found one real defect in INSTRUCTIONS.md

§2 claimed `perception_models`' imports were "all already satisfied by the
install above". Checked against `open_clip_torch` 3.3.0's **declared
requirements** rather than against this machine's populated env — it pulls
`torch`, `torchvision`, `regex`, `ftfy`, `tqdm`, `huggingface-hub`,
`safetensors`, `timm>=1.0.17` transitively, but **not `einops`**. A fresh
machine following the document and then taking the optional fallback path
would have failed on a missing import. Fixed: `einops` named explicitly, and
the transitive set stated so the claim is checkable rather than asserted.

Also recorded there: `perception_models` pins `timm==1.0.15` while
`open_clip_torch` needs `timm>=1.0.17`. `--no-deps` is what keeps those from
fighting — it is not a tidiness measure.

*Method note:* the only way to catch this was to check declared metadata, not
to run the import on a machine where everything is already present. **A
portability claim cannot be verified on the machine it was written on.** That
is why N-9 requires execution on a foreign machine at S9, and this is early
evidence the requirement is right.

---

## s2-green — 2026-09-07 — tiling and COGs; one incident, one stale artifact

**S2 is GREEN.** 79 tests pass on the PM's own re-run.

### INCIDENT — a test wrote under the read-only data root

**What happened.** While producing a RED for D-3, an early sabotage test
**created a directory directly under
`/home/omer/PycharmProjects/Dynamic-Terrain/data`**. The worker caught it,
deleted it, rewrote the test against a scratch `tmp_path`, and **disclosed it
unprompted** in the handback.

**PM forensic verification — not taken on the worker's word:**

| Check | Result |
|---|---|
| `tree_digest()` recomputed vs the value recorded before S2 ran | **identical** — `sha256:7cacf2fe…` |
| Anything in the tree modified today | **only the root directory's own mtime** (21:39). Every subdirectory still dated March-August |
| Empty or stray directories | none |
| `.aux.xml` sidecars | 26, **all dated April-August** — none created by us |
| Lock/tmp/backup files | none |

**Impact: none.** No file was written, modified, moved or lost. The only
residue is the root inode's mtime, which a create+delete cannot avoid and which
cannot be reverted without a further write — so it stays, documented.

**Root cause is the brief, not just the worker.** For D-3 specifically, the
RED-then-GREEN requirement and the read-only rule **pull in opposite
directions**: the natural way to prove "we never write outside `index/`" fails
when sabotaged is to attempt a write outside `index/`. My brief demanded the
RED and did not say where to aim it. The worker independently reached the right
fix; the brief should have specified it.

*Standing rule added to `CLAUDE.md`:* a D-3 RED must target a **scratch
stand-in configured as the data root**, never the real one.

**Not a send-back.** The deliverable is correct, impact is nil, and it was
disclosed rather than concealed. Punishing voluntary disclosure of a
remediated, zero-impact error is how a project teaches concealment — which
would cost far more than this did.

### The handback was true and the artifact was still wrong

The handback reported all F-1 counts matching. The **tests** did assert
5,198 / 20,704 / 82,640 and pass. But the **emitted plans on disk carried only
scale 448** — 5,198 tiles total, and the demo AOI at 529 instead of 11,109.
S3 consumes those files, not the tests.

Cause was benign: stale output from an earlier partial run, never regenerated.
`SCALES = (448, 224, 112)` and `plan_all()` were correct all along. PM
regenerated via the module's own entry point; it took **~1 second** (pure
metadata, as designed) and produced exactly 108,542 / 11,109 / 1,012.

*Lesson, now standard practice:* **verify the emitted artifact, not only the
test that computed it.** A green suite says the code is right; it does not say
the file on disk was produced by that code. This is the second time in this
project that checking the artifact rather than the report changed the verdict.

### Results

COG conversion verified independently: metadata preserved, **six random 448²
windows byte-identical**, block layout `(1, 20179)` strips -> `(256, 256)`
tiles, **7 overview levels** added. Read amplification 45.04x -> 1.31-2.94x,
which is the entire justification for the stage and is now measured rather than
asserted.

Per-scene planned/valid/nodata is emitted at every scale — e.g.
`X605_Y3388` 529/410/14.2% at 448, 8464/7091/14.2% at 112.

### Two findings accepted

- **UTM lon/lat tile bboxes are not bit-exact at shared edges** (~0.26 m worst
  case) — meridian convergence, not a coverage gap; the worker proved coverage
  separately in projected space, which is the correct place to prove it.
  Mercator tiles are bit-exact. Consequence for F-6: an AOI bbox filter on UTM
  scenes must not assume bit-exact edge adjacency.
- **`X605_Y3388` valid count is 410, not `docs/DATA.md`'s 397** — measured
  under a *stated* >50%-nodata rule against recon's unstated one. Seven of eight
  scenes match exactly. The measured figure is authoritative; `DATA.md`
  reconciled, and its "~4,146" total withdrawn (the column sums to 4,575,
  measured 4,588).

---

## s3-review — 2026-09-07 — retrieval works; three defects closed; N-4 amended

**S3 is GREEN after one send-back.** The demo AOI is indexed: 11,109 planned,
1,478 skipped as 100% nodata, **9,631 vectors** at 242 tiles/sec in 39.8 s,
14.8 MB.

### The result that matters

**The PM ran a real text query against the finished index** — before S4 exists —
because if tile ids were misaligned with vectors, nothing but an end-to-end
query would reveal it.

- **`a car` at 112 px: 5/5 genuine vehicles** out of 9,631 tiles. Better than
  S0's 4/5, and on the full AOI rather than 336 calibration crops.
- **`dirt road` at 448 px:** real roads as linear features — which an 11.2 m
  crop physically cannot contain. **The owner's per-scale ranking decision is
  now validated empirically, not just argued.**
- **`aircraft carrier`: no carriers.** It reaches for large deck-like roofs and
  the shoreline surf — the nearest things present — topping out at 0.2332
  against `a car`'s 0.2650. A workable margin for the relative-gap empty state.

Two scares, both resolved and worth recording:

- **`No pretrained weights loaded ... initialized randomly`** prints on every
  load. If the checkpoint had silently failed, every vector would come from a
  random network and everything would still look numerically fine — this
  project's recurring failure mode. **Two independent processes produce
  byte-identical embeddings**, so the weights are real; `open_clip` builds a
  random skeleton before RemoteCLIP loads over it. *Suppress or annotate this
  warning before M1* — it reads like a failure and is not.
- **The first crops rendered solid black** while the manifest reported zero
  nodata. The tile id encodes **grid indices, not pixel offsets** (`col 0..22`
  for a 23x23 grid). The PM's parse was wrong, not the index — but nothing in
  the manifest states the convention, so **S4 must join to the tile plan for
  geo rather than interpreting ids itself** (F-10).

### The fp16 question — settled

F-2's `||v|| = 1.0 +/- 1e-5` **cannot** hold for raw fp16 storage. Verified
independently, twice over: for *random* 768-d unit vectors an fp16 round trip
puts **43.4%** outside 1e-5; for the real embeddings, **89.8%** (mean 6.38e-5,
max 2.63e-4). Renormalised on load, max deviation is **1.19e-7**.

**The PM's brief caused this** — it demanded the norm hold *after* the fp16
round trip, which is stricter than the spec and mathematically impossible.
F-2 itself is satisfiable and satisfied for every vector the system queries.

The reviewer upheld renormalise-on-load and, valuably, showed it is
**load-bearing rather than cosmetic**: raw stored norms vary 0.9997-1.0003, so
an un-renormalised dot product is not a valid cosine *across* tiles. Ranking
genuinely changed on 1 of 5 real queries in the top-10 and all 5 in the top-50.

### Review findings — 3 fixed, 1 ledgered

The reviewer tried to defeat the mixed-embedder guard (four ways), kill-and-
resume, tile identity, nodata correctness and cross-process determinism, and
**could not**. It broke the *safety net* instead:

1. **NaN silently bypassed the sanity check.** `np.abs(nan - 1.0) > ATOL` is
   `False`, so a NaN vector was never flagged and `load_index` divided by it,
   returning NaN in every element as a "successfully loaded" unit vector with
   **no exception**. Not a threshold miss — the comparison's domain was
   defeated. **Fixed:** non-finite detection now runs before the bound; NaN and
   Inf both raise, naming rows. PM verified.
2. **`build_index` reported success without validating resumed state** — a
   manifest listing 16 tiles against 10 stored vectors returned
   `{"embedded_total": 16}`. The write-ordering argument does hold (the reviewer
   could not break it with real kills) but **nothing asserted it at runtime**.
   **Fixed:** consistency checked on load, raising with both counts.
3. **The bound was asserted in three places and tested in none** — disabling it
   (`ATOL = 1e9`) left all 6 tests passing. **Fixed:** gross-norm and
   noise-tolerance tests, proven by mutation.
4. **Subtle corruption (~3e-4 to 1e-2) is undetectable** and gets laundered by
   renormalisation. **Ledgered with owner sign-off** — untightenable by
   construction, and the index rebuilds in 40 s.

### AMENDMENT — N-4 gains its invariance condition. Owner signed.

Cross-**process** determinism is **exact (0.000e+00)**. But the same tile alone
vs in a batch of 32 differs by **2.574e-4** — 25x the tolerance — from GPU fp16
GEMM non-associativity. Not fixable in our code.

Amended to state the condition and require the manifest to record `batch_size`
(now **256**; the index was rebuilt so the shipped artifact satisfies the
amended spec rather than merely the old one).

Rejected: widening to 1e-3, which would loosen the bar for the case that passes
*exactly*; and batch size 1, which buys unconditional truth at ~50x cost and no
retrieval benefit.

**The sharpest point the reviewer made:** neither shipped test *could* detect
this, because both hold batch composition constant by construction. The
criterion was passing for a **structural reason rather than because the property
held**. That is a category of false green worth watching for elsewhere.

---

## s4-review — 2026-09-08 — retrieval core; U-3 proved unsatisfiable; export budget re-chosen

S4 returned `DONE_WITH_CONCERNS`. Sent back on three items. Two owner decisions
taken.

### What S4 got right

Per-scale ranking, F-3/F-5/F-6, **CPU latency 9-18 ms against a 200 ms budget**
(~15x margin), reload 0.13 s against 5 s.

And one genuinely good catch: the first PCA implementation scored 0.00-0.10
overlap, and the worker correctly diagnosed **why** rather than accepting it as
a finding — this corpus's **mean vector has norm 0.906**, so textbook
mean-centering discards signal rather than noise. Switching to an uncentered
second-moment PCA took it to ~0.71. A worse worker would have reported "PCA
export destroys retrieval" as a measured result.

### FINDING — U-3's automatic empty state is not achievable. AMENDED.

S4 reported that U-3's literal wording failed and substituted a different
statistic, calibrated to the five values the PM had supplied. **The PM tested
that substitute on unseen queries and it does not generalise:**

- 112 px: misses `submarine`, `a ski slope`, `penguins` (absent, called strong).
- 448 px: flags `sand` and `a building with a flat roof` — **both present** — as
  weak, and misses three absent queries.

So the PM asked the prior question: can **any** relative statistic do this? Over
8 present and 8 absent queries on the real index:

| statistic | present range | absent range | separable? |
|---|---|---|---|
| `top1` | 0.257-0.299 | 0.210-0.277 | no |
| `z_mean` | 1.619-4.607 | 2.685-4.469 | no |
| `gap_top10` | 0.126-0.684 | 0.294-**0.812** | no |
| `gap_top50` | 0.236-1.697 | 0.629-1.473 | no |
| `gap_top100` | 0.307-2.149 | 0.895-1.884 | no |
| `top10_z` | 1.553-4.222 | 2.495-3.946 | no |
| `skew` | -0.948-0.668 | -0.881-0.984 | no |
| `n_within_1pct` | 1-6 | 1-4 | no |

**Every one overlaps.** Raw scores overlap too: `a snowy mountain` (absent)
0.2769 beats `a car` (present) 0.2650.

**The isolation statistics run backwards**, which is the deep reason no boolean
can work. `gap_top10` reaches **0.812 for absent** queries against **0.684 for
present** ones: an abundant class like `tents` has many near-equal matches and
therefore a *small* gap, while `penguins` retrieves a handful of odd tiles that
stand alone and therefore a *large* one. **Isolation measures novelty, not
presence.** Any "does the top hit stand out" heuristic is measuring the wrong
thing.

*Owner decision:* replace the boolean with a **calibrated confidence band**
against a stored, seeded background set of ~30 unrelated queries, per scale,
shown alongside results and never gating them — worded **"may not be present"**,
never "is not present".

*Why this is not a weakening:* it swaps an unachievable binary for an achievable
and more informative display, and adds two requirements (reproducible stored
background; mandated wording). **A large present/absent overlap in the bands is
the correct outcome and must not be tuned away.**

*And it is itself a product finding.* "This embedder cannot reliably tell you
when a thing is absent" is exactly what a scoping demo exists to surface. Hiding
it behind a confident binary would corrupt the decision the demo informs.

### The owner's third question, answered — and the default changed

**Retrieval@10 overlap, full precision vs export, measured by the PM over 8
queries on the real index:**

| dim | dtype | MB | 448 | 224 | 112 | mean |
|---|---|---|---|---|---|---|
| 128 | int8 | 1.23 | 0.738 | 0.750 | 0.650 | **0.713** *(old default)* |
| 256 | int8 | 2.47 | 0.838 | 0.825 | 0.738 | 0.800 |
| **384** | **int8** | **3.70** | 0.850 | 0.838 | 0.812 | **0.833** *(chosen)* |
| 384 | fp16 | 7.40 | 0.875 | 0.900 | 0.838 | 0.871 |
| 768 | fp16 | 14.79 | 1.000 | 1.000 | 1.000 | **1.000** |

**So yes — PCA-to-128 costs measurable quality: ~29% of top-10 results change.**

Two things the curve makes clear that a single number would not:

1. **768-d fp16 is exactly 1.000** — lossless by construction, since the local
   index is already fp16. So the loss is **dimensionality reduction**, with int8
   adding a few points on top; it is not an artifact of quantisation alone.
2. **128 was the worst point on the curve** and was inherited from a budget
   estimate made when the embedder was 1280-d PE-Core. At 768-d the sums changed
   and nobody had rechecked. *Lesson: a config default set under superseded
   assumptions is not a decision, it is a leftover.*

*Owner chose 384-d int8* — 0.833 at 3.70 MB, best quality per megabyte, leaving
~12 MB of the 16 MB cap for thumbnails, which dominate the budget at 9,631
tiles.

### DEFECT — the test suite was destroying a production artifact

`tests/test_tiling.py` writes to the **real** index root in three places, one of
them `plan_all(scales=(448,))`, which **truncates `index/tileplan/*.json` to
scale 448**. The PM regenerated the full three-scale plan after S2; running the
suite silently destroyed it again.

S4 then hit the damage, correctly diagnosed it as a test side effect — and
**worked around it** by re-planning in memory rather than flagging it. That is
the wrong call: it treated a self-inflicted wound as an external constraint, and
it left F-10's map-location join unproven against the artifact that actually
ships. Sent back to fix the tests and prove the join from the real files.

*Standing lesson, third instance in this project:* **check the artifact, not the
report.** A green suite says the code is right; it says nothing about what is on
disk — and here the suite was itself the thing corrupting the disk.

---

## s4-green — 2026-09-08 — S4 gated; and a hard limit found for S5

**S4 is GREEN after one send-back.** 116 tests on the PM's own run.

The acceptance criterion for the tileplan fix was deliberately **not** "the test
passes" but "the artifact survives" — verified by running the full suite and
then re-reading the files: **108,542 tiles, all three scales, all eight
scenes.** The suite no longer destroys its own inputs.

Export rebuilt at **384-d int8, 3.70 MB**, with a stored background query set.
Re-measured overlap ~0.82 mean, matching the PM's ~0.833 within query-set noise.
Latency 9-18 ms against a 200 ms budget; reload 0.13 s against 5 s.

The fix round also turned up two real bugs while proving RED: `confidence_band`
raised on numpy input, and `build_background`'s in-memory return used string
keys where the reloaded copy used ints — silently breaking same-process use.
Both the kind of thing that only surfaces when you actually exercise the path.

### BLOCKER for S5 — a standalone file cannot embed typed text

F-8 requires the export to open and query "with no server and no network".
Answering a typed query means running the text tower. **Measured:**

| | |
|---|---|
| RemoteCLIP total | 427.6 M params |
| **text tower alone** | **123.7 M params** |
| as fp16 | **247.3 MB** |
| as int8 | 123.7 MB |
| F-8 cap, *including thumbnails* | **16 MB** |

**The text tower is 15.5x the entire budget in fp16, and 7.7x in int8.** No
compression closes that gap; this is not a tuning problem.

This is the same family as the F-8/U-8 conflict logged at S2 (tile Q&A needs a
VLM, equally impossible offline). Both come from one root: **F-8's offline
guarantee and the interactive premise are in tension, and the spec never
resolved which wins.**

Notably, `notes.md#intake` decision 3 already separated "full local index" from
"the HTML viewer is a per-AOI export" — the two-component architecture was
always there. What was never stated is that **free-text querying lives on the
local side only**. Put to the owner rather than decided by the PM, because it
determines what "explore the demo" means.

---

## s5-green — 2026-09-08 — **M1 REACHED.** The local app works.

**135 tests pass** on the PM's own run. Tile plan intact at 108,542 across three
scales after the suite; embedding index untouched; data-root digest unchanged.

### PM verification — ran it, not read about it

Launched the app against the production index and used it as an impatient user.

**Consistency check that mattered most:** `a car` returns **+0.2650** at 112 px
*through the API* — byte-identical to the score the PM measured directly against
the index before S4 existed. The app is not quietly re-deriving anything.

| abuse case | result |
|---|---|
| `a car` (first query) | 200, **7.9 s** — the lazy model load |
| every subsequent query | 200, **20-70 ms** |
| empty / whitespace | guiding empty state, no error |
| 500 characters | 200, ranked results |
| Hebrew | 200, ranked results |
| nonsense string | 200, ranked results |
| `penguins` (known-absent) | 200, results shown **with** a band — correct under the amended U-3 |
| bbox matching nothing | 200, **zero results, no raise** (F-6) |
| malformed bbox / malformed JSON | 422, handled |
| **tracebacks or 500s anywhere** | **zero** |

The rendered page carries U-5's caveat prominently and specifically ("35-45 cm,
not the 10 cm pixel grid... `a car` and `a white car` retrieve the same crops,
and the colour word makes results measurably worse... does not do
attribute-level search"). Each scale row states **"scores are only comparable
within this row"**. The confidence chip reads **"not a presence guarantee"**.
Date control correctly reports "No dated imagery indexed for this AOI" rather
than offering a filter that cannot work.

### Two items carried, neither blocking

- **The tile modal pushes its metadata below a scrollbar.** Polish, not a
  defect; the enlarged crop is the point and it works.
- **U-8 is partially met at M1 *by design*.** The larger view ships at S5; the
  question box ships at **S5a** with F-9. The worker flagged that the brief was
  narrower than `spec.md`'s own U-8 amendment rather than silently resolving it
  — exactly right, and the spec now records the split so it reads as sequencing
  rather than an unmet criterion.

### Ledgered

Two S4 tests rebuild the **cheap** `index/export/` derivatives (PCA basis,
background set) against the real index by design. Same family as the tileplan
defect but materially different: deterministic, seconds, and it regenerates
rather than truncates. The expensive `index/emb/` pass is untouched. Recorded
rather than fixed under M1 pressure.

### What M1 actually demonstrates

A user types a description and gets back the aerial tiles that match it, ranked
by cosine similarity, with map locations, in tens of milliseconds — on real
imagery, with nothing trained. The vehicle result is the one that was genuinely
in doubt: **5/5 at 112 px, and 0/5 at 448 px for every candidate tested.** The
multi-scale pyramid is not a refinement of this system; it is the reason it
works at all.

---

## s6-s5b — 2026-09-08 — full index, shareable export; three agents lost to a spend limit

### S6 — all 8 scenes indexed. PM-verified.

| check | result |
|---|---|
| embedded + skipped | **104,374 + 4,168 = 108,542** — reconciles exactly |
| `leb` by date | **20,944 / 20,944** — F-7 halves the pool *exactly*, not approximately |
| N-1 end-to-end at full scale | **65.7 ms mean, 104.5 ms max** vs a 200 ms budget |
| N-1 raw dot product (PM) | 6.2 ms mean, 10.7 ms max — measures a different thing; quote the end-to-end figure |
| throughput | **237 tiles/sec**, consistent with S3's 242 |
| peak RSS, full index | **1,419.6 MiB** (vectors alone are 321 MB fp32) |
| corpus load | inside F-5's 5 s |

The throughput figure is worth keeping: at ~5 tiles/sec the CPU-shadow trap would
have turned a 7-minute build into six hours **while looking fine throughout**.
237 tiles/sec is the evidence that `PYTHONNOUSERSITE=1` held on every one of
those invocations, not an assumption that it did.

*The destruction pair is now reachable* — `leb` 2022-10-29 (intact village core)
against 2025-06-06 (the same pixels as rubble). That is the most compelling
content in the corpus and it was unreachable an hour ago.

### S5b — the shareable export exists

**`index/export/X605_Y3388/export.html`, 10.13 MB against the 16 MB cap**, with
**zero external references** (no CDN, no font, no analytics — grep finds no
`http(s)://` at all). One embedded basemap JPEG rather than 9,631 thumbnails,
cropped client-side from grid indices: the design decision that made the budget
fit with 6 MB to spare.

U-5's caveat survived into the artifact, which is the thing that mattered most
given this file reaches people who cannot ask what it means:

> *"...effective optical resolution is approximately 35-45 cm on the ground (the
> 10-12 cm pixel grid is upsampled 3-4x). The system can indicate presence and a
> coarse class of object — it cannot support fine attribute search (colour,
> make, model, exact count). Treat every result as 'something like this may be
> here', not a confirmed identification."*

*PM note on method:* a first grep for the caveat returned zero and looked like a
missing requirement. It was **HTML-entity encoding** (`&ndash;`) defeating the
pattern, not an absent caveat. Worth recording as a near-miss: a negative grep
result is evidence about the pattern before it is evidence about the artifact.

### Three agents died on a spend limit, mid-stage

S6 (after its work landed), S5b (after building the export and passing its 12
tests, before the size table and full suite) and the polish pass (before doing
anything). The PM completed verification directly rather than re-dispatching.

**Unfinished and carried:** S5b's handback, its export/local ranking-parity
check, and the whole polish pass (modal metadata, `pyproj` startup warning,
S4 tests touching `index/export/`, the measured-RSS note in `INSTRUCTIONS.md`).
None blocks use; all are recorded rather than quietly dropped.
