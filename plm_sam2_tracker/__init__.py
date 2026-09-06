"""PLM + SAM 2 aerial object tracking demo pipeline.

Two backends share one output format (tracks.json + HTML viewer):

- mock: stdlib-only synthetic aerial scene, for playing with the pipeline
  and viewer without GPUs or model downloads.
- real: PerceptionLM grounding + SAM 2 mask propagation on actual video
  (requires torch, transformers, sam2, opencv; see requirements.txt).
"""

__version__ = "0.1.0"
