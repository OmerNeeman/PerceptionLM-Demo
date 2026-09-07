"""One loader interface over the S0 embedder candidates.

Contract (brief S0, spec F-1a / F-2 / F-3 / N-3):

    emb = load_embedder("PE-Core-L14-336")
    emb.embed_images([pil, ...]) -> np.ndarray (n, emb.dim) float32, L2-normalised
    emb.embed_texts(["tent", ...]) -> np.ndarray (n, emb.dim) float32, L2-normalised

Every candidate is off-the-shelf. Nothing here trains or fine-tunes anything.
Retrieval is on the **pooled** output only -- PE-Core's patch tokens never
entered the contrastive objective (see notes.md#research-1).

Environment, non-negotiable (CLAUDE.md traps 1 and 2):

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 <python> ...

(see INSTRUCTIONS.md for the interpreter path on this dev machine)

dtype is device-dependent, not hardcoded (spec N-8): see `device.py`. On this
project's Turing cards (cc 7.5) that resolves to fp16, never bf16 -- bf16 is
emulated there at 7.0 TFLOP/s against 38.9 for fp16 -- but the rule reaches
the opposite conclusion on Ampere and later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import torch
from PIL import Image

import config
from device import resolve_device, select_dtype

log = logging.getLogger(__name__)

# Fixed seeds and deterministic kernels (spec N-4, F-3).
SEED = 0


@dataclass(frozen=True)
class CandidateSpec:
    """Static, checked-in facts about one candidate."""

    id: str
    dim: int
    image_size: int
    hf_repo: str
    hf_file: str
    loader: str  # "pe" | "open_clip"
    open_clip_arch: str | None = None


CANDIDATES: dict[str, CandidateSpec] = {
    "PE-Core-G14-448": CandidateSpec(
        id="PE-Core-G14-448",
        dim=1280,
        image_size=448,
        hf_repo="facebook/PE-Core-G14-448",
        hf_file="PE-Core-G14-448.pt",
        loader="pe",
    ),
    "PE-Core-L14-336": CandidateSpec(
        id="PE-Core-L14-336",
        dim=1024,
        image_size=336,
        hf_repo="facebook/PE-Core-L14-336",
        hf_file="PE-Core-L14-336.pt",
        loader="pe",
    ),
    "RemoteCLIP-ViT-L-14": CandidateSpec(
        id="RemoteCLIP-ViT-L-14",
        dim=768,
        image_size=224,
        hf_repo="chendelong/RemoteCLIP",
        hf_file="RemoteCLIP-ViT-L-14.pt",
        loader="open_clip",
        open_clip_arch="ViT-L-14",
    ),
}


def _seed_everything() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Embedder:
    """A loaded candidate. Two public methods, per the brief."""

    def __init__(
        self,
        spec: CandidateSpec,
        model: torch.nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
        tokenize: Callable[[Sequence[str]], torch.Tensor],
        revision: str,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.model = model
        self.model_id = spec.id
        self.dim = spec.dim
        self.image_size = spec.image_size
        self.revision = revision
        self.device = device
        self._preprocess = preprocess
        self._tokenize = tokenize

    # -- introspection the acceptance tests need (N-3) ------------------

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the model's parameters as actually loaded."""
        return next(self.model.parameters()).dtype

    # -- the interface --------------------------------------------------

    @torch.inference_mode()
    def embed_images(
        self, images: List[Image.Image], batch_size: int = 32
    ) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            batch = torch.stack([self._preprocess(im) for im in chunk])
            batch = batch.to(self.device, dtype=self.dtype, non_blocking=False)
            feats = self.model.encode_image(batch)
            out.append(self._normalise(feats))
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            tokens = self._tokenize(chunk).to(self.device)
            feats = self.model.encode_text(tokens)
            out.append(self._normalise(feats))
        return np.concatenate(out, axis=0)

    # -- internals ------------------------------------------------------

    def _normalise(self, feats: torch.Tensor) -> np.ndarray:
        """Cast to fp32 *before* normalising, then L2-normalise (F-2).

        The forward pass is fp16; the norm is not. An fp16 norm cannot hit
        1.0 +/- 1e-5 because fp16 eps is 9.8e-4.
        """
        v = feats.float()
        v = v / v.norm(dim=-1, keepdim=True)
        arr = v.cpu().numpy().astype(np.float32)
        if arr.shape[-1] != self.dim:
            raise RuntimeError(
                f"{self.model_id}: expected dim {self.dim}, got {arr.shape[-1]}"
            )
        return arr


def _resolve_revision(spec: CandidateSpec) -> str:
    from huggingface_hub import HfApi

    try:
        return HfApi().model_info(spec.hf_repo).sha
    except Exception as exc:  # offline: recorded as unknown, never silent
        log.warning("could not resolve revision for %s: %s", spec.hf_repo, exc)
        return "unknown"


def _load_pe(spec: CandidateSpec, device: torch.device, dtype: torch.dtype) -> Embedder:
    import core.vision_encoder.pe as pe
    from core.vision_encoder.config import PE_TEXT_CONFIG
    from core.vision_encoder.transforms import (
        get_image_transform,
        get_text_tokenizer,
    )

    model = pe.CLIP.from_config(spec.id, pretrained=True)
    model = model.to(device=device, dtype=dtype).eval()

    preprocess = get_image_transform(model.image_size)
    ctx = PE_TEXT_CONFIG[spec.id].context_length
    tok = get_text_tokenizer(ctx)
    return Embedder(
        spec=spec,
        model=model,
        preprocess=preprocess,
        tokenize=lambda ts: tok(list(ts)),
        revision=_resolve_revision(spec),
        device=device,
    )


def _load_open_clip(spec: CandidateSpec, device: torch.device, dtype: torch.dtype) -> Embedder:
    import open_clip
    from huggingface_hub import hf_hub_download

    assert spec.open_clip_arch is not None
    # pretrained=None -> random init, no OpenAI download; the checkpoint
    # below replaces every weight.
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.open_clip_arch, pretrained=None
    )
    cache_dir = config.get_model_cache_root()
    ckpt_path = hf_hub_download(
        spec.hf_repo, spec.hf_file, cache_dir=str(cache_dir) if cache_dir else None
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Loud, not silent: a wholesale mismatch means random weights.
    log.info(
        "%s: loaded checkpoint, missing=%d unexpected=%d",
        spec.id,
        len(missing),
        len(unexpected),
    )
    real_missing = [k for k in missing if not k.endswith("logit_scale")]
    if real_missing:
        raise RuntimeError(
            f"{spec.id}: {len(real_missing)} parameters not in checkpoint, "
            f"first few: {real_missing[:5]}"
        )
    model = model.to(device=device, dtype=dtype).eval()

    tokenizer = open_clip.get_tokenizer(spec.open_clip_arch)
    return Embedder(
        spec=spec,
        model=model,
        preprocess=preprocess,
        tokenize=lambda ts: tokenizer(list(ts)),
        revision=_resolve_revision(spec),
        device=device,
    )


def load_embedder(candidate_id: str, device: str | torch.device | None = None) -> Embedder:
    """Load one candidate.

    `device` (or AERIAL_DEVICE, see config.py) forces the compute device --
    pass e.g. "cpu" to deliberately run without a GPU (N-8). Left as None
    (the default, and unchanged from pre-N-8 behaviour), this requires CUDA
    to be visible: `torch.cuda.is_available() is False` here is CLAUDE.md
    trap 1 (a shadowed CPU-only torch), not "no GPU present", so it fails
    loudly rather than silently degrading -- degrading is only correct when
    a caller explicitly asked for it.

    dtype is never hardcoded: it is `device.select_dtype(device)` (N-8), so
    it tracks whatever device this call resolves to.
    """
    if candidate_id not in CANDIDATES:
        raise KeyError(
            f"unknown candidate {candidate_id!r}; "
            f"known: {sorted(CANDIDATES)}"
        )
    spec = CANDIDATES[candidate_id]

    forced = device if device is not None else config.get_forced_device()
    if forced is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False. This is CLAUDE.md trap 1: a "
                "CPU-only torch in ~/.local shadows the CUDA build. Re-run with "
                "PYTHONNOUSERSITE=1. To deliberately run without a GPU, pass "
                "device='cpu' to load_embedder(...) or set AERIAL_DEVICE=cpu."
            )
        resolved_device = torch.device("cuda")
    else:
        resolved_device = resolve_device(forced)

    dtype = select_dtype(resolved_device)
    _seed_everything()
    log.info("loading %s (%s) on %s in %s", spec.id, spec.loader, resolved_device, dtype)
    if spec.loader == "pe":
        return _load_pe(spec, resolved_device, dtype)
    if spec.loader == "open_clip":
        return _load_open_clip(spec, resolved_device, dtype)
    raise RuntimeError(f"no loader {spec.loader!r}")
