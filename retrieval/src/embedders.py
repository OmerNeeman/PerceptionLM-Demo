"""One loader interface over the S0 embedder candidates.

Contract (brief S0, spec F-1a / F-2 / F-3 / N-3):

    emb = load_embedder("PE-Core-L14-336")
    emb.embed_images([pil, ...]) -> np.ndarray (n, emb.dim) float32, L2-normalised
    emb.embed_texts(["tent", ...]) -> np.ndarray (n, emb.dim) float32, L2-normalised

Every candidate is off-the-shelf. Nothing here trains or fine-tunes anything.
Retrieval is on the **pooled** output only -- PE-Core's patch tokens never
entered the contrastive objective (see notes.md#research-1).

Environment, non-negotiable (CLAUDE.md traps 1 and 2):

    env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
        /home/omer/anaconda3/envs/geo/bin/python ...

fp16 always, bf16 never: these are Turing cards (cc 7.5) where bf16 is
emulated at 7.0 TFLOP/s against 38.9 for fp16.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# fp16, never bf16 -- CLAUDE.md trap 2.
DTYPE = torch.float16

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
            batch = batch.to(self.device, dtype=DTYPE, non_blocking=False)
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


def _load_pe(spec: CandidateSpec, device: torch.device) -> Embedder:
    import core.vision_encoder.pe as pe
    from core.vision_encoder.config import PE_TEXT_CONFIG
    from core.vision_encoder.transforms import (
        get_image_transform,
        get_text_tokenizer,
    )

    model = pe.CLIP.from_config(spec.id, pretrained=True)
    model = model.to(device=device, dtype=DTYPE).eval()

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


def _load_open_clip(spec: CandidateSpec, device: torch.device) -> Embedder:
    import open_clip
    from huggingface_hub import hf_hub_download

    assert spec.open_clip_arch is not None
    # pretrained=None -> random init, no OpenAI download; the checkpoint
    # below replaces every weight.
    model, _, preprocess = open_clip.create_model_and_transforms(
        spec.open_clip_arch, pretrained=None
    )
    ckpt_path = hf_hub_download(spec.hf_repo, spec.hf_file)
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
    model = model.to(device=device, dtype=DTYPE).eval()

    tokenizer = open_clip.get_tokenizer(spec.open_clip_arch)
    return Embedder(
        spec=spec,
        model=model,
        preprocess=preprocess,
        tokenize=lambda ts: tokenizer(list(ts)),
        revision=_resolve_revision(spec),
        device=device,
    )


def load_embedder(candidate_id: str) -> Embedder:
    """Load one candidate onto GPU 0 in fp16."""
    if candidate_id not in CANDIDATES:
        raise KeyError(
            f"unknown candidate {candidate_id!r}; "
            f"known: {sorted(CANDIDATES)}"
        )
    spec = CANDIDATES[candidate_id]
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. This is CLAUDE.md trap 1: a "
            "CPU-only torch in ~/.local shadows the CUDA build. Re-run with "
            "PYTHONNOUSERSITE=1."
        )
    _seed_everything()
    device = torch.device("cuda")
    log.info("loading %s (%s) in %s", spec.id, spec.loader, DTYPE)
    if spec.loader == "pe":
        return _load_pe(spec, device)
    if spec.loader == "open_clip":
        return _load_open_clip(spec, device)
    raise RuntimeError(f"no loader {spec.loader!r}")
