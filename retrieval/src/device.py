"""Device selection and the dtype rule (spec N-8).

CLAUDE.md trap 2 -- "fp16, never bf16" -- is correct for the Quadro RTX 6000
(Turing, cc 7.5) this project was developed on: bf16 there is emulated at
7.0 TFLOP/s against 38.9 for fp16, 5.6x slower and slower than fp32. It is
**not** a universal rule. Ampere and later (compute capability >= 8.0) have
native bf16 and it is the better choice there; CPU (and MPS) have no usable
fp16 path at all. `select_dtype()` below expresses that as a pure function
of device + compute capability, so this machine's cc 7.5 -> float16 becomes
one case of the rule rather than the rule itself, and the owner's next GPU
(if Ampere or later) reaches the opposite conclusion automatically.

    device                            dtype       why
    --------------------------------  ----------  --------------------------
    CUDA, compute capability < 8.0    float16     Turing/Volta: bf16 emulated
    CUDA, compute capability >= 8.0   bfloat16    Ampere+: native bf16
    CPU or MPS                        float32     fp16 unsupported/pathological
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

#: Compute-capability major version at/above which CUDA has native
#: (non-emulated) bf16 -- Ampere and later.
AMPERE_CC_MAJOR = 8


def resolve_device(force: str | torch.device | None = None) -> torch.device:
    """Pick the compute device.

    `force` (e.g. "cpu", "cuda", "mps") overrides autodetection -- pass it,
    or set AERIAL_DEVICE (see config.py), to deliberately run without a GPU
    (N-8's CPU-only path). With no force, prefers CUDA, then MPS, then CPU.
    """
    if force is not None:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(
    device: torch.device, capability: tuple[int, int] | None = None
) -> torch.dtype:
    """The dtype rule, as a pure function of device capability (N-8).

    `capability` lets a caller fabricate the (major, minor) compute
    capability pair to exercise a branch this machine's hardware cannot
    reach (e.g. an Ampere cc 8.6) -- matching hardware is never required to
    test it. When omitted on an actual CUDA device, it is read from the
    device itself via `torch.cuda.get_device_capability`.

    CPU and MPS ignore `capability` entirely: neither concept applies there,
    and both always resolve to float32.
    """
    if device.type != "cuda":
        return torch.float32
    major, _minor = (
        capability if capability is not None else torch.cuda.get_device_capability(device)
    )
    return torch.bfloat16 if major >= AMPERE_CC_MAJOR else torch.float16
