"""Pre-edit operators (MATH.md section 2).

Every editor implements the same protocol:
  editor.begin_sequence(seq_id)   -> reset per-sequence state
  editor.edit_frame(img, t)       -> edited frame (float32 RGB in [0,1], HWC)
  editor.name                     -> stable identifier used in result tables

Two editors ship:
  * BlurBackgroundPreEditor — training-free fallback (MATH.md arm A2).
    Pure M* I + (1-M) * Blur_k(I): the bounded-residual branches (Delta_obj,
    Delta_bg) require training, so this editor is the alpha=beta=0 member of
    the SCOPE family. Still a genuine pre-edit arm: it changes content.
  * ScopePreEditor — wraps the released miccunifi/SCOPE model per frame
    (MATH.md arm A3). Requires the upstream repo + weights; guarded import.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

MaskProvider = Callable[[str, int], "np.ndarray"]  # (seq_id, t) -> HxW float mask


class BlurBackgroundPreEditor:
    """Training-free pre-edit: keep masked content, blur the background.

    img = M * I + (1-M) * (G_sigma * I)
    This is the alpha=0, beta=0 boundary of the bounded SCOPE family
    (MATH.md section 2). Deterministic. CV2-free path uses scipy Gaussian
    filtering per channel.
    """

    def __init__(self, mask_provider: MaskProvider, sigma: float = 8.0):
        self.mask_provider = mask_provider
        self.sigma = sigma
        self.name = f"blurbg_s{sigma:g}"
        self._seq = None

    def begin_sequence(self, seq_id: str) -> None:
        self._seq = seq_id

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        if self._seq is None:
            raise RuntimeError("call begin_sequence before edit_frame")
        mask = self.mask_provider(self._seq, t)  # (H, W) in [0, 1]
        if mask.shape != img.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} != frame shape {img.shape[:2]} (seq={self._seq}, t={t})"
            )
        blurred = _gaussian_blur(img, self.sigma)
        m3 = mask[..., None].astype(np.float32)
        out = m3 * img + (1.0 - m3) * blurred
        return np.clip(out, 0.0, 1.0)


class ScopePreEditor:
    """Per-frame wrapper around the released miccunifi/SCOPE image pre-editor.

    NOTE (VERIFY): the adapter below targets the published repo's inference
    API. If the upstream signature differs, adapt `_load`/`_edit_one` only —
    the arm protocol (begin_sequence/edit_frame/name) must not change.
    Scope conditioning (family, severity) is applied identically for every
    frame of a sequence so that the edit is content-consistent in severity.
    """

    def __init__(self, weights_path: str, family: str, severity: float):
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ScopePreEditor requires torch + the miccunifi/SCOPE code. "
                "Install per upstream README, or use BlurBackgroundPreEditor for E1."
            ) from exc
        self.weights_path = weights_path
        self.family = family
        self.severity = float(severity)
        self.name = f"scope_{family}_s{severity:g}"
        self._seq = None
        self._model = self._load(weights_path)

    def _load(self, weights_path: str):
        # VERIFY against miccunifi/SCOPE actual API at integration time.
        from scope_model import load_preeditor  # upstream repo on sys.path

        return load_preeditor(weights_path)

    def begin_sequence(self, seq_id: str) -> None:
        self._seq = seq_id

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        # VERIFY: upstream call signature (image tensor CHW [0,1], family, severity).
        out = self._model(img, family=self.family, severity=self.severity)  # type: ignore[operator]
        return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)


def _gaussian_blur(img: "np.ndarray", sigma: float) -> "np.ndarray":
    """Per-channel Gaussian blur, float32, same shape. SciPy backend."""
    from scipy.ndimage import gaussian_filter

    return np.stack(
        [gaussian_filter(img[..., c], sigma=sigma, mode="reflect") for c in range(img.shape[-1])],
        axis=-1,
    ).astype(np.float32)


def build_editor(cfg: dict, mask_provider: Optional[MaskProvider] = None):
    """Factory from a config dict: {type: blurbg|scope, ...}."""
    kind = cfg.get("type")
    if kind == "blurbg":
        if mask_provider is None:
            raise ValueError("blurbg editor needs a mask_provider")
        return BlurBackgroundPreEditor(mask_provider, sigma=cfg.get("sigma", 8.0))
    if kind == "scope":
        return ScopePreEditor(
            cfg["weights_path"], cfg.get("family", "jpeg"), cfg.get("severity", 0.5)
        )
    raise ValueError(f"unknown editor type: {kind!r}")
