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

from .run_identity import validate_edit_strength

logger = logging.getLogger(__name__)

MaskProvider = Callable[[str, int], "np.ndarray"]  # (seq_id, t) -> HxW float mask


class BlurBackgroundPreEditor:
    """Training-free pre-edit: keep masked content, blur the background.

    img = M * I + (1-M) * (G_sigma * I)
    This is the alpha=0, beta=0 boundary of the bounded SCOPE family
    (MATH.md section 2). Deterministic. CV2-free path uses scipy Gaussian
    filtering per channel.
    """

    def __init__(self, mask_provider: MaskProvider, sigma: float = 8.0,
                 dilate_px: int = 0):
        self.mask_provider = mask_provider
        self.sigma = sigma
        self.dilate_px = dilate_px  # Argon baseline: feather/expand the kept
        # region so context AROUND objects survives (separates context loss
        # from background destruction); 0 = original behaviour
        self.name = f"blurbg_s{sigma:g}" + (f"_d{dilate_px}" if dilate_px else "")
        self._seq = None

    def begin_sequence(self, seq_id: str) -> None:
        self._seq = seq_id

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        if self._seq is None:
            raise RuntimeError("call begin_sequence before edit_frame")
        mask = self.mask_provider(self._seq, t)  # (H, W) in [0, 1]
        if self.dilate_px > 0:
            from scipy.ndimage import binary_dilation

            m = mask > 0.5
            mask = binary_dilation(m, iterations=self.dilate_px).astype(np.float32)
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

    Adapter verified against upstream `scope/inference.py` (2026-11):
      load_scope_model(ckpt_path, device, exp_cfg=None)
      apply_scope_bgr(img_bgr, scope_ckpt, codec_family, severity,
                      device=None, exp_cfg=None)  # BGR uint8 in/out
    Families: diff_jpeg | bmshj | bpg_like; severity in [0, 1].
    Requires the upstream repo on sys.path (clone miccunifi/SCOPE) and the
    checkpoint at <repo>/runs/<exp>/best_val.pt (Google Drive folder link in
    their README; fetch with gdown --folder).
    """

    def __init__(self, scope_repo: str, ckpt_path: str, family: str, severity: float):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ScopePreEditor requires torch + the miccunifi/SCOPE code."
            ) from exc
        import sys
        if scope_repo not in sys.path:
            sys.path.insert(0, scope_repo)
        from scope.inference import load_scope_model  # upstream API

        self.scope_repo = scope_repo
        self.ckpt_path = ckpt_path
        self.family = family
        self.severity = float(severity)
        self.name = f"scope_{family}_s{severity:g}"
        self._seq = None
        self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._model = load_scope_model(ckpt_path, __import__("torch").device(self._device))
        self._ckpt = ckpt_path

    def begin_sequence(self, seq_id: str) -> None:
        self._seq = seq_id

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        from scope.inference import apply_scope_bgr

        bgr = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[..., ::-1]  # RGB->BGR
        out = apply_scope_bgr(bgr, self._ckpt, self.family, self.severity,
                              device=self._device)
        return np.asarray(out, dtype=np.float32)[..., ::-1] / 255.0  # BGR->RGB [0,1]


def _gaussian_blur(img: "np.ndarray", sigma: float) -> "np.ndarray":
    """Per-channel Gaussian blur, float32, same shape. SciPy backend."""
    from scipy.ndimage import gaussian_filter

    return np.stack(
        [gaussian_filter(img[..., c], sigma=sigma, mode="reflect") for c in range(img.shape[-1])],
        axis=-1,
    ).astype(np.float32)


class IdentityPreEditor:
    """Exact no-edit control; no model or mask dependency."""

    name = "identity"

    def begin_sequence(self, seq_id: str) -> None:
        pass

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        return img.copy()


class AttenuatedPreEditor:
    """Blend once after the complete edit, including background modification."""

    def __init__(self, editor, strength: float):
        self.editor = editor
        self.strength = strength
        self.name = f"{editor.name}_strength{strength:g}"

    def begin_sequence(self, seq_id: str) -> None:
        self.editor.begin_sequence(seq_id)

    def edit_frame(self, img: "np.ndarray", t: int) -> "np.ndarray":
        edited = self.editor.edit_frame(img.copy(), t)
        if edited.shape != img.shape or not np.isfinite(edited).all():
            raise ValueError("edited frame must have the original shape and finite pixels")
        original = img.astype(np.float32, copy=False)
        result = original + self.strength * (edited.astype(np.float32) - original)
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def build_editor(cfg: dict, mask_provider: Optional[MaskProvider] = None):
    """Factory from a config dict: {type: blurbg|scope, edit_strength: 1, ...}."""
    strength = validate_edit_strength(cfg.get("edit_strength", 1.0))
    kind = cfg.get("type")
    if kind not in ("blurbg", "scope"):
        raise ValueError(f"unknown editor type: {kind!r}")
    if strength == 0.0:
        return IdentityPreEditor()
    if kind == "blurbg":
        if mask_provider is None:
            raise ValueError("blurbg editor needs a mask_provider")
        editor = BlurBackgroundPreEditor(mask_provider, sigma=cfg.get("sigma", 8.0),
                                         dilate_px=cfg.get("dilate_px", 0))
    else:
        editor = ScopePreEditor(
            cfg["scope_repo"], cfg["ckpt_path"],
            cfg.get("family", "diff_jpeg"), cfg.get("severity", 0.5),
        )
    return editor if strength == 1.0 else AttenuatedPreEditor(editor, strength)
