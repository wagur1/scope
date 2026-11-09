"""Whole-edit strength validation and one-strength-per-workdir provenance."""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path


IDENTITY_VERSION = 1


def validate_edit_strength(value=1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("edit_strength must be a finite number in [0, 1]")
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("edit_strength must be a finite number in [0, 1]")
    return strength


def read_run_identity(root: Path) -> dict:
    path = root / "results" / "run_identity.json"
    try:
        identity = json.loads(path.read_text())
        if not isinstance(identity, dict) or type(identity.get("version")) is not int:
            raise ValueError("invalid identity schema")
        if identity["version"] != IDENTITY_VERSION:
            raise ValueError("unsupported identity version")
        strength = validate_edit_strength(identity["edit_strength"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{root}: missing or invalid run identity; use a fresh workdir") from exc
    return {"version": IDENTITY_VERSION, "edit_strength": strength}


def ensure_run_identity(root: Path, strength=1.0, *, create: bool = False) -> dict:
    expected = {"version": IDENTITY_VERSION, "edit_strength": validate_edit_strength(strength)}
    path = root / "results" / "run_identity.json"
    if path.exists():
        saved = read_run_identity(root)
        if saved != expected:
            raise ValueError(f"{root}: edit_strength differs from saved run; use a fresh workdir")
        return saved
    if not create:
        return read_run_identity(root)
    # Inputs and masks may already exist. Derived artifacts without an identity
    # cannot establish which edit strength produced them, including legacy runs.
    if any(p.is_file() for directory in ("edited", "enc", "results")
           for p in (root / directory).rglob("*")):
        raise ValueError(f"{root}: existing artifacts have no run identity; use a fresh workdir")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(expected, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return expected
