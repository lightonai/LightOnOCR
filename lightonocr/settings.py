"""Model server settings shared by `run` and `viewer`: out/settings.json, written from the viewer.

Precedence: --base-url / --model flags, then this file, then LIGHTONOCR_BASE_URL / LIGHTONOCR_MODEL,
then the defaults in client.py.
"""

from __future__ import annotations

import json
from pathlib import Path

FILE = "settings.json"
KEYS = ("base_url", "model")


def read_settings(out: Path) -> dict:
    """{"base_url": ..., "model": ...} from out/settings.json; unset keys are left out."""
    try:
        data = json.loads((Path(out) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: data[k] for k in KEYS if data.get(k)}


def write_settings(out: Path, **values: str | None) -> dict:
    """Update out/settings.json with the given keys (an empty value unsets a key) and return the result."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {**read_settings(out), **{k: v for k, v in values.items() if k in KEYS}}
    settings = {k: v for k, v in settings.items() if v}
    (out / FILE).write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings
