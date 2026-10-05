"""What differs between LightOnOCR generations, told apart by the model name.

| Checkpoint                                   | Generation | Modes             | Page size (longest edge) |
|----------------------------------------------|------------|-------------------|--------------------------|
| LightOnOCR-1B-1025                           | 1          | plain             | 1540 px                  |
| LightOnOCR-2-1B (and its -bbox, -base, ...)  | 2          | plain             | 1540 px                  |
| LightOnOCR-3-1B-*                            | 3          | plain, grounding  | 1540 px                  |
| LightOnOCR-3-0.8B-*, LightOnOCR-3-4B-*       | 3          | plain, grounding  | 2048 px                  |

Grounding is a LightOnOCR-3 mode: generations 1 and 2 were never trained on it. The 1B models
(Pixtral vision encoder) have a processor that downscales images to 1540 px, so pages are rendered at
that size; the Qwen3.5-based models take the image as sent. A name that matches none of these (e.g. a
custom ``--served-model-name``) is treated as LightOnOCR-3 at 2048 px.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    generation: int | None  # 1, 2, 3, or None when the name does not say
    grounding: bool  # whether the model has a grounding mode
    longest_edge: int  # page size the model was trained on, in px
    top_p: float  # the model card's sampling, with temperature 0.2


def profile(model: str) -> Profile:
    """The profile of a served model name such as ``lightonai/LightOnOCR-2-1B``."""
    name = model.lower()
    if m := re.search(r"lightonocr-(\d+)-", name):  # LightOnOCR-2-1B, LightOnOCR-3-0.8B-...
        generation = int(m.group(1))
    elif "lightonocr-1b" in name:  # LightOnOCR-1B-1025
        generation = 1
    else:
        generation = None
    if generation in (1, 2):
        return Profile(generation, grounding=False, longest_edge=1540, top_p=0.9)
    pixtral = "lightonocr-3-1b" in name
    return Profile(generation, grounding=True, longest_edge=1540 if pixtral else 2048, top_p=1.0)


def check_mode(model: str, mode: str) -> None:
    """Raise ValueError when ``mode`` is unknown or the model does not have it."""
    if mode not in ("plain", "grounding"):
        raise ValueError(f"mode must be 'plain' or 'grounding', got {mode!r}")
    if mode == "grounding" and not profile(model).grounding:
        raise ValueError(f"{model} has no grounding mode: grounding needs a LightOnOCR-3 model. Use plain mode.")
