"""OpenAI-compatible client for a LightOnOCR model (1, 2 or 3) served by vLLM."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from pathlib import Path

from openai import APIConnectionError, OpenAI, Timeout
from PIL import Image

from .grounding import GROUNDING_PROMPT, Block, parse_blocks
from .models import Profile, check_mode, profile
from .postprocess import fix_escaped_dollars
from .render import fit

DEFAULT_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_MODEL = "lightonai/LightOnOCR-3-1B"

ImageLike = Image.Image | str | Path | bytes


def to_data_url(image: ImageLike, longest_edge: int | None = None) -> str:
    """Encode a PIL image, image path or raw bytes as a PNG data URL."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    image = fit(image.convert("RGB"), longest_edge)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LightOnOCR:
    """Minimal client. ``mode`` is ``"plain"`` (markdown) or ``"grounding"`` (layout blocks, LightOnOCR-3 only).

    Constructing the client never talks to the server. The served model name is asked on first use
    unless given (``model`` or ``LIGHTONOCR_MODEL``), and ``DEFAULT_MODEL`` is used while the server
    cannot be asked, so a viewer can be served while vLLM is down.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = base_url or os.environ.get("LIGHTONOCR_BASE_URL", DEFAULT_BASE_URL)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.environ.get("LIGHTONOCR_API_KEY", "EMPTY"),
            timeout=Timeout(timeout, connect=10.0),  # a long page may take minutes; an unreachable host fails fast
        )
        self.model_setting: str | None = model or os.environ.get("LIGHTONOCR_MODEL")  # as configured, None = ask the server
        self.known_model: str | None = self.model_setting  # None until discovered
        self._lock = threading.Lock()
        self._retry_at = 0.0

    @property
    def model(self) -> str:
        """The model name put in requests: given, discovered from the server, or the default meanwhile."""
        if self.known_model:
            return self.known_model
        with self._lock:
            if self.known_model is None and time.monotonic() >= self._retry_at:
                self.known_model = self._served_model()
                self._retry_at = time.monotonic() + 30  # a server that is down is not asked again on every call
        return self.known_model or DEFAULT_MODEL

    @property
    def profile(self) -> Profile:
        """What the served model supports: grounding or not, page size, sampling (see models.py)."""
        return profile(self.model)

    def _served_model(self) -> str | None:
        """First model listed by the server, or None if it cannot be asked right now."""
        models = self.check()["models"]
        return models[0] if models else None

    def check(self) -> dict:
        """Ask the server for its models, briefly: {"reachable": bool, "models": [...], "error": str | None}."""
        try:
            models = self.client.with_options(timeout=Timeout(10.0, connect=5.0), max_retries=0).models.list().data
            return {"reachable": True, "models": [m.id for m in models], "error": None}
        except APIConnectionError:
            return {"reachable": False, "models": [], "error": f"cannot reach {self.base_url}"}
        except Exception as e:
            return {"reachable": False, "models": [], "error": f"{type(e).__name__}: {e}"}

    def ocr(
        self,
        image: ImageLike,
        mode: str = "plain",
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        top_p: float | None = None,
        longest_edge: int | None = None,
        fix_dollars: bool = True,
    ) -> str:
        """Run the model on one page image and return its text output.

        plain:     the image alone -> markdown transcription.
        grounding: the image + the text "grounding" -> one `![label](x1,y1,x2,y2) text` block per line.
                   LightOnOCR-3 only: a LightOnOCR-1 or -2 model raises ValueError.
        Images larger than ``longest_edge`` are downscaled before being sent. ``longest_edge`` and
        ``top_p`` default to the model's (``self.profile``): 1540 px or 2048 px, 0.9 or 1.0.
        ``fix_dollars`` repairs escaped math delimiters (``\\$x\\$`` -> ``$x$``, see postprocess/);
        False returns the model's output untouched.
        """
        check_mode(self.model, mode)
        defaults = self.profile
        content: list[dict] = [{"type": "image_url", "image_url": {"url": to_data_url(image, longest_edge or defaults.longest_edge)}}]
        if mode == "grounding":
            content.append({"type": "text", "text": GROUNDING_PROMPT})
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            top_p=defaults.top_p if top_p is None else top_p,
            # LightOnOCR-3 0.8B and 4B were trained without thinking; other chat templates ignore the flag.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            **extra,
        )
        text = response.choices[0].message.content or ""
        return fix_escaped_dollars(text) if fix_dollars else text

    def plain(self, image: ImageLike, **kwargs) -> str:
        """Markdown transcription of the page."""
        return self.ocr(image, "plain", **kwargs)

    def grounding(self, image: ImageLike, **kwargs) -> list[Block]:
        """Layout blocks with 0-1000 normalised boxes (LightOnOCR-3 only). ``ocr(image, "grounding")`` gives the raw text."""
        return parse_blocks(self.ocr(image, "grounding", **kwargs))
