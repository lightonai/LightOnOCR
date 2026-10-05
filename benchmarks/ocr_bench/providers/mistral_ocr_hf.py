"""Mistral OCR with the API's ``extract_header`` / ``extract_footer`` (``mistral_ocr_hf`` provider).

With them, Mistral moves the page header/footer out of ``markdown``. Upstream
``MistralOCRProvider`` builds its request inline, so this module globally replaces its module's
``requests`` with a stand-in that adds the flags. The stand-in passes requests through unchanged
except inside this provider's ``run_inference`` (thread-local), so upstream pipelines are unaffected.
"""

from __future__ import annotations

import threading
from typing import Any

import requests
from parse_bench.inference.providers.parse import mistral_ocr
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult

_extra_body = threading.local()


class _Requests:
    """``mistral_ocr.requests``: plain ``requests``, plus this thread's extra body keys on ``post``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(requests, name)

    def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any) -> requests.Response:
        extra = getattr(_extra_body, "value", None)
        return requests.post(url, json={**json, **extra} if extra and json is not None else json, **kwargs)


mistral_ocr.requests = _Requests()


@register_provider("mistral_ocr_hf")
class MistralOCRHeaderFooterProvider(mistral_ocr.MistralOCRProvider):
    """Config (on top of upstream's): ``extract_header`` / ``extract_footer`` (bool, default True)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._extract = {
            "extract_header": bool(self.base_config.get("extract_header", True)),
            "extract_footer": bool(self.base_config.get("extract_footer", True)),
        }

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        _extra_body.value = self._extract
        try:
            result = super().run_inference(pipeline, request)
        finally:
            _extra_body.value = None
        result.raw_output["_config"].update(self._extract)
        return result
