"""Single-page runner shared by ``ocr-bench-olmocr`` and ``ocr-bench-frbench``.

Inference is the ParseBench provider's ``run_inference``, but its ``normalize`` (tuned for
ParseBench's scorers) is skipped: the page markdown is the model's raw output
(``RAW_MARKDOWN``), with only layout wrappers (bboxes, labels, JSON) removed. LightOnOCR's markdown is
then post-processed (``POSTPROCESS``): escaped-dollar repair. PP3 (olmOCR-bench, omit-margins / omit-strict) runs after
the conversion, on a copy of the output (``FOLDER_POSTPROCESS``).
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightonocr.postprocess import fix_escaped_dollars, pp3
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.inference.providers.base import (
    ProviderRateLimitError,
    ProviderTransientError,
)
from parse_bench.inference.providers.parse.cohere_parse import (
    _extract_markdown_and_layout,
)
from parse_bench.inference.providers.parse.infinity_parser2 import InfinityParser2Provider
from parse_bench.inference.providers.parse.mistral_ocr import _inject_image_annotations
from parse_bench.inference.providers.registry import create_provider
from parse_bench.inference.runner import (
    BACKOFF_MULTIPLIER,
    INITIAL_BACKOFF_S,
    MAX_RETRIES,
)
from parse_bench.schemas.pipeline_io import InferenceRequest
from parse_bench.schemas.product import ProductType

import ocr_bench.providers  # noqa: F401  (registers the local providers)
from ocr_bench.providers.loocr_grounding import LoocrGroundingProvider

_RENDER_LOCK = threading.Lock()


def _join(texts: list[str]) -> str:
    return "\n\n".join(t.strip() for t in texts if t and t.strip())


def _page_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-page dicts of the providers that keep one page inline and more in ``page_results``."""
    return raw.get("page_results") or [raw]


# Chandra 2 (ocr_layout task): top-level <div data-bbox="..." data-label="...">...</div> per block.
_CHANDRA_DIV_RE = re.compile(r'<div data-bbox="[^"]*" data-label="[^"]*">(.*?)</div>(?=\s*(?:<div data-bbox=|$))', re.DOTALL)


def _unwrap_divs(html: str) -> str:
    return _join(_CHANDRA_DIV_RE.findall(html)) or html


def _dots_page(page: dict[str, Any]) -> str:
    # Layout prompts return a JSON list of {bbox, category, text}; unparseable JSON keeps the raw text.
    items = page.get("layout_items")
    return _join([it.get("text") or "" for it in items]) if items else page.get("raw_response", "")


def _mistral(raw: dict[str, Any]) -> str:
    # Annotation pipelines: each figure's images[].image_annotation replaces its ![](id) placeholder.
    return _join([_inject_image_annotations(p.get("markdown") or "", p.get("images") or []) for p in raw.get("pages") or []])


# ParseBench provider name -> extractor. A pipeline name entry overrides its provider's.
RAW_MARKDOWN: dict[str, Callable[[dict[str, Any]], str]] = {
    "chandra2": lambda raw: _join([_unwrap_divs(p.get("markdown", "")) for p in _page_results(raw)]),
    # Visual-element blocks are bbox/type metadata around an optional <table>: keep the table.
    "cohere_parse": lambda raw: _join(
        [_extract_markdown_and_layout(p.get("markdown") or "", (0, 0))[0] for p in raw.get("pages") or []]
    ),
    "dots_ocr_parse": lambda raw: _join([_dots_page(p) for p in raw.get("pages", [])]),
    # doc2json: a JSON list of {bbox, category, text} per page, kept in the model's order.
    "infinity_parser2": lambda raw: _join(
        [e.get("text") or "" for p in _page_results(raw) for e in InfinityParser2Provider._load_elements(p.get("result"))]
    ),
    "jinaocr": lambda raw: _join([p.get("markdown", "") for p in _page_results(raw)]),
    "loocr_grounding": lambda raw: _join(
        [b["text"] for p in raw.get("pages", []) for b in LoocrGroundingProvider.parse_blocks(p["raw_response"])]
    ),
    "loocr_grounding_omit_margins": lambda raw: _join(
        [LoocrGroundingProvider.omit_margins(p["raw_response"]) for p in raw.get("pages", [])]
    ),
    "loocr_grounding_omit_strict": lambda raw: _join(
        [LoocrGroundingProvider.omit_strict(p["raw_response"]) for p in raw.get("pages", [])]
    ),
    "loocr_grounding_raw": lambda raw: "\n\n".join(p["raw_response"] for p in raw.get("pages", [])),
    "mistral_ocr": _mistral,
    "mistral_ocr_hf": _mistral,  # header/footer already out of the markdown
    "ovisocr2": lambda raw: _join([p.get("markdown", "") for p in _page_results(raw)]),
    # ocr-bench-predict surya's "html" is chandra-format divs; its "markdown" (chandra's
    # parse_markdown, headers/footers dropped) is not used.
    "surya2": lambda raw: _join([_unwrap_divs(p.get("html", "")) for p in _page_results(raw)]),
}


# (benchmark, pipeline name) -> post-processing of the extracted markdown; ``*`` matches any benchmark.
# loocr_grounding_raw stays the model output as is.
POSTPROCESS: dict[tuple[str, str], Callable[[str], str]] = {
    ("*", "loocr_grounding_parse_with_layout"): fix_escaped_dollars,
    ("*", "loocr_grounding_omit_margins"): fix_escaped_dollars,
    ("*", "loocr_grounding_omit_strict"): fix_escaped_dollars,
}

# (benchmark, pipeline name) -> post-processing applied after conversion, to a copy of the output:
# ``<name>-nopp`` keeps the markdown before it and ``<name>`` gets it, so ``ocr-bench score`` reports both.
FOLDER_POSTPROCESS: dict[tuple[str, str], tuple[str, Callable[[str], str]]] = {
    # Dollar repair, then formatting rules tuned on olmOCR-bench (lightonocr/postprocess).
    ("olmocr-bench", "loocr_grounding_omit_margins"): ("pp3", pp3),
    ("olmocr-bench", "loocr_grounding_omit_strict"): ("pp3", pp3),
}


def make_runner(pipeline_name: str, benchmark: str) -> Callable[..., str]:
    """``fn(pdf_path, page_num=1) -> markdown`` for a ParseBench pipeline (olmOCR-bench runner signature)."""
    pipeline = get_pipeline(pipeline_name)
    extract = RAW_MARKDOWN.get(pipeline_name) or RAW_MARKDOWN.get(pipeline.provider_name)
    if extract is None:
        raise NotImplementedError(f"{pipeline_name}: no raw-markdown extractor for provider {pipeline.provider_name!r}")
    postprocess = POSTPROCESS.get((benchmark, pipeline_name)) or POSTPROCESS.get(("*", pipeline_name))
    provider = create_provider(pipeline)
    # PyMuPDF (cohere_parse's renderer) is not thread-safe: serialize rendering.
    if hasattr(provider, "_render_pages"):
        render = provider._render_pages

        def locked_render(*args, **kwargs):
            with _RENDER_LOCK:
                return render(*args, **kwargs)

        provider._render_pages = locked_render

    def run(pdf_path: str, page_num: int = 1) -> str:
        if page_num != 1:
            raise ValueError(f"expected single-page PDFs, got page {page_num} of {pdf_path}")
        request = InferenceRequest(
            example_id=Path(pdf_path).stem, source_file_path=os.path.abspath(pdf_path), product_type=ProductType.PARSE
        )
        # parse-bench's own retry policy.
        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = provider.run_inference(pipeline, request).raw_output
                break
            except (ProviderTransientError, ProviderRateLimitError):
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(INITIAL_BACKOFF_S * BACKOFF_MULTIPLIER**attempt)
        # QwenProvider-based providers return request errors in raw_output instead of raising.
        if isinstance(raw, dict) and raw.get("_error"):
            raise RuntimeError(f"{pipeline_name}: {raw['_error']}")
        markdown = extract(raw)
        return postprocess(markdown) if postprocess else markdown

    return run
