"""Minimal inference client, parser, CLI and viewer for LightOnOCR (1, 2 and 3) served with vLLM."""

from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, LightOnOCR, to_data_url
from .grounding import GROUNDING_PROMPT, HEADER_FOOTER, LABELS, Block, parse_blocks, to_markdown
from .models import Profile, profile
from .pipeline import DEFAULT_CONCURRENCY, finish, ocr_pages, parse_pages, process_page
from .postprocess import fix_escaped_dollars
from .render import DEFAULT_LONGEST_EDGE, fit, load_pages

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_LONGEST_EDGE",
    "DEFAULT_MODEL",
    "GROUNDING_PROMPT",
    "HEADER_FOOTER",
    "LABELS",
    "Block",
    "LightOnOCR",
    "Profile",
    "finish",
    "fix_escaped_dollars",
    "fit",
    "load_pages",
    "ocr_pages",
    "parse_blocks",
    "parse_pages",
    "process_page",
    "profile",
    "to_data_url",
    "to_markdown",
]
