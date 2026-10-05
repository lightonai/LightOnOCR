"""Provider for LightOnOCR-3 served on a vLLM server (``loocr_grounding``).

The model emits each page as a sequence of blocks, separated by blank lines:

    ![<label>](x1,y1,x2,y2) <text content>

Boxes are 0-1000 normalized (left, top, right, bottom). ``table`` text is HTML,
``formula`` is LaTeX, ``title`` / ``figure_title`` text already carries its own ``#``
heading, and figure/chart/image blocks carry a description or nothing. A ``+`` label
suffix (``text+``) marks a block continued from the previous page or column.

One inference gives both the markdown and the layout, so the single pipeline
``loocr_grounding_parse_with_layout`` scores all five ParseBench dimensions. ``normalize`` repairs
escaped math delimiters in each block's text (``lightonocr.postprocess.fix_escaped_dollars``).

Config, on top of QwenProvider's (``server_url_env``, ``dpi``, ``max_tokens``, ``temperature``,
``timeout``): ``prompt`` (default ``"grounding"``) and ``max_pixels`` (a page over this many
pixels renders at the DPI that fits it; unset = no cap), ``longest_dim`` (pages render with their longest
edge at this many pixels, whatever the page size; replaces ``dpi`` and ``max_pixels``; unset = off).
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
from pathlib import Path
from typing import Any

import aiohttp
from lightonocr.postprocess import fix_escaped_dollars
from parse_bench.inference.providers.base import ProviderPermanentError
from parse_bench.inference.providers.parse.qwen import QwenProvider, _build_layout_page
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import PageIR, ParseLayoutPageIR, ParseOutput
from parse_bench.schemas.pipeline_io import InferenceResult, RawInferenceResult
from parse_bench.schemas.product import ProductType

# A block's text runs from its marker to the next marker (or the end of the string).
_BLOCK_MARKER_RE = re.compile(r"!\[([a-zA-Z_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*")

# Raw label -> Canonical17 label (the scorer derives title_level / text_role from these).
# Other labels (list, caption, code) map to Text.
LABEL_MAP: dict[str, str] = {
    "paragraph": "Text",
    "title": "Title",
    "figure_title": "Caption",
    "footnote": "Footnote",
    "formula": "Formula",
    "table": "Table",
    "header": "Page-header",
    "footer": "Page-footer",
    "page_number": "Page-footer",
    "figure": "Picture",
    "chart": "Picture",
    "image": "Picture",
    # ParseBench's layout GT labels an image in the header/footer band as Picture.
    "header_image": "Picture",
    "footer_image": "Picture",
}

# PageSectionRule (is_header / is_footer) reads these dedicated page fields, not layout items.
# Header/footer images are left out: their text describes a visual, not the header.
_PAGE_SECTIONS = {
    "page_header_markdown": {"header", "page_header"},
    "page_footer_markdown": {"footer", "page_footer", "footnote", "page_footnote"},
    "printed_page_number": {"page_number"},
}

# Dropped by omit_margins: page margins and side notes. Footnotes are document content and are kept.
MARGIN_LABELS = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "page_header",
        "page_footer",
        "header_image",
        "footer_image",
        "aside_text",
    }
)

# Dropped by omit_strict: page margins and footnotes; side notes are kept. The label set of the
# training-side eval_olmocr_bench.py ``grounding_omit_strict`` postprocessor.
STRICT_LABELS = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "footnote",
        "page_footnote",
        "page_header",
        "page_footer",
        "header_image",
        "footer_image",
    }
)

# The model emits no language tag on `code` blocks; CodeBlockRule needs one. Only the languages
# ParseBench's text_formatting tests use are detected; anything else gets a bare fence.
_CODE_LANG_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("json", re.compile(r"^\s*[\{\[].*[\}\]]\s*$", re.DOTALL)),
    ("python", re.compile(r"\b(def |import |from \w+ import|self\.|print\()")),
    ("cpp", re.compile(r"#include|::|\bstd::|\bvoid\b.*\(|;\s*$", re.MULTILINE)),
    ("fortran", re.compile(r"^\s{6}\S|\bIF\(|\bEND\s*DO\b|\bCALL\b", re.MULTILINE)),
]
_LEADING_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")


def _base_label(raw_label: str) -> str:
    return raw_label.strip().lower().rstrip("+")


def _strip_outer_formula_delimiters(text: str) -> str:
    """Strip one outer ``$$``/``$`` pair, only when the interior holds no other ``$``."""
    stripped = text.strip()
    for delim in ("$$", "$"):
        if len(stripped) > 2 * len(delim) and stripped.startswith(delim) and stripped.endswith(delim):
            interior = stripped[len(delim) : -len(delim)]
            if "$" not in interior and interior.strip():
                return interior.strip()
    return text


def _render_block(label: str, text: str) -> str:
    """Markdown for one block, adding the markup its label implies when the model gave none.

    Like upstream's ``items_to_markdown``, except that a title keeps the model's own ``#`` level
    (the model uses one ``title`` label for titles and section headers); H1 is added only when
    the text has no marker.
    """
    if not text.strip():
        return ""
    label = _base_label(label)
    if label == "code" and not text.lstrip().startswith("```"):
        lang = next((lang for lang, rx in _CODE_LANG_HINTS if rx.search(text)), "")
        return f"```{lang}\n{text}\n```"
    if label == "title" and not _LEADING_HEADING_RE.match(text):
        return f"# {text.strip()}"
    if label == "formula":
        return f"$$\n{_strip_outer_formula_delimiters(text)}\n$$"
    return text


@register_provider("loocr_grounding")
class LoocrGroundingProvider(QwenProvider):
    """QwenProvider with the grounding block format. Config: ``model`` (default "loocr-grounding")."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)
        self._model = self.base_config.get("model", "loocr-grounding")
        self._prompt = self.base_config.get("prompt", "grounding")
        self._max_pixels = self.base_config.get("max_pixels")
        self._longest_dim = self.base_config.get("longest_dim")
        # Sweep overrides, read from the environment of `ocr-bench run`.
        if os.environ.get("LOOCR_DPI"):
            self._dpi = int(os.environ["LOOCR_DPI"])
        if os.environ.get("LOOCR_MAX_PIXELS"):  # "none" removes the cap
            v = os.environ["LOOCR_MAX_PIXELS"]
            self._max_pixels = None if v.lower() == "none" else int(float(v))
        if os.environ.get("LOOCR_LONGEST_DIM"):
            self._longest_dim = int(os.environ["LOOCR_LONGEST_DIM"])
        if os.environ.get("LOOCR_TEMPERATURE"):
            self._temperature = float(os.environ["LOOCR_TEMPERATURE"])

    @staticmethod
    def parse_blocks(content: str) -> list[dict[str, Any]]:
        """The raw grounding string as ordered ``{label, bbox, text}`` blocks."""
        matches = list(_BLOCK_MARKER_RE.finditer(content))
        ends = [m.start() for m in matches[1:]] + [len(content)] if matches else []
        return [
            {"label": m.group(1), "bbox": [int(m.group(i)) for i in range(2, 6)], "text": content[m.end() : end].strip()}
            for m, end in zip(matches, ends, strict=True)
        ]

    @classmethod
    def omit_margins(cls, content: str) -> str:
        """Raw grounding string -> markdown without margin blocks (``MARGIN_LABELS``) or markers.

        Block texts are joined by blank lines and text before the first marker is dropped; an output
        with no marker is kept as is.
        """
        return cls.omit_labels(content, MARGIN_LABELS)

    @classmethod
    def omit_strict(cls, content: str) -> str:
        """``omit_margins`` with ``STRICT_LABELS``: footnotes are dropped too, ``aside_text`` is kept."""
        return cls.omit_labels(content, STRICT_LABELS)

    @classmethod
    def omit_labels(cls, content: str, labels: frozenset[str]) -> str:
        """Raw grounding string -> markdown without the blocks whose label (``+`` stripped) is in ``labels``."""
        blocks = cls.parse_blocks(content)
        if not blocks:
            return content.strip()
        return "\n\n".join(b["text"] for b in blocks if b["text"] and _base_label(b["label"]) not in labels)

    def _pdf_to_images_with_size(self, pdf_path: Path) -> list[tuple[bytes, int, int]]:
        """Pages at ``self._dpi``, each lowered to fit ``self._max_pixels`` before it is rasterized
        (rendering first and shrinking after hits PIL's decompression-bomb limit on huge scans)."""
        if self._longest_dim:
            return self._pdf_to_images_longest_dim(pdf_path)
        if self._max_pixels is None:
            return super()._pdf_to_images_with_size(pdf_path)
        from pdf2image import convert_from_path
        from PIL import Image
        from pypdf import PdfReader

        pages = []
        for i, page in enumerate(PdfReader(pdf_path).pages):
            box = page.cropbox
            w_in, h_in = abs(float(box.width)) / 72, abs(float(box.height)) / 72  # some boxes are inverted
            dpi = min(self._dpi, math.floor(math.sqrt(self._max_pixels / (w_in * h_in))))
            img = convert_from_path(pdf_path, dpi=dpi, first_page=i + 1, last_page=i + 1)[0]
            if img.width * img.height > self._max_pixels:  # poppler rounds page sizes up
                scale = math.sqrt(self._max_pixels / (img.width * img.height))
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pages.append((buf.getvalue(), img.width, img.height))
        if not pages:
            raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
        return pages

    def _pdf_to_images_longest_dim(self, pdf_path: Path) -> list[tuple[bytes, int, int]]:
        """Pages rasterized straight to ``self._longest_dim`` px on the longest edge (poppler's -scale-to)."""
        from pdf2image import convert_from_path

        pages = []
        for img in convert_from_path(pdf_path, size=self._longest_dim):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            pages.append((buf.getvalue(), img.width, img.height))
        if not pages:
            raise ProviderPermanentError(f"No pages found in PDF: {pdf_path}")
        return pages

    async def _run_inference_async(self, image_bytes: bytes, img_width: int, img_height: int) -> dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            raw_content = await self._call_api(session, base64.b64encode(image_bytes).decode())
        layout_items = [
            # raw_label alongside the (lossy) canonical category: footer and page_number both
            # map to Page-footer, but the page-section fields need them apart.
            {
                "bbox": b["bbox"],
                "category": LABEL_MAP.get(_base_label(b["label"]), "Text"),
                "text": b["text"],
                "raw_label": b["label"],
            }
            for b in self.parse_blocks(raw_content)
        ]
        return {
            "pages": [
                {
                    "page_index": 0,
                    "width": img_width,
                    "height": img_height,
                    "raw_response": raw_content,
                    "layout_items": layout_items,
                }
            ],
            "prompt_mode": "parse_with_layout",
            "_config": {"server_url": self._server_url, "model": self._model, "dpi": self._dpi, "max_pixels": self._max_pixels, "longest_dim": self._longest_dim, "temperature": self._temperature},
        }

    async def _run_inference_pages_async(self, pages: list[tuple[bytes, int, int]]) -> dict[str, Any]:
        results = [await self._run_inference_async(*page) for page in pages]
        return {**results[0], "pages": [{**r["pages"][0], "page_index": i} for i, r in enumerate(results)]}

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"LoocrGroundingProvider only supports PARSE, got {raw_result.product_type}")

        pages: list[PageIR] = []
        layout_pages: list[ParseLayoutPageIR] = []
        for page in sorted(raw_result.raw_output.get("pages") or [], key=lambda p: p.get("page_index", 0)):
            items = [{**it, "text": fix_escaped_dollars(str(it.get("text", "")))} for it in page.get("layout_items", [])]
            rendered = (_render_block(str(it.get("raw_label", "")), str(it.get("text", ""))) for it in items)
            markdown = "\n\n".join(x for x in rendered if x)
            if markdown:
                markdown = self._sanitize_html_attributes(self._convert_md_tables_to_html(markdown))
            pages.append(PageIR(page_index=page.get("page_index", 0), markdown=markdown))
            width, height = page.get("width", 0), page.get("height", 0)
            if items and width > 0 and height > 0:
                layout_page = _build_layout_page(
                    layout_items=items,
                    page_number=page.get("page_index", 0) + 1,
                    img_width=width,
                    img_height=height,
                    page_markdown=markdown,
                )
                for field, labels in _PAGE_SECTIONS.items():
                    texts = (
                        str(it.get("text", "")).strip() for it in items if _base_label(str(it.get("raw_label", ""))) in labels
                    )
                    setattr(layout_page, field, "\n\n".join(t for t in texts if t))
                layout_pages.append(layout_page)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown="\n\n".join(p.markdown for p in pages),
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )
