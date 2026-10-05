"""Parsing of LightOnOCR-3 grounding output (grounding is a LightOnOCR-3 mode; versions 1 and 2 only do plain).

In grounding mode the model emits one layout block per marker::

    ![label](x1,y1,x2,y2) block text ...

A block's text runs until the next marker, so a table or a list spans several lines, and the
text may start on the line after the marker. Coordinates are normalised to 0-1000 on both axes,
independent of the image size. A trailing "+" on the label (``text+``) marks a block that
continues the previous one, e.g. a paragraph flowing into the next column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GROUNDING_PROMPT = "grounding"

# Labels the model emits. The parser accepts any label, so a new one from a future checkpoint
# still parses; the viewer draws labels it does not know in grey.
LABELS = [
    "title", "section", "text", "list", "caption", "footnote", "aside_text",
    "header", "footer", "page_number",
    "table", "chart", "formula", "code", "image",
]

# Labels dropped for a header/footer-omitting transcription. Same set as the post-processing
# behind the model card's olmOCR-bench grounding numbers (footnotes count as footer content there).
HEADER_FOOTER = (
    "header", "footer", "page_number", "footnote",
    "page_header", "page_footer", "page_footnote", "header_image", "footer_image",
)

MARKER = re.compile(
    r"!\[([A-Za-z_][\w-]*?)(\+?)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)[ \t]*"
)


@dataclass
class Block:
    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 normalised to 0-1000
    text: str
    continues: bool = False  # label carried a trailing "+"

    def to_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Bounding box in pixels for an image of the given size."""
        x1, y1, x2, y2 = self.bbox
        return (x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000)

    def to_dict(self) -> dict:
        return {"label": self.label, "bbox": list(self.bbox), "text": self.text, "continues": self.continues}


def parse_blocks(raw: str) -> list[Block]:
    """Split raw grounding output into blocks. Each marker owns the text up to the next marker."""
    marks = list(MARKER.finditer(raw))
    blocks = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        blocks.append(
            Block(
                label=m.group(1),
                continues=m.group(2) == "+",
                bbox=(int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))),
                text=raw[m.end():end].strip(),
            )
        )
    return blocks


def to_markdown(blocks: list[Block], drop: tuple[str, ...] = ()) -> str:
    """Join block texts back into markdown, optionally dropping some labels (e.g. HEADER_FOOTER).

    Continuation blocks (``label+``) are appended to the preceding block's paragraph.
    """
    parts: list[str] = []
    prev_kept = False
    for b in blocks:
        if b.label in drop or not b.text:
            prev_kept = False
            continue
        if b.continues and prev_kept:
            parts[-1] = f"{parts[-1]} {b.text}"
        else:
            parts.append(b.text)
        prev_kept = True
    return "\n\n".join(parts)
