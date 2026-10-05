"""Run the model over a document and write the results to out/<name>/. Shared by `run` and `viewer`.

Per page: page-NNN.png (the image as sent to the model), page-NNN.md (model output, escaped dollars repaired by the client) and, in grounding
mode, page-NNN.json (parsed blocks). finish() then writes meta.json with the title, mode and pages.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

from PIL import Image

from .client import LightOnOCR
from .grounding import parse_blocks

# Pages kept in flight at once. vLLM batches concurrent requests, so this is what sets throughput.
DEFAULT_CONCURRENCY = 64


def parse_pages(spec: str) -> list[int]:
    """'1,3-5' -> [1, 3, 4, 5]"""
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(part))
    return pages


def process_page(
    ocr: LightOnOCR, out: Path, n: int, img: Image.Image, mode: str, *, temperature: float = 0.2, max_tokens: int | None = None
) -> dict:
    """OCR one page and write its files. Runs in a worker thread; the OpenAI client is thread-safe."""
    stem = f"page-{n:03d}"
    img.save(out / f"{stem}.png")
    raw = ocr.ocr(img, mode, temperature=temperature, max_tokens=max_tokens, longest_edge=max(img.size))  # already rendered at size
    (out / f"{stem}.md").write_text(raw, encoding="utf-8")
    blocks = None
    if mode == "grounding":
        blocks = [b.to_dict() for b in parse_blocks(raw)]
        (out / f"{stem}.json").write_text(
            json.dumps({"page": n, "width": img.width, "height": img.height, "coordinates": "x1,y1,x2,y2 normalised to 0-1000", "blocks": blocks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {"page": n, "image": f"{stem}.png", "width": img.width, "height": img.height, "raw": raw, "blocks": blocks}


def ocr_pages(
    ocr: LightOnOCR,
    out: str | Path,
    numbers: list[int],
    images: list[Image.Image],
    mode: str,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    ordered: bool = True,
    **options,
) -> Iterator[dict]:
    """Send the pages to the server, up to ``concurrency`` at a time, and yield each page's result dict.

    ``ordered=True`` yields in page order (for printing), ``False`` in completion order (for progress).
    ``options`` (temperature, max_tokens) go to process_page.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(images))))
    try:
        futures = [pool.submit(process_page, ocr, out, n, img, mode, **options) for n, img in zip(numbers, images)]
        for fut in futures if ordered else as_completed(futures):
            yield fut.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # on an error or Ctrl-C, do not wait for the requests in flight


def finish(out: str | Path, title: str, mode: str, results: list[dict]) -> None:
    """Write meta.json: the title, mode and page numbers of the run."""
    pages = sorted(r["page"] for r in results)
    meta = {"title": title, "mode": mode, "pages": pages, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (Path(out) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
