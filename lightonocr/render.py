"""Turn a PDF or image file into page images sized for the model."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Longest edge used to render training data; the model applies no further resizing,
# so this is the effective inference resolution.
DEFAULT_LONGEST_EDGE = 2048


def fit(image: Image.Image, longest_edge: int | None = DEFAULT_LONGEST_EDGE) -> Image.Image:
    """Downscale so that max(width, height) <= longest_edge. Never upscales."""
    if longest_edge is None:
        return image
    w, h = image.size
    scale = longest_edge / max(w, h)
    if scale >= 1:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def render_pdf(path: str | Path, longest_edge: int = DEFAULT_LONGEST_EDGE, pages: list[int] | None = None) -> list[Image.Image]:
    """Render PDF pages (1-based page numbers; all pages by default) at the given longest edge."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    numbers = list(range(1, len(pdf) + 1)) if pages is None else pages
    images = []
    for n in numbers:
        page = pdf[n - 1]
        w, h = page.get_size()  # points
        images.append(page.render(scale=longest_edge / max(w, h)).to_pil().convert("RGB"))
    return images


def load_pages(path: str | Path, longest_edge: int = DEFAULT_LONGEST_EDGE, pages: list[int] | None = None) -> list[Image.Image]:
    """Load a PDF (one image per page) or a single image, ready to send to the model."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return render_pdf(path, longest_edge, pages)
    return [fit(Image.open(path).convert("RGB"), longest_edge)]
