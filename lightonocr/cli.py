"""Command line. ``lightonocr <file>`` runs OCR into out/<name>/; ``lightonocr viewer`` opens the web viewer."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from openai import APIConnectionError

from .client import LightOnOCR
from .pipeline import DEFAULT_CONCURRENCY, finish, ocr_pages, parse_pages
from .render import load_pages
from .settings import read_settings


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lightonocr", description="LightOnOCR (1, 2 or 3) served by vLLM: run OCR, and browse the results.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="OCR a PDF or image into out/<name>/ (the default: `lightonocr paper.pdf`)")
    run.add_argument("input", help="PDF or image file")
    run.add_argument("--mode", choices=["plain", "grounding"], default="plain", help="plain: markdown; grounding: layout blocks with boxes (LightOnOCR-3 only)")
    run.add_argument("--pages", help="pages to process, e.g. '1,3-5' (PDF only; default: all)")
    run.add_argument("--name", help="run name, i.e. its folder under --out (default: the file name without extension)")
    run.add_argument("--max-tokens", type=int, help="cap on generated tokens (default: server remaining context)")
    run.add_argument("--print", dest="echo", action="store_true", help="also print the model output to stdout")
    run.set_defaults(func=run_command)

    viewer = sub.add_parser("viewer", aliases=["serve"], help="start the web viewer: browse the runs in out/ and OCR new files from the browser")
    viewer.add_argument("--host", default="127.0.0.1")
    viewer.add_argument("--port", type=int, help="port to listen on (default: 8000, or the next free port)")
    viewer.set_defaults(func=viewer_command)

    for sp in (run, viewer):
        sp.add_argument("--out", default="out", help="folder holding the runs (default: out/)")
        sp.add_argument("--base-url", help="vLLM OpenAI-compatible URL (default: out/settings.json, then $LIGHTONOCR_BASE_URL, then http://127.0.0.1:8010/v1)")
        sp.add_argument("--model", help="served model name (default: out/settings.json, then the first model listed by the server)")
        sp.add_argument("--longest-edge", type=int, help="render/resize pages to this longest edge in px (default: the model's, 1540 for the 1B models, 2048 otherwise)")
        sp.add_argument("--temperature", type=float, default=0.2)
        sp.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"max pages sent to the model server at once (default: {DEFAULT_CONCURRENCY})")
    return p


def run_command(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        return fail(f"{src} is not a file")
    if args.concurrency < 1:
        return fail("--concurrency must be at least 1")
    out = Path(args.out) / (args.name or src.stem)

    ocr = make_client(args)  # first, as the model decides the page size and whether grounding exists
    if args.mode == "grounding" and not ocr.profile.grounding:
        return fail(f"{ocr.model} has no grounding mode: grounding needs a LightOnOCR-3 model. Use --mode plain.")

    page_numbers = parse_pages(args.pages) if args.pages else None
    status(f"rendering {src.name}…")
    started = time.monotonic()
    images = load_pages(src, args.longest_edge or ocr.profile.longest_edge, page_numbers)
    numbers = page_numbers or list(range(1, len(images) + 1))
    status(f"rendered {len(images)} page{'s' if len(images) != 1 else ''} in {time.monotonic() - started:.1f}s", done=True)

    workers = max(1, min(args.concurrency, len(images)))
    print(f"{ocr.model} @ {ocr.base_url} · {args.mode} · up to {workers} page{'s' if workers != 1 else ''} in flight", file=sys.stderr)
    progress = Progress(len(images))
    results = []
    try:
        # Completion order, so the bar moves as soon as any page is back.
        for page in ocr_pages(ocr, out, numbers, images, args.mode, concurrency=args.concurrency, ordered=False, temperature=args.temperature, max_tokens=args.max_tokens):
            results.append(page)
            info = f"{len(page['blocks'])} blocks" if page["blocks"] is not None else f"{len(page['raw'])} chars"
            progress.step(f"page {page['page']}: {info}")
    except APIConnectionError:
        progress.close()
        return fail(f"cannot reach {ocr.base_url}. Is vLLM running?")
    except KeyboardInterrupt:
        progress.close()
        print("interrupted", file=sys.stderr)
        sys.stdout.flush()
        os._exit(130)  # exit now instead of waiting for the requests still in flight
    progress.close()

    finish(out, src.name, args.mode, results)
    if args.echo:
        for page in sorted(results, key=lambda r: r["page"]):
            print(page["raw"])
    elapsed = time.monotonic() - progress.started
    per_page = f", {elapsed / len(results):.1f}s per page" if results else ""
    print(f"done: {len(results)} page{'s' if len(results) != 1 else ''} in {elapsed:.1f}s{per_page} -> {out}/ (open it with `lightonocr viewer`)", file=sys.stderr)
    return 0


# ---- Terminal output: a status line and a progress bar on stderr ---------------------------------
# On a terminal the line is redrawn in place; when stderr is a file or a pipe, each update is a line.

TTY = sys.stderr.isatty()


def status(text: str, done: bool = False) -> None:
    """Show a transient status line; ``done=True`` keeps it and moves to the next line."""
    if TTY:
        print(f"\r\033[K{text}", end="\n" if done else "", file=sys.stderr, flush=True)
    else:
        print(text, file=sys.stderr, flush=True)


class Progress:
    """``[████░░░░] 7/18 pages · 12s · ~19s left · page 4: 13 blocks``"""

    WIDTH = 28

    def __init__(self, total: int):
        self.total, self.done, self.started = total, 0, time.monotonic()

    def step(self, note: str = "") -> None:
        self.done += 1
        elapsed = time.monotonic() - self.started
        left = elapsed / self.done * (self.total - self.done)
        filled = round(self.WIDTH * self.done / max(1, self.total))
        bar = "█" * filled + "░" * (self.WIDTH - filled)
        line = f"{bar} {self.done}/{self.total} pages · {elapsed:.0f}s"
        if self.done < self.total:
            line += f" · ~{left:.0f}s left"
        status(f"{line} · {note}" if note else line)

    def close(self) -> None:
        """End the bar's line on a terminal (no-op when nothing was drawn or output is not a terminal)."""
        if TTY and self.done:
            print(file=sys.stderr)


def viewer_command(args: argparse.Namespace) -> int:
    from .server import serve  # keeps `run` free of the server code

    ocr = make_client(args)
    return serve(Path(args.out), args.host, args.port, ocr, longest_edge=args.longest_edge, temperature=args.temperature, concurrency=max(1, args.concurrency))


def make_client(args: argparse.Namespace) -> LightOnOCR:
    """The model server: --base-url/--model, else out/settings.json (set from the viewer), else env and defaults."""
    saved = read_settings(Path(args.out))
    return LightOnOCR(base_url=args.base_url or saved.get("base_url"), model=args.model or saved.get("model"))


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "viewer", "serve", "-h", "--help"):
        argv.insert(0, "run")  # `lightonocr paper.pdf` is `lightonocr run paper.pdf`
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
