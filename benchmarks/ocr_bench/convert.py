"""Run a ParseBench pipeline on each benchmark's documents.

* ParseBench: ``parse-bench run`` (inference, normalization and evaluation).
* olmOCR-bench and fr-bench: one call per single-page PDF through :mod:`ocr_bench.runner` (the model's
  raw markdown). Pages that already have a ``.md`` are skipped, so a rerun resumes and retries failures.
  Pipelines with a ``FOLDER_POSTPROCESS`` convert into ``<name>-nopp`` and post-process a copy into
  ``<name>``, so the score is reported before and after it.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ocr_bench.config import BENCHMARKS


def olmocr_bench_dir(path: Path) -> Path:
    """An olmOCR-bench data dir at ``path``: ``pdfs/`` and ``*.jsonl`` linked from data/, one folder per model."""
    data = BENCHMARKS["olmocr-bench"].data / "bench_data"
    if not (data / "pdfs").is_dir():
        raise SystemExit(f"missing {data}: run `ocr-bench download olmocr-bench`")
    path.mkdir(parents=True, exist_ok=True)
    for src in [data / "pdfs", *data.glob("*.jsonl")]:
        if not (path / src.name).is_symlink():
            (path / src.name).symlink_to(src.resolve())
    return path


def parsebench(pipeline: str, out: Path, parallel: int) -> None:
    data = BENCHMARKS["parsebench"].data
    if not data.is_dir():
        raise SystemExit(f"missing {data}: run `ocr-bench download parsebench`")
    cmd = [sys.executable, "-m", "ocr_bench.cli", "parsebench", "run", pipeline,
           "--input_dir", str(data), "--output_dir", str(out), "--open_report", "False",
           "--max_concurrent", str(parallel)]  # fmt: skip
    subprocess.run(cmd, check=True)


def olmocr_bench(pipeline: str, bench_dir: Path, name: str, parallel: int, repeats: int = 1, force: bool = False) -> None:
    from olmocr.bench.convert import process_pdfs

    from ocr_bench.runner import FOLDER_POSTPROCESS, make_runner

    # With a folder post-processing, pages convert into <name>-nopp and <name> is derived from it.
    folder_pp = FOLDER_POSTPROCESS.get(("olmocr-bench", pipeline))
    folder = f"{name}-nopp" if folder_pp else name
    if folder_pp:
        (bench_dir / folder).mkdir(exist_ok=True)
        (bench_dir / folder / "run.json").write_bytes((bench_dir / name / "run.json").read_bytes())
    config = {pipeline: {"method": make_runner(pipeline, "olmocr-bench"), "kwargs": {}, "folder_name": folder}}
    asyncio.run(
        process_pdfs(
            config,
            pdf_directory=str(bench_dir / "pdfs"),
            data_directory=str(bench_dir),
            repeats=repeats,
            remove_text=False,
            force=force,
            max_parallel=parallel,
        )
    )
    if folder_pp:
        postprocess_folder(bench_dir / folder, bench_dir / name, *folder_pp)


def postprocess_folder(src: Path, dst: Path, label: str, postprocess) -> None:
    """Every ``.md`` of ``src`` through ``postprocess`` into ``dst``; dst's run.json pipeline gets ``+<label>``."""
    meta = json.loads((src / "run.json").read_text())
    meta["pipeline"] = f"{meta['pipeline']}+{label}"
    dst.mkdir(exist_ok=True)
    (dst / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    for md in tqdm(sorted(src.rglob("*.md")), desc=f"{label} -> {dst.name}"):
        out = dst / md.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(postprocess(md.read_text()))


def fr_bench(pipeline: str, out: Path, parallel: int, force: bool = False) -> None:
    """Writes ``<out>/md/<category>/<test>.md`` and ``<out>/latency.jsonl``."""
    from ocr_bench.runner import make_runner

    data = BENCHMARKS["fr-bench"].data
    pdfs = sorted((data / "pdfs").glob("*/*/*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {data}/pdfs: run `ocr-bench download fr-bench`")
    todo = [(pdf, out / "md" / pdf.parent.parent.name / f"{pdf.stem}.md") for pdf in pdfs]
    todo = [(pdf, md) for pdf, md in todo if force or not md.exists()]
    print(f"{pipeline} -> {out}: {len(todo)}/{len(pdfs)} pages to convert", flush=True)

    run = make_runner(pipeline, "fr-bench")
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with open(out / "latency.jsonl", "a") as latency_log:

        def convert(item: tuple[Path, Path]) -> bool:
            pdf, md = item
            tic = time.perf_counter()
            try:
                text = run(str(pdf.resolve()))
            except Exception:
                print(f"FAILED {pdf}\n{traceback.format_exc()}", flush=True)
                return False  # no .md: a rerun retries the page
            latency = time.perf_counter() - tic
            md.parent.mkdir(parents=True, exist_ok=True)
            md.write_text(text or "")
            with lock:
                latency_log.write(json.dumps({"pdf": pdf.stem, "latency": latency}) + "\n")
                latency_log.flush()
            return True

        with ThreadPoolExecutor(parallel) as pool:
            ok = list(tqdm(pool.map(convert, todo), total=len(todo)))
    failed = ok.count(False)
    print(f"done: {len(ok) - failed} converted, {failed} failed")
    if failed:
        raise SystemExit(f"{failed} pages failed: rerun to retry them")
