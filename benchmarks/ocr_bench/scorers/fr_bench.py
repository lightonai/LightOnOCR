"""Score fr-bench outputs with upstream benchpdf2md. Runs in benchpdf2md's own venv: imports nothing from ocr_bench.

Each .md page is wrapped as the vlmparse Document zip upstream reads. One deviation: a test that
raises on malformed output fails instead of aborting the run. Writes upstream's
``<folder>/test_results/<time>/metrics.json``; ``ocr-bench score fr-bench`` reads it.

    python ocr_bench/scorers/fr_bench.py data/fr-bench-pdf2md runs/<date>/fr-bench/<model> ...
"""

import functools
import json
import shutil
import sys
from pathlib import Path

import joblib
from benchpdf2md import run_benchmark
from benchpdf2md.bench_tests import benchmark_tsts
from benchpdf2md.run_benchmark import get_ds, run_and_save_benchmark
from vlmparse.data_model.document import Document, Page


def _fail_on_error(run):
    @functools.wraps(run)
    def safe_run(self, content):
        try:
            return run(self, content)
        except Exception as e:
            return False, f"test raised {type(e).__name__}: {e}", 0.0

    return safe_run


for cls in vars(benchmark_tsts).values():
    if isinstance(cls, type) and issubclass(cls, benchmark_tsts.BasePDFTest) and "run" in vars(cls):
        cls.run = _fail_on_error(cls.run)
# Worker processes would re-import the unpatched classes: use threads.
run_benchmark.Parallel = functools.partial(joblib.Parallel, prefer="threads")

data, *folders = sys.argv[1:]
ds = get_ds(data)
pdf_by_stem = {Path(p).stem: p for p in ds.pdf_path.unique()}

for folder in map(Path, folders):
    latency = {}
    if (folder / "latency.jsonl").exists():
        with open(folder / "latency.jsonl") as f:
            for line in f:
                r = json.loads(line)
                latency[r["pdf"]] = r["latency"]
    shutil.rmtree(folder / "results", ignore_errors=True)
    for md in sorted((folder / "md").glob("*/*.md")):
        lat = latency.get(md.stem)
        doc = Document(file_path=pdf_by_stem[md.stem], pages=[Page(text=md.read_text(), latency=lat)], latency=lat)
        doc.to_zip(folder / "results" / md.stem)
    run_and_save_benchmark(ds=ds, inference_folder=folder, save_folder=folder, model=folder.name)
