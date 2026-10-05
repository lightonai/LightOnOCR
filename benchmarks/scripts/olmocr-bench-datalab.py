"""chandra-ocr-2 on olmOCR-bench with Datalab's own evaluation (chandra/scripts/olmocr_bench.py), scored
with and without its post-processing (folders chandra-ocr-2-postproc / -raw).

    uv run scripts/olmocr-bench-datalab.py [--gpu 0] [--run <date>] [--dpi 300] [--publish]

Outputs: runs/<run>/olmocr-bench-datalab/. Resumes: pages with an existing -raw .md are skipped,
and -postproc is rebuilt from -raw.
"""

import argparse
import datetime
import json
import shutil
import signal
import sys
from contextlib import nullcontext

from ocr_bench.config import RUNS, SERVERS
from ocr_bench.convert import olmocr_bench_dir
from ocr_bench.score import checkout, report, score_olmocr
from ocr_bench.serve import serving

CHANDRA = ("datalab-to/chandra", "d4f7467435aa4137d9539f000ddf0b7ced3eb43f")
SERVER = "chandra-ocr-2"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--gpu", default="0")
p.add_argument("--run", default=datetime.date.today().isoformat())
p.add_argument("--dpi", type=int, default=300)
p.add_argument("--workers", type=int, default=32)
p.add_argument("--no-serve", action="store_true", help="use a server already started with `ocr-bench serve up`")
p.add_argument("--publish", action="store_true", help="update results/olmocr-bench.csv and the README tables")
a = p.parse_args()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a plain exit, so `serving` stops the container

sys.path.insert(0, str(checkout(*CHANDRA)))  # this commit's scripts, ahead of the installed chandra-ocr
from chandra.scripts.olmocr_bench import postprocess, run_inference  # noqa: E402

bench_dir = olmocr_bench_dir(RUNS / a.run / "olmocr-bench-datalab")
logs = RUNS / a.run / "logs"
port = SERVERS[SERVER].port
server = (
    nullcontext()
    if a.no_serve
    else serving(
        SERVER, name="ocrbench-olmocr-bench-datalab", port=port, gpu=a.gpu, save_logs=logs / "olmocr-bench-datalab.vllm.log"
    )
)
with server:
    run_inference(str(bench_dir), "chandra-ocr-2-raw", a.dpi, a.workers, f"http://127.0.0.1:{port}/v1", apply_postprocess=False)

raw, post = bench_dir / "chandra-ocr-2-raw", bench_dir / "chandra-ocr-2-postproc"
shutil.rmtree(post, ignore_errors=True)
for src in sorted(raw.glob("**/*.md")):
    dst = post / src.relative_to(raw)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(postprocess(src.read_text(encoding="utf-8")), encoding="utf-8")
for name in ("postproc", "raw"):
    meta = {"model": "chandra-ocr-2", "pipeline": f"datalab-eval-{name}"}
    (bench_dir / f"chandra-ocr-2-{name}" / "run.json").write_text(json.dumps(meta) + "\n")

rows = score_olmocr(bench_dir, ["chandra-ocr-2-postproc", "chandra-ocr-2-raw"])
report("olmocr-bench", rows, bench_dir / "scores.csv", a.publish)
