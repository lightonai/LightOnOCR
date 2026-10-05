"""Infinity-Parser2-Pro on olmOCR-bench with INF-MLLM's own evaluation, scored with and without its
post-processing (folders Infinity-Parser2-Pro-postproc / -raw), with the olmocr commit INF-MLLM pins.

    uv run scripts/olmocr-bench-inf-mllm.py [--gpu 0,1] [--run <date>] [--publish]

Outputs: runs/<run>/olmocr-bench-inf-mllm/. infer.py resumes from its checkpoint.
"""

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import nullcontext

from ocr_bench.config import RUNS, SERVERS
from ocr_bench.convert import olmocr_bench_dir
from ocr_bench.score import checkout, report, score_olmocr
from ocr_bench.serve import check_gpus, serving

INF_MLLM = ("infly-ai/INF-MLLM", "9a93df02e725ee98ccd545d02580ec2c12c203ae")
OLMOCR = ("allenai/olmocr", "f7cfe4c22098b154c76b6ec950d1c0a464eecf8d")  # pinned by INF-MLLM
SERVER = "Infinity-Parser2-Pro-infly"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--gpu", default="0,1")
p.add_argument("--run", default=datetime.date.today().isoformat())
p.add_argument("--batch", type=int, default=32)
p.add_argument("--no-serve", action="store_true", help="use a server already started with `ocr-bench serve up`")
p.add_argument("--publish", action="store_true", help="update results/olmocr-bench.csv and the README tables")
a = p.parse_args()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # a plain exit, so `serving` stops the container

inf_mllm, olmocr = checkout(*INF_MLLM), checkout(*OLMOCR)
eval_dir = inf_mllm / "Infinity-Parser2" / "evaluation" / "olmocr-bench"
bench_dir = olmocr_bench_dir(RUNS / a.run / "olmocr-bench-inf-mllm")
logs = RUNS / a.run / "logs"
postproc, raw = bench_dir / "Infinity-Parser2-Pro-postproc", bench_dir / "Infinity-Parser2-Pro-raw"
port = SERVERS[SERVER].port

if not a.no_serve:
    check_gpus(SERVER, a.gpu)
server = (
    nullcontext()
    if a.no_serve
    else serving(
        SERVER, name="ocrbench-olmocr-bench-inf-mllm", port=port, gpu=a.gpu, save_logs=logs / "olmocr-bench-inf-mllm.vllm.log"
    )
)
with server:
    # Absolute --pdf_dir: infer.py's resume checkpoint is keyed on the path.
    cmd = [sys.executable, eval_dir / "infer.py", "--pdf_dir", (bench_dir / "pdfs").resolve(), "--output_dir", postproc,
           "--batch_size", str(a.batch), "--api_url", f"http://127.0.0.1:{port}/v1/chat/completions"]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": str(inf_mllm / "Infinity-Parser2"), "PYTHONUNBUFFERED": "1"}
    subprocess.run(cmd, env=env, check=True)

# The raw folder: INF-MLLM's inference.jsonl written as .md without its post-processing.
sys.path.insert(0, str(eval_dir))
from infer import write_str_to_md  # noqa: E402

shutil.rmtree(raw, ignore_errors=True)
with open(postproc / "inference.jsonl", encoding="utf-8") as f:
    for line in filter(str.strip, f):
        r = json.loads(line)
        write_str_to_md(str(raw), r["pdf"], r["markdown"] or "")
for folder, pipeline in ((postproc, "inf-mllm-eval-postproc"), (raw, "inf-mllm-eval-raw")):
    (folder / "run.json").write_text(json.dumps({"model": "Infinity-Parser2-Pro", "pipeline": pipeline}) + "\n")

rows = score_olmocr(bench_dir, [postproc.name, raw.name], olmocr_src=olmocr)
report("olmocr-bench", rows, bench_dir / "scores.csv", a.publish)
