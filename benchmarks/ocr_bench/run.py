"""Run models on a benchmark: start each model's server, convert the documents, stop the server.

    ocr-bench run olmocr-bench OvisOCR2 --gpu 0
    ocr-bench run fr-bench mistral-ocr-4-1:annotation            # <model>:<variant>
    ocr-bench run parsebench LightOnOCR --model-id <HF repo> --revision <commit>
    ocr-bench run olmocr-bench OvisOCR2 dots.mocr cohere-parse-5 --gpu 0,1   # in parallel, GPUs split

Outputs go to runs/<run>/<benchmark>/<name>/ and logs to runs/<run>/logs/. Unknown flags go to
``vllm serve`` (e.g. ``--gpu-memory-utilization 0.7``). Score with ``ocr-bench score <benchmark>``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from ocr_bench import convert
from ocr_bench.config import BENCHMARKS, MODELS, ROOT, RUNS, SERVERS
from ocr_bench.serve import check_gpus, serving


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def repo_commit() -> str:
    """This repo's commit, ``-dirty`` when tracked files have local changes."""
    git = ["git", "-C", str(ROOT)]
    commit = subprocess.run([*git, "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run([*git, "diff", "--quiet", "HEAD"]).returncode
    return f"{commit or '?'}{'-dirty' if dirty else ''}"


def _parse_model(spec: str) -> tuple[str, str]:
    key, _, variant = spec.partition(":")
    if key not in MODELS:
        raise SystemExit(f"unknown model {key!r}; models: {', '.join(MODELS)}")
    if variant not in MODELS[key].pipelines:
        variants = ", ".join(f"{key}:{v}" if v else key for v in MODELS[key].pipelines)
        raise SystemExit(f"unknown variant {spec!r}; variants: {variants}")
    return key, variant


def default_name(spec: str, model_id: str | None = None) -> str:
    key, variant = _parse_model(spec)
    base = model_id.rstrip("/").rsplit("/", 1)[-1] if model_id else key
    return f"{base}-{variant}" if variant else base


@contextmanager
def predict_sidecar(kind: str, backend_port: int, port: int, log: Path) -> Iterator[None]:
    """``ocr-bench predict <kind>`` on ``port`` for the duration of the block."""
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "ocr_bench.cli", "predict", kind,
           "--port", str(port), "--backend", f"http://127.0.0.1:{backend_port}/v1"]  # fmt: skip
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"ocr-bench predict {kind} exited: see {log}")
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(1)
        yield
    finally:
        proc.terminate()
        proc.wait()


def run_one(bench: str, spec: str, a: argparse.Namespace, vllm_args: list[str]) -> None:
    key, variant = _parse_model(spec)
    model = MODELS[key]
    pipeline = model.pipelines[variant]
    server = SERVERS[model.server] if model.server else None
    if model.api_key_env and not os.environ.get(model.api_key_env):
        raise SystemExit(f"{key} needs {model.api_key_env} (in .env)")
    if server and server.model_id_required and not (a.model_id and a.revision):
        raise SystemExit(f"{key} needs --model-id <HF repo> --revision <commit>")
    if a.model_id and not (server and server.model_id_required):
        raise SystemExit(f"{key} serves a fixed model: drop --model-id")
    if server and not a.no_serve:
        check_gpus(model.server, a.gpu)

    name = a.name or default_name(spec, a.model_id)
    run_dir = RUNS / a.run
    logs = run_dir / "logs"
    out = run_dir / bench / name
    if bench == "olmocr-bench":
        out = convert.olmocr_bench_dir(run_dir / bench) / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": default_name(key, a.model_id), "pipeline": pipeline, "model_id": a.model_id,
            "revision": a.revision, "image": a.image, "vllm_args": vllm_args,
            "command": shlex.join(["ocr-bench", *sys.argv[1:]]),
            "ocr_bench_commit": repo_commit(), "date": datetime.date.today().isoformat(),
            "env_overrides": {k: v for k, v in os.environ.items() if k.startswith("LOOCR_") and k != "LOOCR_SERVER_URL"}}  # fmt: skip
    (out / "run.json").write_text(json.dumps({k: v for k, v in meta.items() if v}, indent=2) + "\n")
    os.environ.update(model.env)

    with ExitStack() as stack:
        if server:
            port = server.port if server.fixed_port else a.port or (server.port if a.no_serve else _free_port())
            if not a.no_serve:
                container = f"ocrbench-{bench}-{name}".lower().replace(".", "-")
                stack.enter_context(serving(
                    model.server, name=container, port=port, save_logs=logs / f"{bench}.{name}.vllm.log", gpu=a.gpu,
                    model_id=a.model_id, revision=a.revision, image=a.image, env=a.env, vllm_args=vllm_args,
                ))  # fmt: skip
            if model.predict:
                predict_port = server.port + 10 if a.no_serve else _free_port()
                if not a.no_serve:
                    stack.enter_context(predict_sidecar(model.predict, port, predict_port, logs / f"{bench}.{name}.predict.log"))
                port = predict_port
            if model.url_env:
                os.environ[model.url_env] = f"http://127.0.0.1:{port}{model.url_path}"

        parallel = a.parallel or model.parallel
        if bench == "parsebench":
            convert.parsebench(pipeline, out, parallel)
        elif bench == "olmocr-bench":
            convert.olmocr_bench(pipeline, out.parent, name, parallel, a.repeats, a.force)
        else:
            convert.fr_bench(pipeline, out, parallel, a.force)
    print(f"done: {out}")


def run_many(bench: str, specs: list[str], a: argparse.Namespace, vllm_args: list[str]) -> None:
    """One ``ocr-bench run`` per model, in parallel; self-hosted models take the next GPUs of ``--gpu``."""
    if a.name or a.model_id or a.port:
        raise SystemExit("--name, --model-id and --port take a single model")
    gpus = a.gpu.split(",")
    cmds = {}
    for spec in specs:  # assign every GPU before starting anything
        cmd = [sys.executable, "-m", "ocr_bench.cli", "run", bench, spec, "--run", a.run, "--repeats", str(a.repeats)]
        if a.parallel:
            cmd += ["--parallel", str(a.parallel)]
        if a.force:
            cmd.append("--force")
        server = MODELS[_parse_model(spec)[0]].server
        if server:
            n = SERVERS[server].gpus
            if len(gpus) < n:
                raise SystemExit(f"--gpu {a.gpu}: not enough GPUs for {', '.join(specs)}")
            cmd += ["--gpu", ",".join(gpus[:n]), *vllm_args]
            gpus = gpus[n:]
        cmds[spec] = cmd
    logs = RUNS / a.run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = {}
    try:
        for spec, cmd in cmds.items():
            log = logs / f"{bench}.{default_name(spec)}.log"
            print(f"{spec}: {log}")
            with open(log, "w") as f:
                procs[spec] = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        failed = [spec for spec, proc in procs.items() if proc.wait()]
    finally:
        for proc in procs.values():  # on interrupt: each run stops its own server
            proc.terminate()
        for proc in procs.values():
            proc.wait()
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)} (see {logs})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ocr-bench run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("benchmark", choices=list(BENCHMARKS))
    p.add_argument("models", nargs="+", metavar="model[:variant]", help=f"one of: {', '.join(MODELS)}")
    p.add_argument("--gpu", default="0", help="GPU indices, comma-separated (default: 0)")
    p.add_argument("--run", default=datetime.date.today().isoformat(), help="run folder in runs/ (default: today)")
    p.add_argument("--name", help="output folder (default: the model, or the --model-id repo, plus -<variant>)")
    p.add_argument("--model-id", help="HF repo, for models that take any checkpoint (LightOnOCR, LightOnOCR-1B)")
    p.add_argument("--revision", help="commit of --model-id (required with it: benches pin exact versions)")
    p.add_argument("--image", help="override the vLLM docker image")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="server container env var")
    p.add_argument("--port", type=int, help="server host port (default: a free one)")
    p.add_argument("--no-serve", action="store_true", help="use a server already started with `ocr-bench serve up`")
    p.add_argument("--parallel", type=int, help="concurrent pages, or documents on ParseBench (default: per model)")
    p.add_argument("--repeats", type=int, default=1, help="olmOCR-bench: outputs per page")
    p.add_argument("--force", action="store_true", help="olmOCR-bench / fr-bench: redo pages that have output")
    a, vllm_args = p.parse_known_args(argv)
    if len(a.models) > 1:
        run_many(a.benchmark, a.models, a, vllm_args)
    else:
        run_one(a.benchmark, a.models[0], a, vllm_args)


if __name__ == "__main__":
    main()
