"""Run the self-hosted models as vLLM OpenAI-compatible servers in docker (specs: ``SERVERS`` in config.py).

Containers are named ``ocrbench-<server>`` (or ``ocrbench-<--model-id repo>``) and bound to 127.0.0.1.

    ocr-bench serve up OvisOCR2 --gpu 0
    ocr-bench serve logs OvisOCR2
    ocr-bench serve down OvisOCR2
    ocr-bench serve print OvisOCR2      # the docker command only
    ocr-bench serve list
    ocr-bench serve build LightOnOCR-1B  # build a server's image from docker/ (up and run do it when missing)
    ocr-bench serve up LightOnOCR --model-id <HF repo> --revision <commit> --gpu 1

Unknown flags (e.g. ``--gpu-memory-utilization 0.7``) are passed to ``vllm serve``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from ocr_bench.config import DOCKER_DIR, RUNS, SERVERS

# The vLLM images ship a CUDA forward-compatibility libcuda (/usr/local/cuda/compat) that their ld cache
# resolves ahead of the host driver's. When its version differs from the host kernel module, CUDA fails
# to initialize (cudaGetDeviceCount error 803). The host driver's library, mounted by the NVIDIA runtime,
# goes first; `--env LD_LIBRARY_PATH=...` still overrides it.
HOST_LIBCUDA_FIRST = "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"


def container_name(server: str, model_id: str | None = None) -> str:
    name = model_id.rstrip("/").rsplit("/", 1)[-1] if model_id else server
    return "ocrbench-" + name.lower().replace(".", "-")


def check_gpus(server: str, gpu: str) -> None:
    n = len(gpu.split(","))
    if n != SERVERS[server].gpus:
        raise SystemExit(f"{server} runs on {SERVERS[server].gpus} GPU(s), got --gpu {gpu}")


def docker_cmd(
    server: str,
    *,
    name: str,
    gpu: str,
    port: int,
    model_id: str | None = None,
    revision: str | None = None,
    image: str | None = None,
    env: Sequence[str] = (),
    vllm_args: Sequence[str] = (),
) -> list[str]:
    spec = SERVERS[server]
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--runtime",
        "nvidia",
        "--gpus",
        f'"device={gpu}"',  # quoted: docker splits "device=2,3" on the comma
        "-v",
        f"{hf_home}:/root/.cache/huggingface",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "-p",
        f"127.0.0.1:{port}:8000",
        "--ipc=host",
    ]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-e", "HF_TOKEN"]
    cmd += ["-e", HOST_LIBCUDA_FIRST]
    for kv in env:
        cmd += ["-e", kv]
    for src, dst in spec.mounts.items():
        cmd += ["-v", f"{src}:{dst}:ro"]
    if spec.entrypoint:
        cmd += ["--entrypoint", spec.entrypoint[0], image or spec.image, *spec.entrypoint[1:]]
    else:
        cmd += [image or spec.image]
    model_args = ["--model", model_id, "--revision", revision] if model_id else []
    return cmd + model_args + spec.args + list(vllm_args) + ["--port", "8000"]


def _docker(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=False, **kwargs)


def _running(name: str) -> bool:
    out = _docker("inspect", "-f", "{{.State.Running}}", name, capture_output=True, text=True)
    return out.stdout.strip() == "true"


def wait_ready(name: str, port: int, timeout: float = 1800) -> None:
    """Wait for ``/v1/models``; fail fast, with the log tail, if the container stops."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not _running(name):
            _docker("logs", "--tail", "50", name)
            raise SystemExit(f"container {name} stopped before the server was ready (log tail above)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit(f"{name} not ready after {timeout:.0f}s; see `docker logs {name}`")


def ensure_image(server: str, *, rebuild: bool = False) -> None:
    """Build the server's image from its Dockerfile in docker/ when it is missing (or ``rebuild``).

    A lock file serializes builds, so parallel ``ocr-bench run`` calls build the image once.
    """
    spec = SERVERS[server]
    if not spec.dockerfile:
        if rebuild:
            raise SystemExit(f"{server} uses a published image ({spec.image}): nothing to build")
        return
    missing = lambda: _docker("image", "inspect", spec.image, capture_output=True).returncode != 0  # noqa: E731
    if not rebuild and not missing():
        return
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / ".docker-build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if rebuild or missing():
            print(f"building {spec.image} from docker/{spec.dockerfile} ...", flush=True)
            subprocess.run(
                ["docker", "build", "-t", spec.image, "-f", str(DOCKER_DIR / spec.dockerfile), str(DOCKER_DIR)],
                check=True,
            )


def up(server: str, *, name: str, port: int, replace: bool = False, **kwargs) -> None:
    """Start the server and wait until it answers. ``kwargs`` go to :func:`docker_cmd`."""
    if _docker("container", "inspect", name, capture_output=True).returncode == 0:
        if not replace:
            raise SystemExit(f"container {name} already exists: `docker rm -f {name}` it, or pass --replace")
        _docker("rm", "-f", name, capture_output=True)
    if not kwargs.get("image"):
        ensure_image(server)
    subprocess.run(docker_cmd(server, name=name, port=port, **kwargs), check=True, stdout=subprocess.DEVNULL)
    print(f"waiting for {name} on 127.0.0.1:{port} ...", flush=True)
    wait_ready(name, port)
    print(f"ready: http://127.0.0.1:{port}/v1", flush=True)


def down(name: str, save_logs: Path | None = None) -> None:
    if save_logs:
        save_logs.parent.mkdir(parents=True, exist_ok=True)
        with open(save_logs, "w") as f:
            _docker("logs", name, stdout=f, stderr=subprocess.STDOUT)
    _docker("rm", "-f", name, capture_output=True)


@contextmanager
def serving(server: str, *, name: str, port: int, save_logs: Path | None = None, **kwargs) -> Iterator[None]:
    """The server for the duration of the block; its full log goes to ``save_logs`` on exit."""
    try:
        up(server, name=name, port=port, **kwargs)
        yield
    finally:
        down(name, save_logs)


def main(argv: list[str] | None = None) -> None:
    # No abbreviations: unknown flags go to vLLM, and e.g. --model must not match --model-id.
    p = argparse.ArgumentParser(
        prog="ocr-bench serve",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("action", choices=["up", "down", "logs", "print", "list", "build"])
    p.add_argument("server", nargs="?", choices=list(SERVERS))
    p.add_argument("--gpu", default="0", help="GPU index, or comma-separated indices for tensor parallelism")
    p.add_argument("--port", type=int, help="host port (default: the server's)")
    p.add_argument("--model-id", help="HF repo, for servers that take any checkpoint (LightOnOCR, LightOnOCR-1B)")
    p.add_argument("--revision", help="commit of --model-id (required with it)")
    p.add_argument("--image", help="override the docker image")
    p.add_argument("--name", help="container name (default: ocrbench-<server or --model-id repo>)")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="container env var (repeatable)")
    p.add_argument("--replace", action="store_true", help="up: replace an existing container of the same name")
    p.add_argument("--save-logs", type=Path, help="down: write the full container log here first")
    a, vllm_args = p.parse_known_args(argv)

    if a.action == "list":
        width = max(map(len, SERVERS))
        for s, spec in SERVERS.items():
            print(f"{s:{width}s}  port {spec.port}  {spec.gpus} GPU  {spec.image}  ({spec.source})")
        return
    if not a.server:
        p.error("server is required")
    spec = SERVERS[a.server]
    if a.action == "build":
        ensure_image(a.server, rebuild=True)
        return
    if spec.model_id_required and not (a.model_id and a.revision):
        p.error(f"{a.server} needs --model-id <HF repo> --revision <commit>")
    if a.model_id and not spec.model_id_required:
        p.error(f"{a.server} serves a fixed model: drop --model-id")
    name = a.name or container_name(a.server, a.model_id)
    kwargs = dict(
        name=name,
        gpu=a.gpu,
        port=a.port or spec.port,
        model_id=a.model_id,
        revision=a.revision,
        image=a.image,
        env=a.env,
        vllm_args=vllm_args,
    )

    if a.action == "print":
        print(shlex.join(docker_cmd(a.server, **kwargs)))
    elif a.action == "up":
        check_gpus(a.server, a.gpu)
        up(a.server, replace=a.replace, **kwargs)
    elif a.action == "down":
        down(name, a.save_logs)
    elif a.action == "logs":
        _docker("logs", "--tail", "200", name)


if __name__ == "__main__":
    main(sys.argv[1:])
