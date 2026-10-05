"""ocr-bench: reproducible OCR benchmarks with ParseBench's providers.

    ocr-bench download <benchmark>...         fetch the pinned datasets into data/
    ocr-bench run <benchmark> <model>...      serve the model, convert the documents, stop the server
    ocr-bench score <benchmark> [name...]     score a run (--publish: update results/ and the README)
    ocr-bench serve up|down|logs|print|list   manage a model's vLLM server by hand
    ocr-bench predict jina|surya              /predict sidecar some providers call (started by `run`)
    ocr-bench parsebench ...                  the parse-bench CLI, with this repo's providers

Benchmarks: parsebench, olmocr-bench, fr-bench. `ocr-bench <command> --help` for details.
"""

from __future__ import annotations

import importlib
import signal
import sys

from huggingface_hub import snapshot_download

from ocr_bench.config import BENCHMARKS, ROOT

COMMANDS = {
    "run": "ocr_bench.run",
    "score": "ocr_bench.score",
    "serve": "ocr_bench.serve",
    "predict": "ocr_bench.predict",
}


def download(names: list[str]) -> None:
    if not names or "-h" in names or "--help" in names:
        raise SystemExit(f"usage: ocr-bench download <benchmark>...  ({', '.join(BENCHMARKS)})")
    for name in names:
        if name not in BENCHMARKS:
            raise SystemExit(f"unknown benchmark {name!r}; benchmarks: {', '.join(BENCHMARKS)}")
        b = BENCHMARKS[name]
        snapshot_download(b.repo, repo_type="dataset", revision=b.revision, local_dir=b.data)
        print(f"{name}: {b.data}")


def parsebench(args: list[str]) -> None:
    from parse_bench.cli import main as parse_bench_main

    import ocr_bench.providers  # noqa: F401  (registers the local providers and pipelines)

    sys.argv = ["ocr-bench parsebench", *args]
    sys.exit(parse_bench_main())


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)  # API keys and HF_TOKEN; optional
    # A plain exit on SIGTERM, so `finally` blocks stop the servers and sidecars this process started.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    command, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("-h", [])
    if command == "download":
        download(args)
    elif command == "parsebench":
        parsebench(args)
    elif command in COMMANDS:
        importlib.import_module(COMMANDS[command]).main(args)
    else:
        print(__doc__)
        sys.exit(0 if command in ("-h", "--help") else 2)


if __name__ == "__main__":
    main()
