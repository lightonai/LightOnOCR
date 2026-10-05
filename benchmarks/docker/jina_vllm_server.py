"""``vllm serve`` for jinaai/jina-ocr-v1 with the model card's ``register()`` and FastMTP kwargs.

``register()`` patches vLLM in-process, so it must run in the same process as the server.
Extra CLI args are passed to ``vllm serve``.
"""

import json
import os
import sys

from huggingface_hub import snapshot_download

MODEL = "jinaai/jina-ocr-v1"
REVISION = "904f815deed0fbeab02d82d4648b70106272ffe3"

snapshot = snapshot_download(MODEL, revision=REVISION)
sys.path.insert(0, snapshot)
# register() puts the resolved blobs/ dir on PYTHONPATH for vLLM's workers; add the snapshot dir.
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in [snapshot, os.environ.get("PYTHONPATH")] if p)
from deepseek_ocr_mtp import register, vllm_llm_kwargs  # noqa: E402

register()
kw = vllm_llm_kwargs(MODEL, num_speculative_tokens=3, mtp_heads=1, mtp_recursive=True)
kw["speculative_config"]["revision"] = REVISION  # the MTP draft is the same checkpoint


def main() -> None:
    argv = [
        "vllm",
        "serve",
        kw["model"],
        "--revision",
        REVISION,
        "--dtype",
        kw["dtype"],
        "--tensor-parallel-size",
        str(kw["tensor_parallel_size"]),
        "--hf-overrides",
        json.dumps(kw["hf_overrides"]),
        "--speculative-config",
        json.dumps(kw["speculative_config"]),
    ]
    if kw.get("trust_remote_code"):
        argv.append("--trust-remote-code")
    argv += sys.argv[1:]
    print("jina launcher:", " ".join(argv), flush=True)
    sys.argv = argv

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


# vLLM's workers re-import this file: register() runs there too, the server only here.
if __name__ == "__main__":
    main()
