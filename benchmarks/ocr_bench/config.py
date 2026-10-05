"""Benchmarks, vLLM servers and models: the one file to edit when adding a model.

Every version is exact: dataset revisions, docker images and model revisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
THIRD_PARTY = ROOT / "third_party"
DOCKER_DIR = ROOT / "docker"


@dataclass(frozen=True)
class Benchmark:
    repo: str  # Hugging Face dataset
    revision: str
    data: Path  # local copy


BENCHMARKS: dict[str, Benchmark] = {
    "parsebench": Benchmark("llamaindex/ParseBench", "2805a1d940f95a203e0ae4b88be9934f7765b3fc", DATA / "parsebench"),
    "olmocr-bench": Benchmark("allenai/olmOCR-bench", "54a96a6fb6a2bd3b297e59869491db4d3625b711", DATA / "olmocr-bench"),
    "fr-bench": Benchmark("pulsia/fr-bench-pdf2md", "7f1f8e7896474cea34a46c15a72a7be3c01e8c4b", DATA / "fr-bench-pdf2md"),
}


@dataclass(frozen=True)
class Server:
    """A vLLM OpenAI-compatible server in docker: the model owner's documented serving command."""

    image: str
    args: list[str]
    source: str  # where the serving command comes from
    port: int  # default host port of `ocr-bench serve up`; `ocr-bench run` picks a free one
    gpus: int = 1  # tensor parallelism: number of GPUs `--gpu` must list
    fixed_port: bool = False  # the provider hardcodes `port`: always serve on it
    entrypoint: list[str] | None = None
    mounts: dict[str, str] = field(default_factory=dict)
    dockerfile: str | None = None  # in docker/, built when `image` is missing
    model_id_required: bool = False  # `--model` comes from `--model-id` (any checkpoint of the family)


# vLLM flags stay paired with their values.
# fmt: off
LIGHTONOCR_ARGS = [
    "--served-model-name", "loocr-grounding",
    "--gpu-memory-utilization", "0.85",
    "--limit-mm-per-prompt", json.dumps({"image": 1}),
]

SERVERS: dict[str, Server] = {
    "chandra-ocr-2": Server(
        image="vllm/vllm-openai:v0.17.0",
        port=8801,
        args=[
            "--model", "datalab-to/chandra-ocr-2",
            "--revision", "af93b47dba1b47b6640c86ccf487ed2260ab9a09",
            "--no-enforce-eager",
            "--max-num-seqs", "64",
            "--dtype", "bfloat16",
            "--max-model-len", "18000",
            "--max_num_batched_tokens", "8192",
            "--gpu-memory-utilization", ".85",
            "--enable-prefix-caching",
            "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
            "--served-model-name", "chandra",
        ],
        source="github.com/datalab-to/chandra chandra/scripts/vllm.py (gpu=h100)",
    ),
    "surya-ocr-2": Server(
        image="vllm/vllm-openai:v0.20.1",
        port=8802,
        args=[
            "--model", "datalab-to/surya-ocr-2",
            "--revision", "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3",
            "--no-enforce-eager",
            "--max-num-seqs", "104",
            "--dtype", "bfloat16",
            "--max-model-len", "18000",
            "--max-num-batched-tokens", "16384",
            "--gpu-memory-utilization", "0.85",
            "--enable-prefix-caching",
            "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
            "--served-model-name", "datalab-to/surya-ocr-2",
            # No MTP speculative decoding (unlike upstream): it crashes vLLM v0.20.1 under load.
        ],
        source="github.com/datalab-to/surya surya/inference/backends/vllm.py (VLLM_GPU_TYPE=h100)",
    ),
    "dots.mocr": Server(
        image="vllm/vllm-openai:v0.17.1",
        port=8803,
        args=[
            "--model", "rednote-hilab/dots.mocr",
            "--revision", "e539fbb52280393adc081b289ec597430a0f9031",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.9",
            "--chat-template-content-format", "string",
            "--served-model-name", "dots-ocr-1.5",  # the model name dots_ocr_1_5_parse requests
            "--trust-remote-code",
        ],
        source="huggingface.co/rednote-hilab/dots.mocr README + github.com/rednote-hilab/dots.mocr README",
    ),
    "OvisOCR2": Server(
        image="vllm/vllm-openai:v0.22.1",
        port=8805,
        args=[
            "--model", "ATH-MaaS/OvisOCR2",
            "--revision", "1fc9221b7823a371d6e97f92d527cc847e24e107",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.8",
            "--gdn-prefill-backend", "triton",
            "--served-model-name", "ovisocr2",
        ],
        source="huggingface.co/ATH-MaaS/OvisOCR2 README (offline LLM kwargs -> server flags)",
    ),
    "jina-ocr-v1": Server(
        image="vllm/vllm-openai:v0.22.1",
        port=8806,
        # FlashAttention-3 (picked on Hopper) crashes while capturing CUDA graphs.
        args=["--attention-config", json.dumps({"flash_attn_version": 2})],
        entrypoint=["python3", "/ocrbench/jina_vllm_server.py"],
        mounts={str(DOCKER_DIR): "/ocrbench"},
        source="huggingface.co/jinaai/jina-ocr-v1 README + deepseek_ocr_mtp.py",
    ),
    "KDL-Frontier-Parser-nano": Server(
        image="vllm/vllm-openai:v0.24.0",
        port=8807,
        args=[
            "--model", "KDLAI/KDL-Frontier-Parser-nano",
            "--revision", "a6cb7d2a9ac5fd13f527764f5411c9ec3ad2c4ec",
            "--served-model-name", "kdl-frontier-parser-nano",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "24",
            "--trust-remote-code",
            "--limit-mm-per-prompt", json.dumps({"image": 1}),
        ],
        source="huggingface.co/KDLAI/KDL-Frontier-Parser-nano README (Serving)",
    ),
    "Infinity-Parser2-Pro": Server(
        image="ocrbench/vllm-openai:v0.17.1-transformers5.3",
        dockerfile="infinity-parser2.Dockerfile",
        port=8000,  # upstream's infinity_parser2 provider requests localhost:8000
        fixed_port=True,
        gpus=2,
        args=[
            "--model", "infly/Infinity-Parser2-Pro",
            "--revision", "b27d470100514329fc6439aada8f16ccea5f9e2a",
            "--trust-remote-code",
            "--reasoning-parser", "qwen3",
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "65536",
            "--mm-encoder-tp-mode", "data",
            "--mm-processor-cache-type", "shm",
            "--enable-prefix-caching",
            "--served-model-name", "infly/Infinity-Parser2-Pro",
            # Not in the card: the SDK's vllm-server backend does not send enable_thinking=False,
            # and without it the model returns no content.
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
        ],
        source="huggingface.co/infly/Infinity-Parser2-Pro README (vllm serve)",
    ),
    # INF-MLLM's olmOCR-bench evaluation command (scripts/olmocr-bench-inf-mllm.py).
    "Infinity-Parser2-Pro-infly": Server(
        image="ocrbench/vllm-openai:v0.17.1-transformers5.3",
        dockerfile="infinity-parser2.Dockerfile",
        port=8809,
        gpus=2,
        args=[
            "--model", "infly/Infinity-Parser2-Pro",
            "--revision", "b27d470100514329fc6439aada8f16ccea5f9e2a",
            "--trust-remote-code",
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
            "--chat-template-content-format", "openai",
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "65536",
            "--max-num-batched-tokens", "32768",
            "--mm-encoder-tp-mode", "data",
            "--mm-processor-cache-type", "shm",
            "--enable-prefix-caching",
            "--served-model-name", "inf-mllm",
            "--tensor-parallel-size", "2",
        ],
        source="github.com/infly-ai/INF-MLLM Infinity-Parser2/evaluation/olmocr-bench/README.md (vllm serve)",
    ),
    # Any LightOnOCR checkpoint (`--model-id <HF repo> --revision <commit>`), on the Qwen3 architecture.
    # --max-model-len 24576: the loocr_grounding pipelines cap pages at 5M pixels (~4.9k image tokens)
    # plus 12288 output tokens. No --mm-processor-kwargs: the checkpoint's processor allows 16.7M pixels.
    # vLLM v0.27.1's compiled mode makes the model loop: use v0.30.0 or later.
    "LightOnOCR": Server(
        image="vllm/vllm-openai:v0.30.0",
        port=8810,
        args=[*LIGHTONOCR_ARGS, "--max-model-len", "24576"],
        source="any LightOnOCR checkpoint (--model-id)",
        model_id_required=True,
    ),
    # LightOnOCR-1B checkpoints (Mistral3/Pixtral architecture): transformers 5.16.1, and the model's
    # 16384-token context (max_position_embeddings): its processor caps images at 1540 px on the
    # longest edge (<= ~3.1k image tokens), which leaves room for the pipeline's 12288 output tokens.
    "LightOnOCR-1B": Server(
        image="ocrbench/vllm-openai:v0.30.0-transformers5.16.1",
        dockerfile="lightonocr-1b.Dockerfile",
        port=8811,
        args=[*LIGHTONOCR_ARGS, "--max-model-len", "16384"],
        source="any LightOnOCR-1B checkpoint (--model-id)",
        model_id_required=True,
    ),
}
# fmt: on


@dataclass(frozen=True)
class Model:
    """A model as `ocr-bench run` runs it: ParseBench pipelines, plus how they reach the model."""

    pipelines: dict[str, str]  # variant -> ParseBench pipeline; "" is the default variant
    server: str | None = None  # key in SERVERS; None for API models
    url_env: str | None = None  # env var the pipeline reads the server URL from
    url_path: str = ""  # appended to http://127.0.0.1:<port>
    predict: str | None = None  # `ocr-bench predict` sidecar the pipeline calls, in front of the server
    api_key_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    parallel: int = 32  # concurrent requests: pages on olmOCR-bench and fr-bench, documents on ParseBench


MODELS: dict[str, Model] = {
    "chandra-ocr-2": Model({"": "chandra2_vllm"}, server="chandra-ocr-2", url_env="CHANDRA2_SERVER_URL"),
    "surya-ocr-2": Model(
        {"": "surya2_sdk"},
        server="surya-ocr-2",
        predict="surya",
        url_env="SURYA2_SERVER_URL",
        url_path="/predict",
        parallel=20,
    ),
    "dots.mocr": Model({"": "dots_ocr_1_5_parse"}, server="dots.mocr", url_env="DOTS_OCR_ENDPOINT_URL", url_path="/v1"),
    "OvisOCR2": Model({"": "ovisocr2_vllm"}, server="OvisOCR2", url_env="OVISOCR2_SERVER_URL"),
    "jina-ocr-v1": Model(
        {"": "jinaocr_vllm"}, server="jina-ocr-v1", predict="jina", url_env="JINAOCR_SERVER_URL", url_path="/predict"
    ),
    "KDL-Frontier-Parser-nano": Model(
        {"": "kdl_frontier_nano"},
        server="KDL-Frontier-Parser-nano",
        url_env="KDL_NANO_ENDPOINT_URL",
        url_path="/v1",
        env={"KDL_NANO_MODEL": "kdl-frontier-parser-nano"},
    ),
    "Infinity-Parser2-Pro": Model({"": "infinity_parser2_pro"}, server="Infinity-Parser2-Pro"),
    # omit-margins / omit-strict / raw differ from the default on olmOCR-bench and fr-bench only (see ocr_bench/runner.py).
    **{
        name: Model(
            {
                "": "loocr_grounding_parse_with_layout",
                "omit-margins": "loocr_grounding_omit_margins",
                "omit-strict": "loocr_grounding_omit_strict",
                "raw": "loocr_grounding_raw",
            },
            server=name,
            url_env="LOOCR_SERVER_URL",
            parallel=128,
        )
        for name in ("LightOnOCR", "LightOnOCR-1B")
    },
    "mistral-ocr-4-1": Model(
        {"": "mistral_ocr_4_1", "annotation": "mistral_ocr_4_1_annotation", "no-hf": "mistral_ocr_4_1_no_hf"},
        api_key_env="MISTRAL_API_KEY",
        parallel=8,
    ),
    "cohere-parse-5": Model({"": "cohere_parse_v5"}, api_key_env="COHERE_API_KEY", parallel=16),
}
