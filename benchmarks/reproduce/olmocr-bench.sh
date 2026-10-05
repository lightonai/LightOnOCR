#!/usr/bin/env bash
# Reproduces every row of results/olmocr-bench.csv, then prints each new score next to the published one.
# Usage: [GPU=0] [GPUS=0,1] [RUN=<run folder>] reproduce/olmocr-bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-reproduce-$(date +%F)}
GPU=${GPU:-0}     # single-GPU models
GPUS=${GPUS:-0,1} # Infinity-Parser2-Pro (tensor parallel 2)
ocr-bench() { uv run --frozen ocr-bench "$@"; }

ocr-bench download olmocr-bench
uv run --frozen playwright install chromium   # math tests render with KaTeX

# Through ParseBench's providers: runs/$RUN/olmocr-bench/
ocr-bench run olmocr-bench mistral-ocr-4-1:no-hf --run "$RUN"        # MISTRAL_API_KEY
ocr-bench run olmocr-bench mistral-ocr-4-1:annotation --run "$RUN"   # MISTRAL_API_KEY
ocr-bench run olmocr-bench cohere-parse-5 --run "$RUN"               # COHERE_API_KEY
ocr-bench run olmocr-bench OvisOCR2 --run "$RUN" --gpu "$GPU"
# LightOnOCR: 3 reads per page (temperature 0.1, 400 DPI, 5M-pixel cap: the pipeline defaults);
# omit-margins also writes <name>-nopp, the output before PP3.
ocr-bench run olmocr-bench LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision eadb22b11fd458626dfdaded5b506c0475424b4f --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run olmocr-bench LightOnOCR-1B:omit-margins --model-id lightonai/LightOnOCR-3-1B --revision eadb22b11fd458626dfdaded5b506c0475424b4f --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run olmocr-bench LightOnOCR --model-id lightonai/LightOnOCR-3-0.8B --revision a00da38601ef40871294b9dbfd525b7ffaa4db39 '--default-chat-template-kwargs={"enable_thinking":false}' --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run olmocr-bench LightOnOCR:omit-margins --model-id lightonai/LightOnOCR-3-0.8B --revision a00da38601ef40871294b9dbfd525b7ffaa4db39 '--default-chat-template-kwargs={"enable_thinking":false}' --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run olmocr-bench LightOnOCR --model-id lightonai/LightOnOCR-3-4B --revision 9d29551af3329f1c6033807c1483d4c36b2eac62 '--default-chat-template-kwargs={"enable_thinking":false}' --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run olmocr-bench LightOnOCR:omit-margins --model-id lightonai/LightOnOCR-3-4B --revision 9d29551af3329f1c6033807c1483d4c36b2eac62 '--default-chat-template-kwargs={"enable_thinking":false}' --repeats 3 --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench score olmocr-bench --run "$RUN"

# With the model authors' own evaluation code (each script also scores its two rows):
# runs/$RUN/olmocr-bench-inf-mllm/ and runs/$RUN/olmocr-bench-datalab/
uv run --frozen scripts/olmocr-bench-inf-mllm.py --run "$RUN" --gpu "$GPUS"
uv run --frozen scripts/olmocr-bench-datalab.py --run "$RUN" --gpu "$GPU"
