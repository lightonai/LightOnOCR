#!/usr/bin/env bash
# Reproduces every row of results/parsebench.csv, then prints each new score next to the published one.
# Usage: [GPU=0] [RUN=<run folder>] reproduce/parsebench.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${RUN:-reproduce-$(date +%F)}
GPU=${GPU:-0}
ocr-bench() { uv run --frozen ocr-bench "$@"; }

ocr-bench download parsebench

ocr-bench run parsebench KDL-Frontier-Parser-nano --run "$RUN" --gpu "$GPU"
ocr-bench run parsebench cohere-parse-5 --run "$RUN"   # COHERE_API_KEY
ocr-bench run parsebench LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision eadb22b11fd458626dfdaded5b506c0475424b4f --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run parsebench LightOnOCR --model-id lightonai/LightOnOCR-3-0.8B --revision a00da38601ef40871294b9dbfd525b7ffaa4db39 '--default-chat-template-kwargs={"enable_thinking":false}' --run "$RUN" --gpu "$GPU"   # HF_TOKEN
ocr-bench run parsebench LightOnOCR --model-id lightonai/LightOnOCR-3-4B --revision 9d29551af3329f1c6033807c1483d4c36b2eac62 '--default-chat-template-kwargs={"enable_thinking":false}' --run "$RUN" --gpu "$GPU"   # HF_TOKEN

ocr-bench score parsebench --run "$RUN"
