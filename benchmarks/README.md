# Benchmarks

`ocr-bench` produces reproducible scores for OCR and document-parsing models on three benchmarks:

- [ParseBench](https://github.com/run-llama/ParseBench) (LlamaIndex)
- [olmOCR-bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) (Ai2)
- [fr-bench-pdf2md](https://huggingface.co/datasets/pulsia/fr-bench-pdf2md) (French documents)

Every model runs through **ParseBench's own provider**, so it gets the same page rendering, prompt
and sampling on all three benchmarks. Self-hosted models are served with their authors' documented
vLLM command. Everything is pinned: packages, datasets, docker images, model revisions and the
external scorers.

This folder is a uv project of its own, separate from the `lightonocr` client at the
[repository root](../README.md): run the commands below from here (`cd benchmarks`).

## Results

Full rows (per-category scores, scorer version, date) are in [`results/`](results). The tables
below are generated from them by `ocr-bench score --publish`.

**ParseBench**: Overall is the mean of the five leaderboard dimensions. "published" is the score
on ParseBench's leaderboard.

<!-- results:parsebench -->
| model | pipeline | Overall | Tables | Charts | Content | Formatting | Grounding | published |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LightOnOCR-3-4B | `loocr_grounding_parse_with_layout` | 75.14 | 83.79 | 66.14 | 89.85 | 67.56 | 68.34 |  |
| LightOnOCR-3-0.8B | `loocr_grounding_parse_with_layout` | 74.61 | 84.51 | 64.67 | 88.91 | 66.72 | 68.23 |  |
| KDL-Frontier-Parser-nano | `kdl_frontier_nano` | 72.39 | 85.92 | 58.57 | 87.69 | 54.98 | 74.81 | 76.36 |
| LightOnOCR-3-1B | `loocr_grounding_parse_with_layout` | 71.40 | 84.84 | 57.30 | 88.62 | 64.49 | 61.74 |  |
| cohere-parse-5 | `cohere_parse_v5` | 50.22 | 87.23 | 1.01 | 86.70 | 67.16 | 9.00 |  |
<!-- /results -->

**olmOCR-bench**: the overall score with its 95% CI, then the score without the `headers_footers`
category.

<!-- results:olmocr-bench -->
| model | pipeline | overall | ± | excl. headers/footers |
|---|---|---:|---:|---:|
| Infinity-Parser2-Pro | `inf-mllm-eval-postproc` | 86.9 | 0.9 | 85.7 |
| LightOnOCR-3-4B | `loocr_grounding_omit_margins+pp3` | 86.1 | 0.8 | 85.6 |
| Infinity-Parser2-Pro | `inf-mllm-eval-raw` | 85.9 | 0.8 | 84.6 |
| chandra-ocr-2 | `datalab-eval-postproc` | 85.8 | 0.8 | 85.0 |
| LightOnOCR-3-0.8B | `loocr_grounding_omit_margins+pp3` | 85.4 | 0.8 | 85.0 |
| LightOnOCR-3-4B | `loocr_grounding_omit_margins` | 85.4 | 0.9 | 84.9 |
| chandra-ocr-2 | `datalab-eval-raw` | 85.1 | 0.9 | 84.4 |
| LightOnOCR-3-0.8B | `loocr_grounding_omit_margins` | 84.7 | 0.9 | 84.2 |
| LightOnOCR-3-1B | `loocr_grounding_omit_margins+pp3` | 84.5 | 0.9 | 84.0 |
| LightOnOCR-3-1B | `loocr_grounding_omit_margins` | 84.0 | 0.9 | 83.5 |
| mistral-ocr-4-1 | `mistral_ocr_4_1_no_hf` | 81.9 | 0.9 | 80.6 |
| mistral-ocr-4-1 | `mistral_ocr_4_1_annotation` | 72.4 | 1.0 | 80.1 |
| cohere-parse-5 | `cohere_parse_v5` | 71.1 | 1.1 | 76.8 |
| OvisOCR2 | `ovisocr2_vllm` | 70.5 | 1.0 | 78.1 |
<!-- /results -->

**fr-bench-pdf2md**: the mean over all categories.

<!-- results:fr-bench -->
| model | pipeline | all categories |
|---|---|---:|
| LightOnOCR-3-4B | `loocr_grounding_parse_with_layout` | 0.741 |
| LightOnOCR-3-0.8B | `loocr_grounding_parse_with_layout` | 0.705 |
| LightOnOCR-3-1B | `loocr_grounding_parse_with_layout` | 0.696 |
| chandra-ocr-2 | `chandra2_vllm` | 0.690 |
| dots.mocr | `dots_ocr_1_5_parse` | 0.593 |
| surya-ocr-2 | `surya2_sdk` | 0.541 |
| mistral-ocr-4-1 | `mistral_ocr_4_1_annotation` | 0.541 |
| jina-ocr-v1 | `jinaocr_vllm` | 0.532 |
<!-- /results -->

## Reproducing the results

Each row of `results/` has one command in [`reproduce/`](reproduce), one script per benchmark:

```bash
GPU=0 reproduce/parsebench.sh
GPU=0 GPUS=0,1 reproduce/olmocr-bench.sh   # GPUS: Infinity-Parser2-Pro runs on two GPUs
GPU=0 reproduce/fr-bench.sh
```

Each script downloads the pinned dataset, runs every published model into
`runs/reproduce-<date>/`, and scores them. The score table has an `in results/` column with the
published score for the same model and pipeline, so differences show directly. API models need
their key in `.env` (see Setup).

Every run records how it was made in `<output folder>/run.json`: the exact `ocr-bench run`
command, this repo's commit (`-dirty` with uncommitted changes), the pipeline, the model revision
and the docker image overrides.

## Requirements

- Linux with NVIDIA GPUs, and Docker with the NVIDIA container runtime (self-hosted models run in
  `vllm/vllm-openai` containers). API models (Mistral, Cohere) need no GPU.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and CPU-only PyTorch: GPU inference
  happens in the containers.

## Setup

```bash
cd benchmarks                        # from the repository root
uv sync
cp .env.example .env                 # API keys, and HF_TOKEN for gated checkpoints
uv run playwright install chromium   # olmOCR-bench math tests render with KaTeX
uv run ocr-bench download parsebench olmocr-bench fr-bench   # or just the ones you need
```

`pyproject.toml` and `uv.lock` pin every package (parse-bench at commit `78264d0`). Dataset
revisions, docker images and model revisions are pinned in
[`ocr_bench/config.py`](ocr_bench/config.py). External scorers and evaluation repos (benchpdf2md,
INF-MLLM, chandra) are cloned into `third_party/` at pinned commits.

## Usage

```bash
uv run ocr-bench run olmocr-bench OvisOCR2 --gpu 0    # serve, convert, stop the server
uv run ocr-bench score olmocr-bench                   # score every model of today's run
```

`ocr-bench run <benchmark> <model>` starts the model's vLLM server on the given GPUs (and a free
port), runs the benchmark through its ParseBench pipeline and stops the server. Outputs go to
`runs/<date>/<benchmark>/<model>/` and logs to `runs/<date>/logs/`. On olmOCR-bench and fr-bench,
pages that already have output are skipped, so a rerun resumes and retries failed pages.

```bash
uv run ocr-bench run fr-bench mistral-ocr-4-1:annotation                 # a pipeline variant
uv run ocr-bench run parsebench chandra-ocr-2 dots.mocr cohere-parse-5 --gpu 0,1   # in parallel
uv run ocr-bench run olmocr-bench surya-ocr-2 --gpu-memory-utilization 0.7         # unknown flags go to vllm serve
uv run ocr-bench score fr-bench --run 2026-09-28 --publish   # update results/ and the tables above
```

With several models, each self-hosted model takes the next GPUs of `--gpu`. `--run` names the run
folder (default: today). `ocr-bench <command> --help` lists every option.

| command | what it does |
|---|---|
| `ocr-bench download <benchmark>...` | Downloads the pinned datasets into `data/`. |
| `ocr-bench run <benchmark> <model>...` | Serves each model, runs the benchmark, stops the server. |
| `ocr-bench score <benchmark> [name...]` | Scores a run with the benchmark's scorer; `--publish` updates `results/` and the README. |
| `ocr-bench serve up/down/logs/print/list` | Manages a model's vLLM server by hand (`ocrbench-<model>` container, bound to 127.0.0.1). |
| `ocr-bench serve build <model>` | (Re)builds a model's custom image from `docker/`. `run` and `serve up` build it when it is missing. |
| `ocr-bench predict jina/surya` | The `/predict` endpoint the `jinaocr` and `surya2` providers call, in front of the vLLM server (`run` starts it). |
| `ocr-bench parsebench ...` | The `parse-bench` CLI, plus this repo's providers and pipelines. |

To keep a server up across runs, start it with `ocr-bench serve up <model>` and pass `--no-serve`
to `ocr-bench run`.

## Models

| model | ParseBench pipelines (`<model>:<variant>`) | serving |
|---|---|---|
| chandra-ocr-2 | `chandra2_vllm` | vLLM |
| surya-ocr-2 | `surya2_sdk` | vLLM + `ocr-bench predict surya` |
| dots.mocr | `dots_ocr_1_5_parse` | vLLM |
| OvisOCR2 | `ovisocr2_vllm` | vLLM |
| jina-ocr-v1 | `jinaocr_vllm` | vLLM + `ocr-bench predict jina` |
| KDL-Frontier-Parser-nano | `kdl_frontier_nano` | vLLM |
| Infinity-Parser2-Pro | `infinity_parser2_pro` | vLLM, 2 GPUs, port 8000 |
| LightOnOCR (any Qwen3-based checkpoint) | `loocr_grounding_parse_with_layout`, `:omit-margins`, `:omit-strict`, `:raw` | vLLM (`--model-id`) |
| LightOnOCR-1B (any Pixtral-based checkpoint) | same as LightOnOCR | vLLM + transformers 5.16.1 (`--model-id`) |
| mistral-ocr-4-1 | `mistral_ocr_4_1`, `:annotation`, `:no-hf` | API (`MISTRAL_API_KEY`) |
| cohere-parse-5 | `cohere_parse_v5` | API (`COHERE_API_KEY`) |

`ocr-bench serve list` shows each server's image, default port and serving source.

### LightOnOCR checkpoints

Two entries serve any LightOnOCR checkpoint from its Hugging Face repo, at an exact commit. They
run the same command and pipelines, and differ only in the docker image. Pick the entry that
matches the checkpoint's architecture (`text_config.model_type` in its `config.json`):

- `LightOnOCR`: Qwen3-based checkpoints (e.g. `LightOnOCR-3-4B-*`), on `vllm/vllm-openai:v0.30.0`.
- `LightOnOCR-1B`: Mistral3/Pixtral-based checkpoints (e.g. `LightOnOCR-3-1B-*`), on
  `docker/lightonocr-1b.Dockerfile` (`ocrbench/vllm-openai:v0.30.0-transformers5.16.1`).

You don't need to build anything first: `ocr-bench run` and `ocr-bench serve up` build the
`LightOnOCR-1B` image the first time they need it. A lock makes parallel runs build it only once.

```bash
uv run ocr-bench run olmocr-bench LightOnOCR --model-id lightonai/LightOnOCR-3-4B --revision <commit> --gpu 1
uv run ocr-bench run olmocr-bench LightOnOCR-1B:omit-margins --model-id lightonai/LightOnOCR-3-1B --revision <commit>
```

The output folder is the repo name (plus `-<variant>`); `--name` changes it. For private repos,
set `HF_TOKEN` in `.env`.

To run the server alone (to query it yourself, or to share it across runs with `--no-serve`):

```bash
uv run ocr-bench serve up LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision <commit> --gpu 0
# ready: http://127.0.0.1:8811/v1, model name "loocr-grounding"
uv run ocr-bench run fr-bench LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision <commit> --no-serve
uv run ocr-bench serve logs LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision <commit>
uv run ocr-bench serve down LightOnOCR-1B --model-id lightonai/LightOnOCR-3-1B --revision <commit>
uv run ocr-bench serve build LightOnOCR-1B   # rebuild the image after editing its Dockerfile
```

### Local providers

These providers live in `ocr_bench/providers/` and are registered by every `ocr-bench` command:

- **`loocr_grounding`**: LightOnOCR-3's output (`![label](x1,y1,x2,y2) text` blocks) turned into
  markdown and layout. One call scores all five ParseBench dimensions. Pages render at 400 DPI,
  capped at 5M pixels, sampled at temperature 0.1. On olmOCR-bench and fr-bench, the `omit-margins` variant drops `header`,
  `footer`, `page_number`, `page_header`, `page_footer`, `header_image`, `footer_image` and
  `aside_text` blocks and keeps footnotes; `omit-strict` drops the same
  page margins plus `footnote` / `page_footnote` and keeps `aside_text`; `raw` keeps the block markers.
  Post-processing (`lightonocr/postprocess/`, applied everywhere except `raw`): escaped math
  delimiters are repaired (`\$x\$` -> `$x$`). On olmOCR-bench, `omit-margins` and `omit-strict` also get PP3:
  the same dollar repair, then markdown formatting rules tuned on olmOCR-bench. The run writes the
  output before PP3 to `<name>-nopp` and after it to `<name>` (pipeline `loocr_grounding_omit_<variant>+pp3`),
  so `ocr-bench score` shows both.
- **`mistral_ocr_hf`**: Mistral OCR with the API's `extract_header` / `extract_footer`, which
  moves page headers and footers out of the markdown (`mistral-ocr-4-1:no-hf`).

## Methodology

- **ParseBench** scores ParseBench's normalized output with its own evaluators.
- **olmOCR-bench and fr-bench-pdf2md** score the model's raw output. The ParseBench provider runs
  the inference, but ParseBench's `normalize` step is skipped because it is tuned for ParseBench's
  scorers. Layout wrappers (bboxes, labels, JSON) are removed, and every block's text is kept
  verbatim (`ocr_bench/runner.py`), apart from LightOnOCR's post-processing (see `loocr_grounding`).
  Scoring uses upstream `olmocr.bench.benchmark` and benchpdf2md.
- **Infinity-Parser2-Pro on olmOCR-bench** uses INF-MLLM's own evaluation
  (`uv run scripts/olmocr-bench-inf-mllm.py --gpu 0,1`), with and without its post-processing.
- **chandra-ocr-2 on olmOCR-bench** also runs with Datalab's own evaluation
  (`uv run scripts/olmocr-bench-datalab.py`, 300 DPI), with and without its post-processing.

## Adding a model

1. If ParseBench has no provider for the model, add one in `ocr_bench/providers/` and register
   its pipeline in `ocr_bench/providers/__init__.py`.
2. Add a raw-markdown extractor for the provider to `RAW_MARKDOWN` in `ocr_bench/runner.py`.
   olmOCR-bench and fr-bench need it.
3. In `ocr_bench/config.py`, add a `Server` with the model's documented serving command (for a
   self-hosted model) and a `Model` with its pipelines and the env var its provider reads the
   server URL from.
4. `ocr-bench run` it on each benchmark, then `ocr-bench score <benchmark> --publish`.
5. Add the `ocr-bench run` command of each published row to `reproduce/<benchmark>.sh`.

## Layout

```
ocr_bench/config.py   benchmarks, vLLM servers and models: what to edit to add a model
ocr_bench/            the ocr-bench CLI and local ParseBench providers
docker/               images and launchers for models that need more than stock vLLM
reproduce/            one script per benchmark: the command behind every published score
scripts/              olmOCR-bench runs with the model authors' own evaluation code
results/              published scores (CSV)
data/, runs/, third_party/   datasets, outputs, pinned upstream checkouts (gitignored)
```

## Notes

- fr-bench-pdf2md's current revision has 2318 tests and a `long_table2` category, while the
  published leaderboard uses 2335 tests. The scores here are not directly comparable with it.
- dots.mocr is served as `dots-ocr-1.5`, the model name the `dots_ocr_1_5_parse` pipeline requests.
- Infinity-Parser2-Pro needs transformers 5 (`docker/infinity-parser2.Dockerfile`) and
  `enable_thinking: false`, which its `vllm-server` backend does not send. Its provider requests
  `localhost:8000`, so its server always uses that port.
- jina-ocr-v1 is served by `docker/jina_vllm_server.py`. It applies the model card's
  `register()` and forces FlashAttention-2.
- `ocr-bench predict surya` sizes surya's batches for an H100 (`VLLM_GPU_TYPE=h100`); set
  `VLLM_GPU_TYPE` for other GPUs.
- LightOnOCR does not work on vLLM v0.27.1, whose compiled mode makes it loop; its servers are
  pinned to v0.30.0. That image ships transformers 5.17, which renamed the `PixtralRotaryEmbedding`
  class that vLLM's `pixtral.py` imports, so `LightOnOCR-1B` pins transformers 5.16.1.
- The 300 DPI pipelines (`cohere_parse_v5`, `infinity_parser2_pro`) fail on pages that exceed
  PIL's pixel limit.

## License

[Apache-2.0](../LICENSE). The benchmarks, models and upstream evaluation code keep their own licenses.

## Acknowledgements

This repo builds on [ParseBench](https://github.com/run-llama/ParseBench),
[olmOCR](https://github.com/allenai/olmocr), [benchpdf2md](https://github.com/ld-lab-pulsia/benchpdf2md),
[INF-MLLM](https://github.com/infly-ai/INF-MLLM) and [chandra](https://github.com/datalab-to/chandra),
and on the model authors' serving instructions.
