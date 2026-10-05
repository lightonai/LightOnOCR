"""``/predict`` servers for ParseBench's ``jinaocr`` and ``surya2`` providers.

Those providers POST ``{"image_base64": ...}`` and expect ``{"markdown": ...}``; upstream's
endpoints are not published. Each server sits in front of an ``ocr-bench serve`` vLLM container.
``ocr-bench run`` starts them itself; by hand:

    ocr-bench predict jina    # in front of `ocr-bench serve up jina-ocr-v1`, on its port + 10
    ocr-bench predict surya   # in front of `ocr-bench serve up surya-ocr-2`, on its port + 10
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import json
import os
import traceback
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image

from ocr_bench.config import SERVERS

SERVER = {"jina": "jina-ocr-v1", "surya": "surya-ocr-2"}  # sidecar -> the vLLM server it calls

# jina-ocr-v1: request and sampling as in the model card.
JINA_MODEL = "jinaai/jina-ocr-v1"
JINA_DEFAULT_PROMPT = "Transcribe the provided document image into a clean Markdown format, preserving the natural reading order."
JINA_REPETITION_DETECTION = {"max_pattern_size": 35, "min_pattern_size": 35, "min_count": 10}


def make_jina(base_url: str) -> Callable[[dict], dict]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=600)

    def predict(req: dict) -> dict:
        image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"])))
        resp = client.chat.completions.create(
            model=JINA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{req['image_base64']}"}},
                        {"type": "text", "text": req.get("prompt") or JINA_DEFAULT_PROMPT},
                    ],
                }
            ],
            max_tokens=req.get("max_tokens") or 4096,
            temperature=0.0,
            extra_body={"repetition_penalty": 1.05, "repetition_detection": JINA_REPETITION_DETECTION},
        )
        return {
            "markdown": (resp.choices[0].message.content or "").strip(),
            "image_width": image.width,
            "image_height": image.height,
        }

    return predict


# surya-ocr-2: full-page recognition through the surya SDK. Surya's page HTML uses chandra's
# div/data-label format, so chandra's parse_markdown converts it to markdown.
def make_surya(base_url: str, include_headers_footers: bool) -> Callable[[dict], dict]:
    os.environ["SURYA_INFERENCE_BACKEND"] = "vllm"
    os.environ["SURYA_INFERENCE_URL"] = base_url
    os.environ.setdefault("VLLM_GPU_TYPE", "h100")  # surya's batch sizes for the GPU type
    os.environ.setdefault("DISABLE_TQDM", "true")
    from chandra.output import parse_markdown
    from surya.inference import SuryaInferenceManager
    from surya.layout.label import LAYOUT_PRED_RELABEL
    from surya.recognition import RecognitionPredictor

    manager = SuryaInferenceManager(method="vllm")
    manager.start()
    predictor = RecognitionPredictor(manager)
    canon_to_raw = {v: k for k, v in LAYOUT_PRED_RELABEL.items()}

    def predict(req: dict) -> dict:
        image = Image.open(io.BytesIO(base64.b64decode(req["image_base64"]))).convert("RGB")
        w, h = image.size
        page = predictor([image], full_page=True)[0]
        blocks = sorted(page.blocks, key=lambda b: b.reading_order)
        divs = []
        for b in blocks:
            label = b.raw_label or canon_to_raw.get(b.label, b.label)
            x0, y0, x1, y1 = b.bbox
            bbox = " ".join(str(round(v)) for v in (x0 / w * 1000, y0 / h * 1000, x1 / w * 1000, y1 / h * 1000))
            divs.append(f'<div data-bbox="{bbox}" data-label="{html_lib.escape(label)}">{b.html}</div>')
        page_html = "\n".join(divs)
        return {
            "markdown": parse_markdown(page_html, include_headers_footers=include_headers_footers),
            "html": page_html,
            "blocks": [{"bbox": list(b.bbox), "label": b.label, "html": b.html} for b in blocks],
            "image_width": w,
            "image_height": h,
        }

    return predict


def serve(predict: Callable[[dict], dict], host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out: dict[str, Any] = {**predict(body), "status": "success"}
                code = 200
            except Exception as e:
                traceback.print_exc()
                out, code = {"status": "error", "error": f"{type(e).__name__}: {e}"}, 500
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    ThreadingHTTPServer.daemon_threads = True
    print(f"ready: http://{host}:{port}/predict", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="ocr-bench predict", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model", choices=list(SERVER))
    p.add_argument("--port", type=int, help="default: the vLLM server's default port + 10")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--backend", help="vLLM base URL (default: the `ocr-bench serve` default port)")
    p.add_argument("--include-headers-footers", action="store_true", help="surya only")
    a = p.parse_args(argv)
    vllm_port = SERVERS[SERVER[a.model]].port
    backend = a.backend or f"http://127.0.0.1:{vllm_port}/v1"
    predict = make_jina(backend) if a.model == "jina" else make_surya(backend, a.include_headers_footers)
    serve(predict, a.host, a.port or vllm_port + 10)


if __name__ == "__main__":
    main()
