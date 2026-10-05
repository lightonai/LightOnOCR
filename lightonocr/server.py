"""HTTP server behind ``lightonocr viewer``: the web viewer, the runs in out/, and the jobs that create runs.

Routes (JSON unless noted):
    GET     /                              the viewer (static/index.html)
    GET     /static/<file>                 its stylesheet, script, logo and favicon
    GET     /api/info                      {"out", "base_url"}
    GET     /api/settings                  the model server: {"base_url", "model", "serving", "grounding", "reachable", "models", "error"}
    PUT     /api/settings                  body {"base_url", "model"}: switch model server, save to out/settings.json
    GET     /api/runs                      runs found in out/, newest first
    POST    /api/runs?name=&mode=&pages=   body: the PDF or image bytes. Starts a job and returns it
    GET     /api/runs/<name>               one run with its pages: image URL, size, raw output, parsed blocks
    DELETE  /api/runs/<name>               remove the run's folder
    GET     /files/<name>/<file>           a run's files (page images, uploaded source)
    GET     /api/jobs                      jobs of this server process, newest first
    DELETE  /api/jobs/<id>                 forget a finished or failed job

Standard library only. A run created here goes through the same pipeline as ``lightonocr run``.
"""

from __future__ import annotations

import errno
import json
import mimetypes
import re
import shutil
import sys
import threading
import time
import unicodedata
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from openai import APIConnectionError
from PIL import Image

from .client import LightOnOCR
from .models import check_mode
from .pipeline import finish, ocr_pages, parse_pages
from .render import load_pages
from .settings import write_settings

STATIC = Path(__file__).with_name("static")
DEFAULT_PORT = 8000
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """A run name is its folder in out/: ASCII letters, digits, '.', '_' and '-', never empty or dot-only.
    Accents are transliterated ("Étude" -> "Etude"); anything else becomes '-'."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = UNSAFE.sub("-", name).strip("._-")
    return name or "document"


def unique_name(out: Path, name: str) -> str:
    """``name``, or ``name-2``, ``name-3``... if that folder exists. Runs are never overwritten."""
    candidate, i = name, 1
    while (out / candidate).exists():
        i += 1
        candidate = f"{name}-{i}"
    return candidate


# ---- Runs: one folder of out/ each ---------------------------------------------------------------


def read_meta(run: Path) -> dict:
    try:
        return json.loads((run / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run_summary(run: Path) -> dict | None:
    """What the sidebar shows, or None if the folder holds no pages."""
    pages = list(run.glob("page-*.md")) if run.is_dir() else []
    if not pages:
        return None
    meta = read_meta(run)
    mode = meta.get("mode") or ("grounding" if any(run.glob("page-*.json")) else "plain")
    return {"name": run.name, "title": meta.get("title", run.name), "mode": mode, "page_count": len(pages),
            "modified": max(p.stat().st_mtime for p in pages)}


def list_runs(out: Path) -> list[dict]:
    runs = [summary for summary in map(run_summary, out.iterdir()) if summary]
    return sorted(runs, key=lambda r: -r["modified"])


def load_run(out: Path, name: str) -> dict | None:
    """The run as the viewer wants it: its summary plus one entry per page."""
    run = out / name
    summary = run_summary(run)
    if not summary:
        return None
    meta = read_meta(run)
    numbers = meta.get("pages") or sorted(int(p.stem[5:]) for p in run.glob("page-*.md"))
    pages = []
    for n in numbers:
        stem = f"page-{n:03d}"
        md, js, png = run / f"{stem}.md", run / f"{stem}.json", run / f"{stem}.png"
        if not md.is_file():
            continue
        blocks = width = height = None
        if summary["mode"] == "grounding" and js.is_file():
            data = json.loads(js.read_text(encoding="utf-8"))
            blocks, width, height = data["blocks"], data["width"], data["height"]
        if width is None and png.is_file():
            with Image.open(png) as im:  # reads the header only
                width, height = im.size
        pages.append({"page": n, "image": f"/files/{name}/{stem}.png", "width": width or 1000, "height": height or 1000,
                      "raw": md.read_text(encoding="utf-8"), "blocks": blocks})
    return {**summary, "pages": pages}


# ---- Server: the runs folder, the model client, the jobs ----------------------------------------


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], out: Path, ocr: LightOnOCR, *, longest_edge: int | None, temperature: float, concurrency: int):
        super().__init__(address, Handler)
        self.out, self.ocr = out, ocr
        self.longest_edge, self.temperature, self.concurrency = longest_edge, temperature, concurrency  # longest_edge None: the model's
        self.jobs: list[dict] = []  # newest first

    def describe_endpoint(self) -> dict:
        """The model server as configured, and whether it answers right now (a few seconds at most)."""
        ocr = self.ocr
        status = ocr.check()
        if status["reachable"] and ocr.known_model is None and status["models"]:
            ocr.known_model = status["models"][0]
        # grounding: whether the served model has the mode (LightOnOCR-3), so the viewer can offer it or not.
        return {"base_url": ocr.base_url, "model": ocr.model_setting, "serving": ocr.known_model, "grounding": ocr.profile.grounding, **status}

    def start_job(self, name: str, title: str, mode: str, pages: list[int] | None, src: Path) -> dict:
        job = {"id": uuid.uuid4().hex[:8], "name": name, "title": title, "mode": mode, "status": "running",
               "done": 0, "total": None, "error": None, "started": time.time()}
        self.jobs.insert(0, job)
        threading.Thread(target=self.work, args=(job, pages, src), daemon=True).start()
        return job

    def work(self, job: dict, pages: list[int] | None, src: Path) -> None:
        run = self.out / job["name"]
        try:
            images = load_pages(src, self.longest_edge or self.ocr.profile.longest_edge, pages)
            numbers = pages or list(range(1, len(images) + 1))
            job["total"] = len(images)
            results = []
            # Completion order, so the progress counter moves as soon as any page is back.
            for page in ocr_pages(self.ocr, run, numbers, images, job["mode"], concurrency=self.concurrency, ordered=False, temperature=self.temperature):
                results.append(page)
                job["done"] += 1
            finish(run, job["title"], job["mode"], results)
            job["status"] = "done"
        except APIConnectionError:  # shown in the sidebar; the server keeps running
            job["status"], job["error"] = "error", f"cannot reach {self.ocr.base_url}. Is vLLM running?"
        except Exception as e:
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        if job["status"] == "error" and not any(run.glob("page-*.md")):  # nothing usable was produced
            shutil.rmtree(run, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def do_PUT(self) -> None:
        self.dispatch("PUT")

    def do_DELETE(self) -> None:
        self.dispatch("DELETE")

    def dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        self.query = {k: v[0] for k, v in parse_qs(url.query).items()}
        for route_method, pattern, handler in ROUTES:
            match = re.fullmatch(pattern, unquote(url.path))
            if match and route_method == method:
                return handler(self, *match.groups())
        self.fail(HTTPStatus.NOT_FOUND, "not found")

    # -- routes ---------------------------------------------------------------------------------

    def index(self) -> None:
        self.send_file(STATIC / "index.html")

    def static(self, file: str) -> None:
        self.send_file(STATIC / file)

    def favicon(self) -> None:
        self.send_file(STATIC / "favicon.png")

    def info(self) -> None:
        self.send_json({"out": str(self.server.out.resolve()), "base_url": self.server.ocr.base_url})

    def settings(self) -> None:
        self.send_json(self.server.describe_endpoint())

    def update_settings(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            base_url, model = (str(body.get("base_url") or "").strip(), str(body.get("model") or "").strip())
        except (ValueError, AttributeError):
            return self.fail(HTTPStatus.BAD_REQUEST, "body must be JSON with base_url and model")
        if base_url and not base_url.startswith(("http://", "https://")):
            return self.fail(HTTPStatus.BAD_REQUEST, "base_url must start with http:// or https://")
        write_settings(self.server.out, base_url=base_url, model=model)
        self.server.ocr = LightOnOCR(base_url=base_url or None, model=model or None)  # empty = env or default
        self.send_json(self.server.describe_endpoint())

    def runs(self) -> None:
        self.send_json(list_runs(self.server.out))

    def run(self, name: str) -> None:
        run = load_run(self.server.out, name) if safe_name(name) == name else None
        self.send_json(run) if run else self.fail(HTTPStatus.NOT_FOUND, "no such run")

    def delete_run(self, name: str) -> None:
        if safe_name(name) != name or not (self.server.out / name).is_dir():
            return self.fail(HTTPStatus.NOT_FOUND, "no such run")
        shutil.rmtree(self.server.out / name)
        self.send_status(HTTPStatus.NO_CONTENT)

    def run_file(self, name: str, file: str) -> None:
        # The name is validated and the file part has no separator, so the path cannot leave out/.
        if safe_name(name) != name or file.startswith("."):
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        self.send_file(self.server.out / name / file)

    def create_run(self) -> None:
        mode = self.query.get("mode", "grounding")
        try:
            check_mode(self.server.ocr.model, mode)  # grounding exists on LightOnOCR-3 only
        except ValueError as e:
            return self.fail(HTTPStatus.BAD_REQUEST, str(e))
        try:
            pages = parse_pages(self.query["pages"]) if self.query.get("pages") else None
        except ValueError:
            return self.fail(HTTPStatus.BAD_REQUEST, "pages must look like 1,3-5")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self.fail(HTTPStatus.BAD_REQUEST, "empty body")
        filename = Path(unquote(self.headers.get("X-Filename") or "document.pdf")).name
        name = unique_name(self.server.out, safe_name(self.query.get("name") or Path(filename).stem))
        run = self.server.out / name
        run.mkdir(parents=True)
        src = run / f"source{Path(filename).suffix.lower() or '.pdf'}"
        src.write_bytes(self.rfile.read(length))
        self.send_json(self.server.start_job(name, filename, mode, pages, src), HTTPStatus.CREATED)

    def jobs(self) -> None:
        self.send_json(self.server.jobs)

    def delete_job(self, job_id: str) -> None:
        self.server.jobs = [j for j in self.server.jobs if j["id"] != job_id or j["status"] == "running"]
        self.send_status(HTTPStatus.NO_CONTENT)

    # -- responses ------------------------------------------------------------------------------

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            return self.fail(HTTPStatus.NOT_FOUND, "not found")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or "javascript" in content_type:
            content_type += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), content_type)

    def send_json(self, obj, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_status(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def fail(self, status: HTTPStatus, message: str) -> None:
        self.send_bytes(message.encode("utf-8"), "text/plain; charset=utf-8", status)

    def log_message(self, format: str, *args) -> None:
        if "/api/jobs" not in self.path:  # progress polling is noise
            super().log_message(format, *args)


ROUTES = [
    ("GET", r"/", Handler.index),
    ("GET", r"/static/([\w.-]+)", Handler.static),
    ("GET", r"/favicon.ico", Handler.favicon),
    ("GET", r"/api/info", Handler.info),
    ("GET", r"/api/settings", Handler.settings),
    ("PUT", r"/api/settings", Handler.update_settings),
    ("GET", r"/api/runs", Handler.runs),
    ("POST", r"/api/runs", Handler.create_run),
    ("GET", r"/api/runs/([\w.-]+)", Handler.run),
    ("DELETE", r"/api/runs/([\w.-]+)", Handler.delete_run),
    ("GET", r"/files/([\w.-]+)/([\w.-]+)", Handler.run_file),
    ("GET", r"/api/jobs", Handler.jobs),
    ("DELETE", r"/api/jobs/(\w+)", Handler.delete_job),
]


def serve(out: Path, host: str, port: int | None, ocr: LightOnOCR, *, longest_edge: int | None, temperature: float, concurrency: int) -> int:
    """Serve until Ctrl-C. ``port=None`` means 8000 or the next free port; a busy explicit port is an error."""
    out.mkdir(parents=True, exist_ok=True)
    for candidate in [port] if port else range(DEFAULT_PORT, DEFAULT_PORT + 20):
        try:
            server = Server((host, candidate), out, ocr, longest_edge=longest_edge, temperature=temperature, concurrency=concurrency)
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
    else:
        wanted = f"port {port}" if port else f"ports {DEFAULT_PORT}-{DEFAULT_PORT + 19}"
        print(f"error: {wanted} on {host} already in use. Pick another with --port.", file=sys.stderr)
        return 1
    print(f"viewer: http://{host}:{server.server_port}/ | runs in {out.resolve()} | model server {ocr.base_url}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
