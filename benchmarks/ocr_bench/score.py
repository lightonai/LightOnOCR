"""Score a run's outputs with each benchmark's own scorer.

    ocr-bench score olmocr-bench                    # every model in runs/<today>/olmocr-bench
    ocr-bench score fr-bench chandra-ocr-2 --run 2026-09-28
    ocr-bench score parsebench --publish            # also update results/ and the README tables

Prints the scores and writes them, in the ``results/<benchmark>.csv`` columns, to ``<dir>/scores.csv``.

* ParseBench: reads the evaluation reports ``ocr-bench run`` wrote (Overall is the mean of the five
  leaderboard dimensions).
* olmOCR-bench: upstream ``olmocr.bench.benchmark``. The overall score is the mean over the JSONL
  files and baseline, with its 95% CI; the second score leaves out ``headers_footers``.
* fr-bench: upstream benchpdf2md, in its own venv (its dependencies conflict with ours).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ocr_bench import results
from ocr_bench.config import BENCHMARKS, RUNS, THIRD_PARTY

BENCHPDF2MD = ("ld-lab-pulsia/benchpdf2md", "ceab1548ac914bff6585fa5c6227e2b24c35cfdb")
BENCHPDF2MD_VENV = Path(os.environ.get("BENCHPDF2MD_VENV", "~/.cache/ocr-bench-benchpdf2md-venv")).expanduser()


def checkout(repo: str, commit: str) -> Path:
    """``third_party/<name>`` at ``commit`` of github.com/<repo>."""
    path = THIRD_PARTY / repo.split("/")[1]
    if not path.is_dir():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", f"https://github.com/{repo}.git", str(path)], check=True)
    if subprocess.run(["git", "-C", str(path), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        subprocess.run(["git", "-C", str(path), "fetch", "-q", "origin"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], check=True)
    return path


def _meta(folder: Path) -> dict[str, str]:
    """Model and pipeline from the run.json ``ocr-bench run`` writes (the folder name otherwise)."""
    meta = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else {}
    return {"model": meta.get("model", folder.name), "pipeline": meta.get("pipeline", "")}


def _folders(bench_dir: Path, names: list[str], has_output) -> list[Path]:
    if names:
        return [bench_dir / n for n in names]
    return sorted(d for d in bench_dir.iterdir() if d.is_dir() and not d.is_symlink() and has_output(d))


def _today() -> str:
    return datetime.date.today().isoformat()


# ParseBench leaderboard column -> (category, metric), as in parse_bench.analysis.
PARSEBENCH_DIMS = {
    "tables": ("table", "grits_trm_composite"),
    "charts": ("chart", "rule_pass_rate"),
    "content_faithfulness": ("text_content", "content_faithfulness"),
    "semantic_formatting": ("text_formatting", "semantic_formatting"),
    "visual_grounding": ("layout", "layout_element_rule_pass_rate"),
}


def score_parsebench(bench_dir: Path, names: list[str]) -> list[dict[str, str]]:
    direct_url = json.loads(importlib.metadata.distribution("parse-bench").read_text("direct_url.json") or "{}")
    scorer = f"parse-bench {direct_url.get('vcs_info', {}).get('commit_id', '?')[:7]}"
    rows = []
    for folder in _folders(bench_dir, names, lambda d: any(d.glob("*/_summary.json"))):
        summary = next(folder.glob("*/_summary.json"))
        dims = {}
        for col, (category, metric) in PARSEBENCH_DIMS.items():
            report = summary.parent / category / "_evaluation_report.json"
            value = json.loads(report.read_text())["aggregate_metrics"].get(f"avg_{metric}") if report.exists() else None
            dims[col] = "" if value is None else f"{100 * value:.2f}"
        overall = f"{sum(map(float, dims.values())) / len(dims):.2f}" if all(dims.values()) else ""
        row = {**_meta(folder), "pipeline": summary.parent.name, "overall": overall, **dims}
        row |= {"failed": str(json.loads(summary.read_text())["failed"]), "scorer": scorer, "date": _today()}
        rows.append(row)
    return rows


def parse_olmocr_log(text: str) -> dict[str, dict[str, str]]:
    """Per-candidate scores from olmocr.bench.benchmark's "Final Summary" (it has no machine-readable output)."""
    out = {}
    for block in re.split(r"\n(?=\S+\s+: Average Score)", text.split("Final Summary", 1)[1])[1:]:
        name, _, ci = re.match(r"(\S+)\s+: Average Score: ([\d.]+)% ± ([\d.]+)%", block).groups()
        counts = re.findall(r"(\S+?)(?:\.jsonl)?\s+: [\d.]+% \((\d+)/(\d+) tests\)", block)
        scores = {("tables" if k == "table_tests" else k): 100 * int(a) / int(b) for k, a, b in counts}
        cats = sorted(k for k in scores if k != "baseline") + ["baseline"]
        no_hf = [scores[k] for k in cats if k != "headers_footers"]
        out[name] = {
            "overall": f"{sum(scores.values()) / len(scores):.1f}",
            "ci95": ci,
            "overall_excl_headers_footers": f"{sum(no_hf) / len(no_hf):.1f}",
            **{k: f"{scores[k]:.1f}" for k in cats},
        }
    return out


def score_olmocr(bench_dir: Path, names: list[str], olmocr_src: Path | None = None) -> list[dict[str, str]]:
    """Runs upstream olmocr.bench.benchmark (from ``olmocr_src`` when given, else the installed olmocr)."""
    if olmocr_src:
        commit = subprocess.run(["git", "-C", str(olmocr_src), "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True)
        scorer = f"olmocr {commit.stdout.strip()}"
    else:
        scorer = f"olmocr {importlib.metadata.version('olmocr')}"
    folders = _folders(bench_dir, names, lambda d: any(d.glob("**/*.md")))
    env = {**os.environ, "PYTHONPATH": str(olmocr_src)} if olmocr_src else None
    (bench_dir.parent / "logs").mkdir(exist_ok=True)
    scores = {}
    for candidate in [f.name for f in folders] if names else [None]:
        log = bench_dir.parent / "logs" / f"{bench_dir.name}.score{'.' + candidate if candidate else ''}.log"
        cmd = [sys.executable, "-m", "olmocr.bench.benchmark", "--dir", str(bench_dir)]
        cmd += ["--candidate", candidate] if candidate else []
        # Log to a file, not a pipe: the KaTeX renderer makes a shared stdout non-blocking.
        with open(log, "w") as f:
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode:
                raise SystemExit(f"olmocr.bench.benchmark failed: see {log}")
        scores |= parse_olmocr_log(log.read_text())
    rows = []
    for folder in folders:
        if folder.name not in scores:
            print(f"{folder.name}: no score (missing pages?), see {bench_dir.parent / 'logs'}", file=sys.stderr)
            continue
        rows.append({**_meta(folder), **scores[folder.name], "scorer": scorer, "date": _today()})
    return rows


FR_BENCH_META = {"total_time", "num_pages", "num_doc_processed", "num_tests", "avg_result", "avg_doc_latency",
                 "avg_page_latency", "avg_time_per_page", "dpi", "concurrency", "model"}  # fmt: skip


def score_fr_bench(bench_dir: Path, names: list[str]) -> list[dict[str, str]]:
    repo = checkout(*BENCHPDF2MD)
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(BENCHPDF2MD_VENV)}
    subprocess.run(["uv", "sync", "-q", "--frozen"], cwd=repo, env=env, check=True)
    folders = _folders(bench_dir, names, lambda d: (d / "md").is_dir())
    script = Path(__file__).parent / "scorers" / "fr_bench.py"
    data = BENCHMARKS["fr-bench"].data
    subprocess.run([BENCHPDF2MD_VENV / "bin" / "python", script, data, *folders], check=True)
    rows = []
    for folder in folders:
        m = json.loads(max((folder / "test_results").glob("*/metrics.json")).read_text())
        cats = ["baseline", *sorted(k for k in m if k not in FR_BENCH_META and k != "baseline")]
        rows.append({
            **_meta(folder),
            "all_categories": f"{m['avg_result']:.3f}",
            **{c: f"{m[c]:.3f}" for c in cats},
            "pages": str(m["num_doc_processed"]),
            "scorer": f"benchpdf2md {BENCHPDF2MD[1][:7]}",
            "date": _today(),
        })  # fmt: skip
    return rows


PRINT_ONLY = {"failed", "pages", "in results/"}  # shown, not published


def report(bench: str, rows: list[dict[str, str]], out: Path, publish: bool) -> None:
    if not rows:
        raise SystemExit("nothing to score")
    # Next to each score, the published one for the same (model, pipeline): what a reproduction compares to.
    main = results.TABLES[bench][0]
    published_rows = {(r["model"], r["pipeline"]): r for r in results.load(bench)[1]}
    for r in rows:
        r["in results/"] = published_rows.get((r["model"], r["pipeline"]), {}).get(main, "")
    columns = list(dict.fromkeys(c for r in rows for c in r))
    print("| " + " | ".join(columns) + " |\n|" + "---|" * len(columns))
    for r in rows:
        print("| " + " | ".join(r.get(c, "") for c in columns) + " |")
    published = [{k: v for k, v in r.items() if k not in PRINT_ONLY} for r in rows]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, [c for c in columns if c not in PRINT_ONLY], lineterminator="\n")
        writer.writeheader()
        writer.writerows(published)
    print(f"\nwrote {out}")
    if publish:
        results.upsert(bench, published)
        results.render_readme()
        print(f"updated results/{bench}.csv and the README tables")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ocr-bench score", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=list(BENCHMARKS))
    p.add_argument("names", nargs="*", help="output folders to score (default: all)")
    p.add_argument("--run", default=datetime.date.today().isoformat(), help="run folder in runs/ (default: today)")
    p.add_argument("--dir", type=Path, help="benchmark output dir (default: runs/<run>/<benchmark>)")
    p.add_argument("--publish", action="store_true", help="update results/<benchmark>.csv and the README tables")
    a = p.parse_args(argv)
    bench_dir = a.dir or RUNS / a.run / a.benchmark
    if not bench_dir.is_dir():
        p.error(f"no outputs in {bench_dir}")
    score = {"parsebench": score_parsebench, "olmocr-bench": score_olmocr, "fr-bench": score_fr_bench}[a.benchmark]
    report(a.benchmark, score(bench_dir, a.names), bench_dir / "scores.csv", a.publish)


if __name__ == "__main__":
    main()
