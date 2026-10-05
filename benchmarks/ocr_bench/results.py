"""Published scores: ``results/<benchmark>.csv``, and the README tables generated from them."""

from __future__ import annotations

import csv
import re

from ocr_bench.config import RESULTS, ROOT

# Benchmark -> (sort column, README columns as (header, CSV column)).
TABLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "parsebench": ("overall", [
        ("model", "model"), ("pipeline", "pipeline"), ("Overall", "overall"), ("Tables", "tables"),
        ("Charts", "charts"), ("Content", "content_faithfulness"), ("Formatting", "semantic_formatting"),
        ("Grounding", "visual_grounding"), ("published", "published_overall"),
    ]),
    "olmocr-bench": ("overall", [
        ("model", "model"), ("pipeline", "pipeline"), ("overall", "overall"), ("±", "ci95"),
        ("excl. headers/footers", "overall_excl_headers_footers"),
    ]),
    "fr-bench": ("all_categories", [("model", "model"), ("pipeline", "pipeline"), ("all categories", "all_categories")]),
}  # fmt: skip


def _num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return float("-inf")


def load(bench: str) -> tuple[list[str], list[dict[str, str]]]:
    path = RESULTS / f"{bench}.csv"
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def upsert(bench: str, new_rows: list[dict[str, str]]) -> None:
    """Add or replace rows by (model, pipeline). Columns the new row lacks (e.g. published_overall) are kept."""
    columns, rows = load(bench)
    by_key = {(r["model"], r["pipeline"]): r for r in rows}
    for new in new_rows:
        columns += [c for c in new if c not in columns]
        by_key[new["model"], new["pipeline"]] = {**by_key.get((new["model"], new["pipeline"]), {}), **new}
    sort_col = TABLES[bench][0]
    rows = sorted(by_key.values(), key=lambda r: -_num(r.get(sort_col, "")))
    with open(RESULTS / f"{bench}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, columns, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(bench: str, rows: list[dict[str, str]]) -> str:
    cols = TABLES[bench][1]
    numeric = [h not in ("model", "pipeline") for h, _ in cols]
    lines = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "|".join("---:" if n else "---" for n in numeric) + "|"]
    for r in rows:
        cells = [f"`{r[c]}`" if c == "pipeline" and r.get(c) else r.get(c, "") for _, c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_readme() -> None:
    """Rewrite the README tables between ``<!-- results:<benchmark> -->`` and ``<!-- /results -->``."""
    readme = ROOT / "README.md"
    text = readme.read_text()
    for bench in TABLES:
        block = f"<!-- results:{bench} -->\n{markdown_table(bench, load(bench)[1])}\n<!-- /results -->"
        text = re.sub(rf"<!-- results:{bench} -->\n.*?<!-- /results -->", lambda _, block=block: block, text, flags=re.S)
    readme.write_text(text)
