#!/usr/bin/env python3
"""Apply the evaluated PP1 OCR Markdown transformations (standard library only).

PP1 is the default, unchanged formatting pipeline for every endpoint.
PP2 repairs escaped-dollar delimiters first, then runs the full PP1 pipeline.
PP3 uses the shared safety dollar repair first, protects math lists, and preserves
trailing dots and form blanks.
No profile changes the scorer.
Input must already have the baseline grounding/margin extraction applied when
reproducing benchmark scores; this script preserves grounding markers.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


PROSE_DOLLARS = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")
# Protect code and existing math from prose transformations. Unclosed fences protect
# the remainder of the document as well.
OPAQUE = re.compile(
    r"^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<ticks>`+)[^`\n]*(?P=ticks)"
    r"|(?<!\\)\$\$.*?(?<!\\)\$\$"
    r"|\\\[.*?\\\]|\\\(.*?\\\)"
    r"|(?<![\\$])\$(?!\$)(?:\\.|[^$\n\\])*?(?<!\\)\$(?!\$)",
    re.MULTILINE | re.DOTALL,
)


def normalize_table_math(text: str) -> tuple[str, Counter[str]]:
    """Render only simple inline math as text inside HTML/Markdown cells.

    No fractions, scripts, arbitrary commands, display equations or currency
    amounts are rewritten. This preserves mathematical content and table shape.
    """
    counts: Counter[str] = Counter()
    symbols = {"diamond": "◊", "times": "×", "pm": "±", "leq": "≤",
               "geq": "≥", "neq": "≠", "alpha": "α", "beta": "β",
               "gamma": "γ", "mu": "μ", "sigma": "σ", "Delta": "Δ"}
    inline = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)|\\\(([^\n]+?)\\\)")

    def cell(content: str) -> str:
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            inner = match.group(1) if match.group(1) is not None else match.group(2)
            rendered = re.sub(r"\\([A-Za-z]+)", lambda m: symbols.get(m[1], m[0]), inner).strip()
            if (not re.fullmatch(r"[A-Za-z0-9\s.,()+\-=/<>αβγμσΔ◊×±≤≥≠]+", rendered)
                    or not re.search(r"[A-Za-zαβγμσΔ◊×±≤≥≠]", rendered)
                    or any(len(word) > 1 for word in re.findall(r"[A-Za-z]+", rendered))):
                return match.group(0)
            changed = True
            counts["normalized_table_math_spans"] += 1
            return rendered

        result = inline.sub(replace, content)
        if changed:
            result = re.sub(r"\(\s+", "(", result)
            result = re.sub(r"\s+\)", ")", result)
        return result

    def transform(part: str) -> str:
        part = re.sub(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)",
                      lambda m: m[1] + cell(m[2]) + m[3], part, flags=re.I | re.S)
        lines = part.splitlines(keepends=True)
        separator = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*")
        index = 1
        while index < len(lines):
            if separator.fullmatch(lines[index]) and "|" in lines[index - 1]:
                lines[index - 1] = cell(lines[index - 1])
                index += 1
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    lines[index] = cell(lines[index])
                    index += 1
            else:
                index += 1
        return "".join(lines)

    # Apply to text outside code; existing math is handled by the cell parser.
    code = re.compile(r"^[ \t]*(?P<f>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=f)[ \t]*(?:\n|$)|\Z)|(?P<t>`+)[^`\n]*(?P=t)", re.M | re.S)
    parts, start = [], 0
    for match in code.finditer(text):
        parts.extend((transform(text[start:match.start()]), match[0]))
        start = match.end()
    parts.append(transform(text[start:]))
    return "".join(parts), counts


CODE = re.compile(
    r"^[ \t]*(?P<f>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=f)[ \t]*(?:\n|$)|\Z)"
    r"|(?P<t>`+)[^`\n]*(?P=t)", re.M | re.S)


def outside_matches(text, protected, transform):
    parts, start = [], 0
    for match in protected.finditer(text):
        parts.extend((transform(text[start:match.start()]), match[0]))
        start = match.end()
    parts.append(transform(text[start:]))
    return "".join(parts)


def transform_cells(text, cell):
    """Visit HTML cells and cells in Markdown tables with separator rows."""
    def transform(part):
        part = re.sub(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]\s*>)",
                      lambda m: m[1] + cell(m[2]) + m[3], part, flags=re.I | re.S)
        lines = part.splitlines(keepends=True)
        separator = re.compile(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*")
        def row(line):
            # Escaped pipes and math containing pipes are ambiguous: preserve them.
            if "\\|" in line or "$" in line:
                return line
            return "|".join(cell(value) for value in line.split("|"))
        index = 1
        while index < len(lines):
            if separator.fullmatch(lines[index]) and "|" in lines[index - 1]:
                lines[index - 1] = row(lines[index - 1])
                index += 1
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    lines[index] = row(lines[index])
                    index += 1
            else:
                index += 1
        return "".join(lines)
    return outside_matches(text, CODE, transform)


def normalize_table_spacing(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    def cell(value):
        body = value.strip()
        if re.fullmatch(r"\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?", body):
            new = re.sub(r"\s*:\s*", ":", body)
        elif re.fullmatch(r"[+-]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:\.\d+)?", body):
            new = re.sub(r"[ \u00a0\u202f]", "", body)
        elif re.fullmatch(r"[μµ]\s+(?:m|g|l|L|s|A|V|W|F|M)", body):
            new = re.sub(r"\s+", "", body)
        else:
            return value
        if new != body:
            counts["normalized_table_spacing_cells"] += 1
        return value.replace(body, new, 1)
    return transform_cells(text, cell), counts


def remove_table_leaders(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    def cell(value):
        if not re.search(r"[^\W\d_]", value) or re.search(r"[<>$\\]", value):
            return value
        new, number = re.subn(r"[ \t]*(?:\.{3,}|…{2,})[ \t]*(?=\s*$)", "", value)
        counts["removed_table_leaders"] += number
        return new
    return transform_cells(text, cell), counts


def shorten_form_blanks(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    def transform(part):
        # Only standalone blanks on lines containing a field label.
        def line(value):
            if not re.search(r"[^\W\d_]", value) or value.lstrip().startswith(("<", "|")):
                return value
            new, number = re.subn(r"(?<!\S)_{10,}(?!\S)", "________", value)
            counts["shortened_form_blanks"] += number
            return new
        return "".join(line(value) for value in part.splitlines(keepends=True))
    return outside_matches(text, OPAQUE, transform), counts


def join_adjacent_math(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    inline = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")
    def transform(part):
        matches = list(inline.finditer(part))
        if not matches:
            return part
        parts, start, index = [], 0, 0
        def eligible(body):
            # Require an equation/relation in each span; currency and prose
            # cannot join. Math itself is copied verbatim.
            return bool(re.search(r"[=<>]|\\(?:leq|geq|neq)\b", body))
        while index < len(matches):
            first = last = matches[index]
            body = first[1]
            while index + 1 < len(matches) and eligible(last[1]):
                following = matches[index + 1]
                gap = part[last.end():following.start()]
                if not re.fullmatch(r"[ \t]*[,;][ \t]*", gap) or not eligible(following[1]):
                    break
                body += gap + following[1]
                last = following
                index += 1
                counts["joined_adjacent_math_spans"] += 1
            parts.extend((part[start:first.start()], "$" + body + "$"))
            start = last.end()
            index += 1
        parts.append(part[start:])
        return "".join(parts)
    protected = re.compile(CODE.pattern + r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)", re.M | re.S)
    return outside_matches(text, protected, transform), counts


def normalize_prose_math(text: str) -> tuple[str, Counter[str]]:
    """Render simple inline notation on prose lines; preserve actual formulas."""
    counts: Counter[str] = Counter()
    symbols = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
               "Delta": "Δ", "mu": "μ", "sigma": "σ", "theta": "θ",
               "pi": "π", "omega": "ω", "tanh": "tanh"}
    protected = re.compile(CODE.pattern + r"|<table\b.*?</table>"
                           r"|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]", re.M | re.S | re.I)
    def transform(part):
        lines = []
        for line in part.splitlines(keepends=True):
            # Mixed dollar errors are deliberately preserved, as requested.
            prose = PROSE_DOLLARS.sub("", line)
            if ("\\$" in line or "|" in line or line.lstrip().startswith(("<", "!["))
                    or len(re.findall(r"\b[A-Za-z]{2,}\b", prose)) < 3):
                lines.append(line)
                continue
            def replace(match):
                body = match[1].strip()
                # A prose sentence can contain a complete equation. Retain its
                # delimiters so formula consumers can still extract it.
                if any(char in body for char in "=<>"):
                    return match[0]
                rendered = re.sub(r"\\([A-Za-z]+)", lambda m: symbols.get(m[1], m[0]), body)
                if (not re.fullmatch(r"[A-Za-zαβγδΔμσθπω0-9 .,+/()=<>-]+", rendered)
                        or not re.search(r"[A-Za-zαβγδΔμσθπω]", rendered)
                        or any(len(w) > 1 and w != "tanh" for w in re.findall(r"[A-Za-z]+", rendered))):
                    return match[0]
                counts["normalized_prose_math_spans"] += 1
                return rendered
            lines.append(PROSE_DOLLARS.sub(replace, line))
        return "".join(lines)
    return outside_matches(text, protected, transform), counts


def normalize_prose_math_lists(text: str) -> tuple[str, Counter[str]]:
    """Keep linked math spans wrapped; normalize isolated prose notation as PP1."""
    counts = Counter()
    protected = re.compile(CODE.pattern + r'|<table\b.*?</table>|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]', re.M | re.S | re.I)
    def transform(part):
        lines = []
        for line in part.splitlines(keepends=True):
            spans = list(PROSE_DOLLARS.finditer(line))
            keep = set()
            for left, right in zip(spans, spans[1:]):
                gap = line[left.end():right.start()]
                if re.fullmatch(r'[ \t]*(?:[,;][ \t]*(?:(?:and|or)[ \t]+)?|(?:and|or)[ \t]+)', gap):
                    keep.update((left.start(), right.start()))
            # Use original normalization on the complete line to preserve its
            # eligibility checks, then reconstruct from each span's decision.
            eligible = normalize_prose_math(line)[0]
            if eligible == line or not keep:
                value, c = normalize_prose_math(line)
                counts.update(c); lines.append(value); continue
            prose = PROSE_DOLLARS.sub('', line)
            def replace(m):
                if m.start() in keep:
                    counts['preserved_linked_math_spans'] += 1
                    return m[0]
                # Original renderer with the same surrounding prose eligibility.
                prefix = ' '.join(prose.splitlines())
                probe = prefix + m[0]
                value, c = normalize_prose_math(probe)
                counts.update(c)
                return value[len(prefix):]
            lines.append(PROSE_DOLLARS.sub(replace, line))
        return ''.join(lines)
    return outside_matches(text, protected, transform), counts


def repair_latex_syntax(text: str) -> tuple[str, Counter[str]]:
    """Expand known operators, escape text ampersands, reconnect split fences.

    Never change dollar escaping or invent missing delimiters/content.
    """
    counts: Counter[str] = Counter()
    math = re.compile(r"(?<!\\)\$\$(.*?)\$\$|\\\[(.*?)\\\]"
                      r"|(?<![\\$])\$(?!\$)([^$\n]*?)(?<!\\)\$(?!\$)|\\\((.*?)\\\)", re.S)
    display = re.compile(r"(?<!\\)\$\$(.*?)\$\$", re.S)
    def transform(part):
        def repair(match):
            original = match[0]
            # Leave every expression containing escaped dollars untouched.
            if "\\$" in original:
                return original
            new, n = re.subn(r"\\esssup(?![A-Za-z])", lambda m: r"\operatorname*{ess\,sup}", original)
            counts["expanded_esssup"] += n
            def textext(m):
                body, n = re.subn(r"(?<!\\)&", r"\\&", m[1])
                counts["escaped_text_ampersands"] += n
                return r"\text{" + body + "}"
            return re.sub(r"\\text\{([^{}]*)\}", textext, new)
        part = math.sub(repair, part)
        matches = list(display.finditer(part))
        parts, start, i = [], 0, 0
        def balance(body):
            level = 0
            for token in re.findall(r"\\(left|right)\b", body):
                level += 1 if token == "left" else -1
                if level < 0:
                    return -1
            return level
        while i < len(matches):
            first = matches[i]
            body, end, j = first[1], first.end(), i
            # Join only adjacent display blocks until an existing \left is
            # closed by an existing \right, without intervening prose.
            while balance(body) > 0 and j + 1 < len(matches):
                following = matches[j + 1]
                if part[end:following.start()].strip():
                    break
                body += "\n" + following[1]
                end, j = following.end(), j + 1
                if balance(body) == 0:
                    break
            if j > i and balance(body) == 0 and "\\$" not in body:
                parts.extend((part[start:first.start()], "$$" + body + "$$"))
                counts["joined_split_display_blocks"] += j - i
                start, i = end, j + 1
            else:
                i += 1
        parts.append(part[start:])
        return "".join(parts)
    return outside_matches(text, CODE, transform), counts


TRANSFORMS = {"table-math": normalize_table_math,
              "table-spacing": normalize_table_spacing,
              "table-leaders": remove_table_leaders,
              "form-blanks": shorten_form_blanks,
              "adjacent-math": join_adjacent_math,
              "prose-math": normalize_prose_math,
              "latex-syntax": repair_latex_syntax}


def merge_inline_math(text):
    counts = Counter()
    marker = '$'
    pattern = r'(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)'
    formula = re.compile(pattern, re.S)
    protected = CODE.pattern + r'|^.*\\\$.*$'
    protected += r'|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)'
    # Avoid DOTALL on the escaped-dollar line alternative.
    protected = re.compile(protected.replace(r'^.*\\\$.*$', r'^[^\n]*\\\$[^\n]*$'), re.M | re.S)
    def transform(part):
        matches = list(formula.finditer(part))
        pieces, start, i = [], 0, 0
        def eligible(m):
            return not re.fullmatch(r'-?\d{1,3}(?:,\d{3})*', m[1].strip())
        while i < len(matches):
            first = last = matches[i]
            body = first[1]
            while eligible(last) and i+1 < len(matches) and eligible(matches[i+1]):
                following = matches[i+1]
                gap = part[last.end():following.start()]
                if not re.fullmatch(r'(?:\s|,|and)*',gap,re.I):
                    break
                body += (gap if gap.strip() else ' ') + following[1]
                last = following
                counts['merged_inline_pairs'] += 1
                i += 1
            pieces.extend([part[start:first.start()],marker+body+marker])
            start = last.end()
            i += 1
        pieces.append(part[start:])
        return ''.join(pieces)
    return outside_matches(text,protected,transform), counts


TRANSFORMS['infinity-inline'] = merge_inline_math

PP1_STEPS = ("table-math", "table-spacing", "table-leaders", "form-blanks",
             "adjacent-math", "latex-syntax", "prose-math", "infinity-inline")


PP3_STEPS = tuple(name for name in PP1_STEPS
                  if name not in ("table-leaders", "form-blanks"))


DOLLAR_PROTECTED = re.compile(
    r'^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)'
    r'|(?P<ticks>`+)[^`\n]*(?P=ticks)'
    r'|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)', re.M|re.S)
DOLLAR_PAIR = re.compile(r'(?<![\\$])(?P<left>\\?\$)(?!\$)(?P<body>(?:\\(?!\$)[^\r\n]|[^$\\\r\n])+?)(?P<right>\\?\$)(?!\$)')
def repair_dollar_delimiters(text):
    """Recover paired/one-sided escaped inline math; preserve ambiguous currency/code."""
    counts=Counter()
    def part(s):
        def replace(m):
            if '\\' not in m['left']+m['right']:return m[0]
            body=m['body'];words=re.findall(r'[A-Za-z]+',re.sub(r'\\[A-Za-z]+','',body))
            evidence=re.search(r'[A-Za-z_^=+*/()\[\]{}<>]|\\[A-Za-z]+',body)
            if (not evidence or any(len(w)>1 for w in words) or re.search(r'[\u4e00-\u9fff`"]',body)
                or (m.end()<len(s) and s[m.end()].isdigit())):
                counts['skipped_ambiguous_pairs']+=1;return m[0]
            counts['repaired_pairs']+=1
            counts['removed_backslashes']+=m['left'].count('\\')+m['right'].count('\\')
            counts['paired' if m['left'].startswith('\\') and m['right'].startswith('\\') else 'one_sided']+=1
            return '$'+body+'$'
        return DOLLAR_PAIR.sub(replace,s)
    pieces=[];start=0
    for m in DOLLAR_PROTECTED.finditer(text):pieces.extend([part(text[start:m.start()]),m[0]]);start=m.end()
    pieces.append(part(text[start:]));return ''.join(pieces),counts


def cleanup(text: str, profile: str = "pp1") -> tuple[str, Counter[str]]:
    if profile not in ("pp1", "pp2", "pp3"):
        raise ValueError(f"Unknown output postprocessing profile: {profile}")
    counts: Counter[str] = Counter()
    if profile in ("pp2", "pp3"):
        repair = repair_dollar_delimiters
        if profile == "pp3":
            if __package__:
                from .fix_escaped_dollars import repair_dollar_delimiters as repair
            else:
                from fix_escaped_dollars import repair_dollar_delimiters as repair
        text, edits = repair(text)
        counts.update(edits)
    for name in (PP3_STEPS if profile == "pp3" else PP1_STEPS):
        transform = (normalize_prose_math_lists
                     if profile == "pp3" and name == "prose-math"
                     else TRANSFORMS[name])
        text, edits = transform(text)
        counts.update(edits)
    return text, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Raw plain or grounding Markdown tree")
    parser.add_argument("--output", type=Path, required=True, help="Separate output tree")
    parser.add_argument("--profile", choices=("pp1", "pp2", "pp3"), default="pp1",
                        help="pp1: formatting only (default); pp2: dollar-delimiter recovery then pp1; pp3: safety dollar recovery then formatting with math-list protection, preserving dots and form blanks")
    args = parser.parse_args()
    source = args.input.resolve()
    target = args.output.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if source == target or source in target.parents or target in source.parents:
        parser.error("Input and output trees must be separate")
    files = sorted(source.rglob("*.md"))
    if not files:
        parser.error(f"No Markdown files in {source}")
    totals: Counter[str] = Counter()
    for path in files:
        original = path.read_text(encoding="utf-8")
        result, counts = cleanup(original, profile=args.profile)
        totals.update(counts)
        totals["files"] += 1
        totals["changed_files"] += result != original
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result, encoding="utf-8")
    print(f"{args.profile}: {source} -> {target}")
    print(", ".join(f"{key}={value}" for key, value in sorted(totals.items())))


if __name__ == "__main__":
    main()
