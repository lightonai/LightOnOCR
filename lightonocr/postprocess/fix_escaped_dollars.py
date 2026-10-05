#!/usr/bin/env python3
"""Conservative escaped-math recovery for OCR Markdown; standard library only.

Uses conservative guards; some ambiguous cases are still misclassified. This is not a complete Markdown/TeX parser and
cannot distinguish all literal-dollar prose from malformed math without context.
Only delimiter backslashes are removed; expression bodies are never rewritten.
"""
import re
from collections import Counter

ATOM = re.compile(r"\\(?:[A-Za-z]+|[^\r\n])|(?:\d+(?:\.\d*)?|\.\d+)|[^\W\d_]\w*|[-−–—+*/=<>:;,.'!|_^%&()\[\]{}]", re.UNICODE)


def math_body(body):
    """Conservative lexical evidence, not proof of mathematical correctness.

    No vocabulary of document/model-specific symbols. Alphabetic runs are
    identifiers; adjacent prose words are not accepted as implicit products.
    Command arguments can contain text (e.g. \\text{a long description}).
    """
    body = body.strip()
    if not body or '<td' in body.lower() or '</' in body:
        return False
    depth = 0
    for token in re.findall(r'\\.|[{}]', body):
        if token == '{': depth += 1
        elif token == '}': depth -= 1
        if depth < 0: return False
    if depth: return False
    # Mask balanced command arguments for lexical checks; preserve the source.
    clean = []
    pos = 0
    while pos < len(body):
        m = re.match(r'\\[A-Za-z]+', body[pos:])
        if m:
            clean.append(m[0]); pos += len(m[0])
            while pos < len(body) and body[pos] == '{':
                depth = 1; pos += 1
                while pos < len(body) and depth:
                    if body[pos] == '\\': pos += 2; continue
                    depth += (body[pos] == '{') - (body[pos] == '}'); pos += 1
                if depth: return False
                clean.append('{}')
        else:
            clean.append(body[pos]); pos += 1
    clean = ''.join(clean)
    previous = None; pos = 0
    for m in ATOM.finditer(clean):
        if clean[pos:m.start()].strip(): return False
        word = bool(re.fullmatch(r'[^\W\d_]\w*', m[0], re.UNICODE))
        if word and previous and (len(m[0]) > 1 or len(previous) > 1): return False
        previous = m[0] if word else None; pos = m.end()
    return not clean[pos:].strip()


def plausible_math(body):
    stripped=body.strip()
    if not math_body(body) or re.fullmatch(r'[^\W\d_]{2,}',stripped):
        return False
    if body != stripped and not re.search(r'[\w\\]', stripped):
        return False
    if len(stripped)>1 and stripped[0] in ',;':
        return False
    # Ignore escaped delimiters and command text arguments for bracket balance.
    # Mixed interval endpoints [a,b) are allowed; missing endpoints abstain.
    simple=re.sub(r"\\text\{[^{}]*\}", "", stripped)
    level=0
    for token in re.findall(r'\\.|[()\[\]]',simple):
        if token in ('(', '['):level+=1
        elif token in (')', ']'):level-=1
        if level<0:return False
    return level==0


# Protect code, document syntax and already-delimited display math.
PROTECTED = re.compile(
    r'^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=fence)[ \t]*(?:\n|$)|\Z)'
    r'|(?P<ticks>`+)[^`\n]*(?P=ticks)'
    r'|^(?: {4}|\t)[^\n]*(?:\n|$)'
    r'|<!--.*?(?:-->|\Z)'
    r'|<(?:pre|code|script|style)\b[^>]*>.*?(?:</(?:pre|code|script|style)\s*>|\Z)'
    r'|<[/!?A-Za-z](?:"[^"]*"|\x27[^\x27]*\x27|[^\x27">])*>'
    r'|(?:https?://|mailto:|www\.)[^\s<>]+'
    r'|^[ \t]{0,3}\[[^\]\n]+\]:[^\n]*'
    r'|(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]|\\\(.*?\\\)',
    re.M | re.S | re.I,
)
TOKEN = re.compile(r'(?<![\\$])\\?\$(?!\$)')
INLINE = re.compile(r'(?<![\\$])\$(?!\$)((?:\\.|[^$\\\n])*)(?<!\\)\$(?!\$)')
NUMERIC_START = re.compile(r'[+\-−]?\s*(?:\d|[.,]\d)')


def _ranges(text):
    spans = [(m.start(),m.end()) for m in PROTECTED.finditer(text)]
    # Protect balanced inline-link destinations, including relative URLs.
    for m in re.finditer(r'\]\(', text):
        pos=m.end();depth=1
        while pos<len(text) and text[pos]!='\n' and depth:
            if text[pos]=='\\':pos+=2;continue
            if text[pos]=='(':depth+=1
            elif text[pos]==')':depth-=1
            pos+=1
        if depth==0:spans.append((m.start(),pos))
    # A literal dollar inside a complete valid-looking math span is not a broken
    # closing delimiter. Protect the complete span before attempting recovery.
    for m in INLINE.finditer(text):
        if plausible_math(m[1]):spans.append((m.start(),m.end()))
    merged=[]
    for start,end in sorted(spans):
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return merged


def repair_dollar_delimiters(text, *, audit=None):
    """Return repaired text and counters; optional audit collects exact edits.

    Paired escaped numeric math is eligible; numeric opening-only spans abstain
    because they are indistinguishable locally from prices before math footnotes.
    Bare multi-letter words abstain. No scorer, model or reference access.
    """
    counts=Counter();edits=[]
    def scan(start,end):
        part=text[start:end];tokens=list(TOKEN.finditer(part));i=0
        while i+1<len(tokens):
            left,right=tokens[i:i+2];body=part[left.end():right.start()]
            el=left[0].startswith('\\');er=right[0].startswith('\\')
            if not el and not er:
                i += 2 if plausible_math(body) else 1
                continue
            numeric=bool(NUMERIC_START.match(body.strip()))
            after=part[right.end():]
            before=part[:left.start()]
            env=re.fullmatch(r'\s*\\begin\{([A-Za-z*]+)\}.*\\end\{\1\}\s*',body,re.S)
            # Currency code glued to a dollar, e.g. US$, S$, HK$.
            code_currency=el and bool(re.search(r'(?<!\w)[A-Z]{1,3}$',before))
            ambiguous=(
                ('\n' in body or '\r' in body) and not env
                or bool(re.search(r'\n[ \t]*\n',body))
                or code_currency
                or (el and not er and numeric)
                or (numeric and bool(NUMERIC_START.match(after.lstrip(' \t'))))
                or bool(re.fullmatch(r'[^\W\d_]{2,}',body.strip()))
                or not plausible_math(body)
            )
            if ambiguous:
                counts['skipped_ambiguous_pairs']+=1;i+=1;continue
            a=start+left.start();b=start+right.end();replacement='$'+body+'$'
            edits.append((a,b,replacement))
            counts['repaired_pairs']+=1;counts['removed_backslashes']+=el+er
            if audit is not None:audit.append(dict(start=a,end=b,before=text[a:b],after=replacement,context=text[max(0,a-90):b+90]))
            i+=2
    previous=0
    for a,b in _ranges(text):scan(previous,a);previous=b
    scan(previous,len(text))
    for a,b,replacement in reversed(edits):text=text[:a]+replacement+text[b:]
    return text,counts


def fix_escaped_dollars(text: str) -> str:
    """Convert one OCR Markdown string; no other formatting transformations."""
    return repair_dollar_delimiters(text)[0]


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='One Markdown file or a directory of .md files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Separate output file or directory; originals are preserved')
    args = parser.parse_args()
    source, target = args.input.resolve(), args.output.resolve()
    if not source.exists():
        parser.error('Input does not exist')
    if source == target or source in target.parents or target in source.parents:
        parser.error('Input and output paths must be separate and non-overlapping')
    is_directory = source.is_dir()
    if target.exists() and target.is_dir() != is_directory:
        parser.error('Output must be a directory for directory input, or a file for file input')
    files = sorted(source.rglob('*.md')) if is_directory else [source]
    if not files:
        parser.error('No Markdown files found')
    # Resolve destinations before writing: reject symlinks pointing into input.
    destinations = [(target / p.relative_to(source) if is_directory else target) for p in files]
    for destination in destinations:
        resolved = destination.resolve()
        if resolved == source or (is_directory and source in resolved.parents):
            parser.error('An output destination resolves into the input tree')
    counts = Counter()
    for path, destination in zip(files, destinations):
        with path.open(encoding='utf-8', newline='') as stream:
            original = stream.read()
        result, edits = repair_dollar_delimiters(original)
        counts.update(edits)
        counts['files'] += 1
        counts['changed_files'] += result != original
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w', encoding='utf-8', newline='') as stream:
            stream.write(result)
    print(', '.join(f'{key}={value}' for key, value in sorted(counts.items())))


if __name__ == '__main__':
    main()
