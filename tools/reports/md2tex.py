"""Convert the technical report's Markdown source into a LaTeX document.

The Markdown dialect (documented in tools/reports/README.md) covers what the report
needs: front matter, headings with labels, paragraphs, lists, math, pipe tables,
fenced blocks for tables, figures and boxes, code blocks, cross-references, citations
and {{name}} placeholders for computed numbers. No external converter is required.
"""

import json
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.tex")
REF_PREFIXES = ("sec", "fig", "tab", "eq", "app")
UNICODE = {
    "×": r"\ensuremath{\times}",
    "≈": r"\ensuremath{\approx}",
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "→": r"\ensuremath{\to}",
    "−": r"\ensuremath{-}",
    "±": r"\ensuremath{\pm}",
    "—": "---",
    "–": "--",
    "…": r"\ldots{}",
    "’": "'",
}


class ConversionError(ValueError):
    pass


# ---------------------------------------------------------------- inline
def _escape_text(text):
    text = re.sub(r'"([^"]+)"', lambda m: "\x00" + m.group(1) + "\x01", text)
    out = []
    for ch in text:
        if ch in "%&#_":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\x00":
            out.append("``")
        elif ch == "\x01":
            out.append("''")
        elif ch == '"':
            out.append("''")
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def _escape_code(text):
    table = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "ρ": r"\ensuremath{\rho}",
    }
    return "".join(table.get(ch, ch) for ch in text)


def _refs(keys):
    return [k for k in keys if k.split(":")[0].lower() in REF_PREFIXES]


INLINE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<math>\$[^$]+\$)"
    r"|(?P<num>\{\{[A-Za-z0-9_]+\}\})"
    r"|(?P<group>\s?\[@[^\]]+\])"
    r"|(?P<ref>@[A-Za-z]+:[A-Za-z0-9:_-]*[A-Za-z0-9_])"
    r"|(?P<url><https?://[^>]+>)"
    r"|(?P<bold>\*\*.+?\*\*)"
    r"|(?P<emph>\*[^*\s][^*]*\*)"
)


def inline(text, numbers):
    """Convert inline Markdown to LaTeX."""
    out, pos = [], 0
    for m in INLINE.finditer(text):
        out.append(_escape_text(text[pos : m.start()]))
        kind, token = m.lastgroup, m.group()
        if kind == "code":
            out.append(r"\code{" + _escape_code(token[1:-1]) + "}")
        elif kind == "math":
            out.append(_numbers(token, numbers))
        elif kind == "num":
            out.append(_number(token[2:-2], numbers))
        elif kind == "group":
            space = token[0].isspace()
            keys = [k.strip().lstrip("@") for k in token.strip()[1:-1].split(";")]
            refs = _refs(keys)
            if refs and len(refs) != len(keys):
                raise ConversionError(f"Mixed references and citations in {token!r}")
            if refs:
                cmd = r"\Cref" if keys[0][0].isupper() else r"\cref"
                keys = [k[0].lower() + k[1:] for k in keys]
                out.append((" " if space else "") + cmd + "{" + ",".join(keys) + "}")
            else:
                out.append("~\\cite{" + ",".join(keys) + "}")
        elif kind == "ref":
            key = token[1:]
            if key.split(":")[0].lower() not in REF_PREFIXES:
                out.append(_escape_text(token))
            else:
                cmd = r"\Cref" if key[0].isupper() else r"\cref"
                out.append(cmd + "{" + key[0].lower() + key[1:] + "}")
        elif kind == "url":
            out.append(r"\url{" + token[1:-1] + "}")
        elif kind == "bold":
            out.append(r"\textbf{" + inline(token[2:-2], numbers) + "}")
        elif kind == "emph":
            out.append(r"\emph{" + inline(token[1:-1], numbers) + "}")
        pos = m.end()
    out.append(_escape_text(text[pos:]))
    return "".join(out)


def _number(name, numbers):
    if name not in numbers:
        raise ConversionError(f"Unknown number placeholder {{{{{name}}}}}")
    return numbers[name]


def _numbers(text, numbers):
    return re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", lambda m: _number(m.group(1), numbers), text)


# ---------------------------------------------------------------- blocks
ATTR = re.compile(r"\{([^{}]*)\}\s*$")


def _attrs(text):
    """Parse a trailing {#id .class key=value key="v w"} attribute block."""
    m = ATTR.search(text)
    if not m or not re.match(r"^\s*[#.\w]", m.group(1)):
        return text, {}, None
    attrs, ident = {}, None
    for part in re.findall(r'[#.]?[\w:-]+(?:="[^"]*"|=\S+)?', m.group(1)):
        if part.startswith("#"):
            ident = part[1:]
        elif part.startswith("."):
            attrs.setdefault("classes", []).append(part[1:])
        elif "=" in part:
            key, value = part.split("=", 1)
            attrs[key] = value.strip('"')
    return text[: m.start()].rstrip(), attrs, ident


def _table_rows(lines):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    return rows


def pipe_table(lines, numbers, cols=None):
    if len(lines) < 2 or not re.match(r"^\s*\|?\s*:?-{3,}", lines[1]):
        raise ConversionError("A pipe table needs a header row and an alignment row")
    header, align, *body = _table_rows(lines)
    if cols is None:
        spec = []
        for a in align:
            spec.append(
                "c" if a.startswith(":") and a.endswith(":") else "r" if a.endswith(":") else "l"
            )
        cols = "".join(spec)
    tabularx = "X" in cols
    colspec = cols.replace(" ", "")
    if tabularx:
        colspec = colspec.replace("X", r">{\raggedright\arraybackslash}X")
    out = [
        (r"\begin{tabularx}{\textwidth}" if tabularx else r"\begin{tabular}")
        + "{@{}"
        + colspec
        + "@{}}",
        r"\toprule",
        " & ".join(inline(c, numbers) for c in header) + r"\\",
        r"\midrule",
    ]
    for row in body:
        if all(re.fullmatch(r"-{3,}", c) for c in row):
            out.append(r"\midrule")
            continue
        out.append(" & ".join(inline(c, numbers) for c in row) + r"\\")
    out += [r"\bottomrule", r"\end{tabularx}" if tabularx else r"\end{tabular}"]
    return "\n".join(out)


def _paragraphs(lines):
    """Split lines into blank-line separated chunks."""
    chunk, chunks = [], []
    for line in lines:
        if line.strip():
            chunk.append(line)
        elif chunk:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _include(path, base):
    target = base / path
    if not target.exists():
        raise ConversionError(f"Missing included file {path}")
    return r"\input{" + path + "}"


def float_block(kind, attrs, ident, lines, numbers, base):
    """A table (caption above) or figure (caption below) built from blank-line separated parts."""
    chunks = _paragraphs(lines)
    if len(chunks) < 2:
        raise ConversionError(f"A {kind} block needs a caption paragraph and content")
    caption = r"\caption{" + inline(" ".join(s.strip() for s in chunks[0]), numbers) + "}"
    label = [r"\label{" + ident + "}"] if ident else []
    content = []
    for i, chunk in enumerate(chunks[1:]):
        if i:
            content.append(r"\vspace{8pt}")
        first = chunk[0].strip()
        if first.startswith("!include "):
            content.append(_include(first.split(None, 1)[1], base))
        elif first.startswith("!["):
            m = re.match(r"!\[\]\(([^)]+)\)", first)
            if not m:
                raise ConversionError(f"Bad figure image line: {first}")
            _, image_attrs, _ = _attrs(first)
            width = image_attrs.get("width", "1")
            content.append(rf"\includegraphics[width={width}\textwidth]{{{m.group(1)}}}")
        elif first.startswith("|"):
            content.append(pipe_table(chunk, numbers, attrs.get("cols")))
        else:
            raise ConversionError(f"Unexpected content in {kind} block: {first}")
    env = "table" if kind == "table" else "figure"
    head = [rf"\begin{{{env}}}[{attrs.get('place', 't')}]", r"\centering"]
    if kind == "table":
        parts = head + [caption, *label, "\\" + attrs.get("size", "small"), *content]
    else:
        parts = head + [*content, caption, *label]
    return "\n".join(parts + [rf"\end{{{env}}}"])


HEADINGS = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}


def convert(markdown, numbers, base):
    """Return (metadata, abstract LaTeX, body LaTeX)."""
    lines = markdown.split("\n")
    meta = {}
    if lines and lines[0].strip() == "---":
        end = lines.index("---", 1)
        for line in lines[1:end]:
            if line.strip():
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
        lines = lines[end + 1 :]

    body, abstract = [], ""
    appendix_started = False
    i = 0
    paragraph = []

    def flush():
        if paragraph:
            body.append(inline(" ".join(s.strip() for s in paragraph), numbers))
            body.append("")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("<!--"):
            while "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            title, attrs, ident = _attrs(m.group(2))
            level = len(m.group(1))
            if "appendix" in attrs.get("classes", []) and not appendix_started:
                body += [
                    r"\bibliographystyle{unsrtnat}",
                    r"{\small\bibliography{refs}}",
                    "",
                    r"\appendix",
                ]
                appendix_started = True
            body.append("\\" + HEADINGS[level] + "{" + inline(title, numbers) + "}")
            if ident:
                body.append(r"\label{" + ident + "}")
            body.append("")
            i += 1
            continue
        if stripped.startswith(":::"):
            flush()
            header = stripped[3:].strip()
            kind = header.split()[0] if header else ""
            _, attrs, ident = _attrs(header)
            block = []
            i += 1
            while lines[i].strip() != ":::":
                block.append(lines[i])
                i += 1
            i += 1
            if kind in ("table", "figure"):
                body.append(float_block(kind, attrs, ident, block, numbers, base))
                body.append("")
            elif kind == "keybox":
                parts = [inline(" ".join(c), numbers) for c in _paragraphs(block)]
                body += [r"\begin{keybox}", "\n\n".join(parts), r"\end{keybox}", ""]
            elif kind == "abstract":
                abstract = "\n\n".join(inline(" ".join(c), numbers) for c in _paragraphs(block))
            else:
                raise ConversionError(f"Unknown block type {kind!r}")
            continue
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            code = []
            i += 1
            while not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            option = {"python": "[language=Python]", "bash": "[language=bash]"}.get(lang, "")
            body += [r"\begin{lstlisting}" + option, *code, r"\end{lstlisting}", ""]
            continue
        if stripped.startswith("$$"):
            flush()
            math = [stripped[2:]]
            i += 1
            while "$$" not in lines[i]:
                math.append(lines[i])
                i += 1
            closing = lines[i].strip()
            math.append(closing[: closing.index("$$")])
            _, attrs, ident = _attrs(closing[closing.index("$$") + 2 :])
            content = _numbers("\n".join(x for x in math if x.strip()), numbers)
            if ident:
                body += [
                    r"\begin{equation}",
                    content,
                    r"\label{" + ident + "}",
                    r"\end{equation}",
                    "",
                ]
            else:
                body += [r"\[", content, r"\]", ""]
            i += 1
            continue
        if re.match(r"^\s*(- |\d+\. )", line):
            flush()
            numbered = bool(re.match(r"^\s*\d+\. ", line))
            items = []
            while i < len(lines) and lines[i].strip():
                if re.match(r"^\s*(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^\s*(- |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            env = "enumerate" if numbered else "itemize"
            opt = "[label=\\arabic*.]" if numbered else ""
            body.append(rf"\begin{{{env}}}{opt}")
            body += [r"  \item " + inline(item, numbers) for item in items]
            body += [rf"\end{{{env}}}", ""]
            continue
        paragraph.append(line)
        i += 1
    flush()
    if not appendix_started:
        body += [r"\bibliographystyle{unsrtnat}", r"{\small\bibliography{refs}}"]
    return meta, abstract, "\n".join(body)


def render(markdown_path, numbers_path, output_path):
    markdown_path, output_path = Path(markdown_path), Path(output_path)
    numbers = json.loads(Path(numbers_path).read_text())
    meta, abstract, body = convert(markdown_path.read_text(), numbers, markdown_path.parent)
    required = ("title", "subtitle", "author", "revision", "date", "code", "package", "license")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ConversionError(f"Front matter is missing {missing}")
    document = TEMPLATE.read_text()
    fields = {k: inline(v, numbers) for k, v in meta.items()}
    fields.update(
        code=r"\url{" + meta["code"] + "}", package=r"\code{" + _escape_code(meta["package"]) + "}"
    )
    fields.update(abstract=abstract, body=body, pdftitle=meta["title"] + ": " + meta["subtitle"])
    for key, value in fields.items():
        document = document.replace("<<" + key + ">>", value)
    leftover = re.findall(r"<<\w+>>", document)
    if leftover:
        raise ConversionError(f"Template fields without values: {leftover}")
    output_path.write_text(document)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: md2tex.py report.md numbers.json main.tex")
    render(*sys.argv[1:])
