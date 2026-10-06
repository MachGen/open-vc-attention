"""Build the technical report PDF from its checked-in Markdown source.

report.en.md is the only hand-edited source. In a temporary directory this tool
runs make_figs.py (tables, charts and computed numbers from the benchmark records),
converts the Markdown to LaTeX with md2tex.py and template.tex, compiles it, checks
for undefined references and unresolved markers, then writes the PDF and a
source/PDF hash manifest into docs/technical-report/.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / "docs/technical-report"
RECORDS = ROOT / "benchmarks/results/b200"
SOURCES = ("report.en.md", "refs.bib", "figures/pipeline.tikz", "figures/traversal.tikz")
# Benchmark records the report reads (via make_figs.py).
REPORT_RECORDS = ("comparison.json", "repair.json")
PROBLEMS = re.compile(
    r"Reference `[^']+' on page \d+ undefined|Citation `[^']+' on page \d+ undefined"
    r"|There were undefined (references|citations)",
)

sys.path.insert(0, str(HERE))
import md2tex  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit(f"Command failed: {' '.join(map(str, command))}")
    return result.stdout + result.stderr


def compile_pdf(workdir):
    if shutil.which("tectonic"):
        log = run(["tectonic", "-X", "compile", "--keep-logs", "main.tex"], workdir)
    elif shutil.which("pdflatex") and shutil.which("bibtex"):
        flags = ["-interaction=nonstopmode", "-halt-on-error"]
        log = run(["pdflatex", *flags, "main.tex"], workdir)
        log += run(["bibtex", "main"], workdir)
        log += run(["pdflatex", *flags, "main.tex"], workdir)
        log += run(["pdflatex", *flags, "main.tex"], workdir)
    else:
        raise SystemExit("Install tectonic, or pdflatex and bibtex, to build the report")
    log += (workdir / "main.log").read_text(errors="replace")
    return workdir / "main.pdf", log


def check_pdf(pdf, render_dir):
    try:
        import pypdfium2
    except ImportError:
        print("pypdfium2 not installed: skipping the PDF text check and page rendering")
        return None
    document = pypdfium2.PdfDocument(str(pdf))
    for index in range(len(document)):
        if "??" in document[index].get_textpage().get_text_range():
            raise SystemExit(f"Unresolved reference marker '??' on page {index + 1}")
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            page = document[index].render(scale=1.5).to_pil()
            page.save(render_dir / f"page-{index + 1:02d}.png")
    return len(document)


def generate(work):
    """Copy the sources and generate every table, chart and number, then the LaTeX source.

    Fails on records missing a field the report reads and on unknown {{name}} references.
    """
    for name in SOURCES:
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPORT / name, work / name)
    print("Generating tables, charts and numbers")
    run([sys.executable, str(HERE / "make_figs.py")], work)
    print("Converting Markdown to LaTeX")
    md2tex.render(work / "report.en.md", work / "tables/numbers.json", work / "main.tex")


def tables_only():
    with tempfile.TemporaryDirectory() as tmp:
        generate(Path(tmp))
    print("Report tables and LaTeX source generated")


def build(render_dir=None, keep=None):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(keep) if keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        generate(work)
        print("Compiling")
        pdf, log = compile_pdf(work)
        problems = sorted({m.group(0) for m in PROBLEMS.finditer(log)})
        if problems:
            raise SystemExit("LaTeX reported: " + "; ".join(problems))
        pages = check_pdf(pdf, render_dir)
        target = REPORT / "pdf/Open-VC-Attn-Technical-Report-en.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, target)
    files = [REPORT / n for n in SOURCES] + [RECORDS / n for n in REPORT_RECORDS] + [target]
    manifest = {
        "schema": "open-vc-report-v1",
        "files": {str(p.relative_to(ROOT)): sha256(p) for p in files},
    }
    (REPORT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {target.relative_to(ROOT)}" + (f" ({pages} pages)" if pages else ""))
    print("Wrote docs/technical-report/manifest.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, metavar="DIR", help="Render every page to PNG")
    parser.add_argument("--keep", type=Path, metavar="DIR", help="Keep the build directory")
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Generate tables, numbers and LaTeX without compiling (no LaTeX needed; for CI)",
    )
    args = parser.parse_args(argv)
    try:
        if args.tables_only:
            tables_only()
            return
        build(args.render, args.keep)
    except md2tex.ConversionError as exc:
        raise SystemExit(f"report.en.md: {exc}")
    if args.render:
        print(f"Rendered pages to {args.render}; inspect every page before publishing")


if __name__ == "__main__":
    main()
