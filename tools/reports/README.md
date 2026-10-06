# Report build

`build.py` builds the [technical report](../../docs/technical-report/README.md) from its Markdown source:

1. `make_figs.py` regenerates the tables, charts and computed numbers from `benchmarks/results/b200/*.json`.
2. `md2tex.py` converts `report.en.md` to LaTeX using `template.tex`, filling `{{name}}` numbers.
3. The LaTeX is compiled with `tectonic`, or with `pdflatex` and `bibtex` when tectonic is not installed. The build fails on undefined references or citations and on unresolved `??` markers.
4. The PDF is written to `docs/technical-report/pdf/` and the source, record and PDF hashes to `docs/technical-report/manifest.json`.

Generated files live in a temporary directory; `--keep DIR` keeps it for debugging.

```bash
python -m pip install -e '.[report]'
python tools/reports/build.py                  # build the PDF and manifest
python tools/reports/build.py --render pages/  # also render every page to PNG
```

Rebuild after changing the report source or the benchmark records, and inspect the rendered pages before publishing. `tools/audit_release.py` fails when the manifest no longer matches the files.
