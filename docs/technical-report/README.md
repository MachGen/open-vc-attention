# Technical report

[`report.en.md`](report.en.md) is the source of the [technical report PDF](pdf/Open-VC-Attn-Technical-Report-en.pdf); `refs.bib` holds the references and `figures/*.tikz` the two diagrams. Every computed number, table and chart is generated at build time from the benchmark records in [`benchmarks/results/b200/`](../../benchmarks/results/b200/); nothing generated is committed. `manifest.json` records the source, record and PDF hashes.

Build with [tools/reports](../../tools/reports/README.md):

```bash
python -m pip install -e '.[report]'
python tools/reports/build.py --render results/report-pages
```

## Markdown conventions

- Front matter between `---` lines gives the title, subtitle, author, revision, date, code URL, package and license.
- `::: abstract`, `::: keybox`, `::: table {#tab:id}` and `::: figure {#fig:id}` open blocks closed by `:::`. A table or figure block starts with its caption paragraph, followed by pipe tables, `![](fig/name.pdf){width=0.5}` images or `!include path` lines. Table attributes: `cols="lX"` for wrapping columns, `size=footnotesize`, `place=h`.
- Headings take labels: `## Accuracy {#sec:eval:acc}`; `{.appendix}` on the first appendix heading. `####` is a run-in paragraph heading.
- Math is `$...$` inline and `$$ ... $$ {#eq:id}` for numbered equations.
- `@sec:id`, `@tab:id`, `@fig:id`, `@eq:id`, `@app:id` are cross-references (capitalize, e.g. `@Tab:id`, at the start of a sentence); `[@sec:a; @sec:b]` groups them. `[@key]` and `[@a; @b]` are citations from `refs.bib`.
- `{{name}}` inserts a number computed by `tools/reports/make_figs.py`; the build fails on unknown names.
