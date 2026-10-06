# Editable PDF sources

The English and Chinese report builders use only public repository data:
`docs/reports/captured-summary.json` and
`docs/reports/VC-Attention-Long-Sequence-Results.json`. Historical timings are
preserved. New reproduction instructions describe the current public release.

```bash
python -m pip install '.[reports]'
python tools/report_pdf/build_en.py
python tools/report_pdf/build_zh.py
```

The Chinese builder uses the PDF CJK font `STSong-Light` by default. For a
self-contained PDF with an embedded font, set `VC_REPORT_CJK_FONT` to a licensed
TrueType font supporting Chinese, then rerun `build_zh.py`. Font files are not
distributed. The published Chinese PDF uses an embedded Arial Unicode font.
Different ReportLab/font versions may change layout and file bytes.

Each builder requires 15 pages and writes extracted text/audit data beneath
`build/report-pdf/`. Render and visually inspect every page before release;
automated page-count/text checks cannot detect all clipping or missing glyphs.
After review, update the sizes and SHA256 values in `docs/reports/manifest.json`
and run `python tools/audit_release.py`. Do not update the manifest merely to
silence an unreviewed PDF mismatch.

[Published reports](../../docs/reports/README.md) · [Repository tools](../README.md)
