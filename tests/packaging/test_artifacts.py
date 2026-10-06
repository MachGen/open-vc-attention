"""Inspect actual wheel and source distribution after python -m build."""

import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_and_sdist_inventory():
    wheels = list((ROOT / "dist").glob("open_vc_attn-*.whl"))
    sources = list((ROOT / "dist").glob("open_vc_attn-*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Build fresh artifacts first"
    with zipfile.ZipFile(wheels[0]) as z:
        names = z.namelist()
        expected_python = {str(p.relative_to(ROOT / "src")) for p in (ROOT / "src").rglob("*.py")}
        assert {n for n in names if n.endswith(".py")} == expected_python
        assert "open_vc_attn/api.py" in names
        assert "open_vc_attn/baselines.py" in names
        assert not any(n.startswith("vc_attn/") for n in names)
        kernel_dirs = {
            n.split("/")[2]
            for n in names
            if n.startswith("open_vc_attn/_kernels/") and len(n.split("/")) > 3
        }
        assert kernel_dirs == {"blackwell"}
        for name in expected_python:
            assert z.read(name) == (ROOT / "src" / name).read_bytes()
    with tarfile.open(sources[0]) as archive:
        names = [n.split("/", 1)[-1] for n in archive.getnames()]
        assert "integrations/sglang/install.py" in names
        assert "docs/technical-report/report.en.md" in names
        assert "tools/records.py" in names
        assert "benchmarks/results/b200/comparison.json" in names
