"""Audit release contents: package layout, private references and documentation links."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "build", "dist", ".pytest_cache", ".ruff_cache", "__pycache__"}
# Git-ignored local output directories (see .gitignore).
LOCAL = {"results", ".venv"}
PRIVATE = re.compile(r"/Users/|/home/[a-z]|/mnt/disk|/workspace/")


def _check_records(directory):
    import importlib.util

    spec = importlib.util.spec_from_file_location("records", ROOT / "tools/reports/records.py")
    records = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(records)
    return records.check(directory)


def audit(root=ROOT):
    errors = []
    # Build artifacts (editable-install metadata, bytecode caches) are not packages.
    packages = sorted(
        p.name
        for p in (root / "src").iterdir()
        if p.is_dir() and not p.name.endswith(".egg-info") and p.name != "__pycache__"
    )
    if packages != ["open_vc_attn"]:
        errors.append(f"Unexpected top-level packages: {packages}")
    kernels = sorted(
        p.name
        for p in (root / "src/open_vc_attn/_kernels").iterdir()
        if p.is_dir() and p.name != "__pycache__"
    )
    if kernels != ["blackwell"]:
        errors.append(f"Unexpected kernel trees: {kernels}")
    report = json.loads((root / "docs/technical-report/manifest.json").read_text())
    for name, digest in report["files"].items():
        if (
            not (root / name).exists()
            or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
        ):
            errors.append("Report manifest mismatch (rebuild with tools/reports/build.py): " + name)
    errors += ["Benchmark record: " + e for e in _check_records(root / "benchmarks/results/b200")]
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        ignored = rel.parts[0] in LOCAL or any(
            x in SKIP or x.endswith(".egg-info") for x in rel.parts
        )
        if ignored or not path.is_file():
            continue
        if path.suffix not in {
            ".py",
            ".md",
            ".json",
            ".toml",
            ".yml",
            ".patch",
            ".tex",
            ".tikz",
            ".cff",
        }:
            continue
        text = path.read_text()
        if path != Path(__file__).resolve() and PRIVATE.search(text):
            errors.append("Private environment reference: " + str(rel))
        if path.suffix == ".md":
            prose = re.sub(r"`[^`]*`", "", text)
            # Images are skipped: report figures are generated at build time.
            for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", prose):
                target = target.split("#")[0]
                if (
                    target
                    and "://" not in target
                    and not target.startswith("mailto:")
                    and not (path.parent / target).exists()
                ):
                    errors.append(f"Broken link: {rel}: {target}")
    if errors:
        raise ValueError("\n".join(errors))
    return {"status": "passed"}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
