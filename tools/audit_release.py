"""Check source boundaries and frozen kernel hashes before distribution."""

import ast
import hashlib
import json
import os
import re
from pathlib import Path


def audit(root):
    failures = []
    # Public PDFs are reviewed separately, then pinned here by manifest hash.
    # All other binary files still fail the release boundary check below.
    report_names = {
        "VC-Attention-Fusedpipe-D-Technical-Report.pdf",
        "VC-Attention-Fusedpipe-D-Tech-Report-zh.pdf",
    }
    reports = json.loads((root / "docs/reports/manifest.json").read_text())["reports"]
    if set(reports) != report_names:
        failures.append("Unexpected public PDF inventory")
    for name in report_names:
        path = root / "docs/reports" / name
        record = reports.get(name, {})
        data = path.read_bytes() if path.is_file() else b""
        if (
            not data.startswith(b"%PDF-")
            or len(data) != record.get("bytes")
            or hashlib.sha256(data).hexdigest() != record.get("sha256")
        ):
            failures.append(f"Missing or changed reviewed PDF: {name}")
    if (root / ".github/README.md").exists():
        failures.append(".github/README.md overrides the project homepage; use DIRECTORY.md")
    if (root / "src/vc_attn/native/b200").exists():
        failures.append("Native v6 is B300-only; remove obsolete B200 native sources")
    excluded = {
        ".git",
        ".venv",
        "build",
        "dist",
        "results",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    forbidden_suffixes = {
        ".so",
        ".pt",
        ".pth",
        ".safetensors",
        ".mp4",
        ".zip",
        ".gz",
        ".tar",
        ".cubin",
    }
    private = re.compile(
        r"/mnt/" + r"disk\d|/Users/|/home/[^\s/]+/|38\.9\.57\.\d+|38\.127\.229\.\d+"
    )
    secrets = re.compile(
        r"gh[pousr]_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    )
    files = []
    for base, dirs, names in os.walk(root):
        for name in dirs + names:
            if (Path(base) / name).is_symlink():
                failures.append(f"Symlink: {Path(base, name).relative_to(root)}")
        dirs[:] = [x for x in dirs if x not in excluded and not x.endswith(".egg-info")]
        for name in names:
            if name == ".git":  # Worktrees store their Git metadata pointer in a file.
                continue
            path = Path(base) / name
            rel = path.relative_to(root)
            if path.suffix in forbidden_suffixes:
                failures.append(f"Disallowed artifact: {rel}")
            if path.suffix == ".pyc":
                continue
            if rel.parent == Path("docs/reports") and path.name in report_names:
                files.append(str(rel))
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                failures.append(f"Unexpected binary: {rel}")
                continue
            if path.name != "audit_release.py" and (private.search(text) or secrets.search(text)):
                failures.append(f"Private host/path or credential pattern: {rel}")
            if path.suffix == ".py":
                for node in ast.walk(ast.parse(text)):
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [x.name for x in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        modules = [node.module]
                    if any(x.startswith(("machgen", "flash_attention_plus")) for x in modules):
                        failures.append(f"Application import: {rel}")
            files.append(str(rel))
    manifest = json.loads((root / "src/vc_attn/source_manifest.json").read_text())
    for version in manifest["versions"].values():
        for rel, record in version["files"].items():
            if hashlib.sha256((root / rel).read_bytes()).hexdigest() != record["sha256"]:
                failures.append(f"Changed frozen kernel: {rel}")
    extra_hashes = {
        "src/vc_attn/quantization.py": manifest["quantization"]["sha256"],
        **{
            "src/vc_attn/native/" + name: digest
            for name, digest in manifest["native_v6"].items()
            if name != "note"
        },
    }
    for rel, digest in extra_hashes.items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            failures.append(f"Changed frozen support source: {rel}")
    for name in ("LICENSE", "NOTICE", "AUTHORS", "LICENSES/Apache-2.0.txt"):
        if not (root / name).is_file():
            failures.append(f"Missing attribution: {name}")
    return {"ok": not failures, "file_count": len(files), "failures": failures}


if __name__ == "__main__":
    result = audit(Path(__file__).resolve().parents[1])
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
