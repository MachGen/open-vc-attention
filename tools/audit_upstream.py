"""Compare an extracted snapshot and quantizers with a local VC-attn Git revision."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRANCH_BASE = "f06014eb6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local upstream Git checkout")
    parser.add_argument("--revision", default="refs/remotes/audit/feat/VC-attn")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(args.source), *argv], text=True)

    revision = git("rev-parse", "--verify", args.revision + "^{commit}").strip()
    manifest = json.loads((ROOT / "src/vc_attn/source_manifest.json").read_text())
    snapshot = manifest["versions"][args.version]
    failures = []
    if snapshot["source_revision"] != revision:
        failures.append("Snapshot revision differs from requested upstream revision")
    matched = []
    for relative, record in snapshot["files"].items():
        suffix = relative.split(f"src/vc_attn/_kernels/{args.version}/", 1)[1]
        original = "flash_attention_plus/" + suffix
        text = git("show", f"{revision}:{original}")
        transformed = text.replace("flash_attention_plus", f"vc_attn._kernels.{args.version}")
        source_hash = hashlib.sha256(text.encode()).hexdigest()
        if source_hash != record["original_sha256"]:
            failures.append(f"Upstream hash mismatch: {original}")
        if transformed.encode() != (ROOT / relative).read_bytes():
            failures.append(f"Namespaced source mismatch: {relative}")
        if hashlib.sha256(transformed.encode()).hexdigest() != record["sha256"]:
            failures.append(f"Manifest hash mismatch: {relative}")
        matched.append(original)
    changed = git(
        "diff", "--name-only", BRANCH_BASE, revision, "--", "flash_attention_plus"
    ).splitlines()
    runtime_changes = [
        path
        for path in changed
        if (path.startswith("flash_attention_plus/flash_attn/") and path.endswith(".py"))
        or path in ("flash_attention_plus/nvfp4.py", "flash_attention_plus/v_smooth.py")
    ]
    for path in runtime_changes:
        if path not in matched:
            failures.append(f"Changed runtime file absent from extraction: {path}")

    def functions(text):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(text).body
            if isinstance(node, ast.FunctionDef)
        }

    upstream = functions(git("show", f"{revision}:machgen/models/ops/mm/triton_kernels.py"))
    extracted = functions((ROOT / "src/vc_attn/quantization.py").read_text())
    for name in manifest["quantization"]["functions"]:
        if name not in extracted or extracted[name] != upstream.get(name):
            failures.append(f"Quantizer function differs: {name}")
    result = {
        "ok": not failures,
        "upstream_revision": revision,
        "version": args.version,
        "matched_source_file_count": len(matched),
        "quantizer_function_count": len(manifest["quantization"]["functions"]),
        "changed_runtime_files": runtime_changes,
        "branch_commits": git(
            "log", "--reverse", "--format=%H %s", f"{BRANCH_BASE}..{revision}"
        ).splitlines(),
        "failures": failures,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
