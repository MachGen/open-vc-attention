"""Apply the narrow registration patch only to its verified source preimages."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--revert", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = json.loads((root / "integrations/sglang/manifest.json").read_text())
    patch = root / "integrations/sglang/register.patch"
    if args.apply and args.revert:
        p.error("Choose --apply or --revert")
    for rel, hashes in spec["files"].items():
        path = args.checkout / rel
        expected = hashes["patched_sha256" if args.revert else "original_sha256"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            p.error(f"Unexpected SGLang file {rel}; use pinned revision {spec['revision']}")
    command = ["git", "apply", "--check"]
    if args.revert:
        command.append("--reverse")
    subprocess.run(command + [str(patch)], cwd=args.checkout, check=True)
    if args.apply or args.revert:
        command.remove("--check")
        subprocess.run(command + [str(patch)], cwd=args.checkout, check=True)
        print("Registration patch reverted" if args.revert else "Registration patch applied")
    else:
        print(patch.read_text())
        print("Preimages match. Re-run with --apply to install.")


if __name__ == "__main__":
    main()
