"""Build the optional native v6 baseline using an explicit CUDA target."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["sm_103a"], required=True)
    p.add_argument("--output", type=Path, default=Path("build/libvc_native_v6.so"))
    args = p.parse_args()
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        p.error("CUDA 13 nvcc required; set CUDA_HOME to its toolkit or add nvcc to PATH")
    cuda = Path(nvcc).resolve().parents[1]
    source = Path(__file__).resolve().parents[1] / "src/vc_attn/native/fa_fp8_v6.cu"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        f"-arch=compute_{args.arch.removeprefix('sm_')}",
        f"-code={args.arch}",
        "--use_fast_math",
        "-lineinfo",
        "-Xptxas=-v",
        "-Xcompiler=-fPIC",
        "-shared",
        str(source),
        "-o",
        str(args.output),
        "-L" + str(cuda / "lib64/stubs"),
        "-lcuda",
    ]
    subprocess.run(cmd, check=True)
    print(args.output)


if __name__ == "__main__":
    main()
