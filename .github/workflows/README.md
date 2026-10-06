# CI workflows

| File | Trigger and environment | Checks |
|---|---|---|
| [cpu.yml](cpu.yml) | Push / pull request; hosted Linux, Python 3.10 and 3.12 | Constraints-checked editable installation without GPU dependencies, lint/format, CPU contracts, release audit, wheel and sdist build |
| [gpu.yml](gpu.yml) | Manual dispatch; self-hosted `linux`, `x64`, `blackwell` runner | Installation/GPU smoke, GPU tests, large pipeline test, benchmark smoke run; uploads JSON results |

The GPU workflow assumes a reserved GPU and compatible CUDA environment.
CPU CI does not establish kernel correctness or GPU performance.

See [test commands](../../tests/README.md) and
[performance](../../docs/performance.md).

[Repository](../../README.md) · [Parent directory](../DIRECTORY.md)
