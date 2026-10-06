# CI workflows

| File | Trigger and environment | Checks |
|---|---|---|
| [cpu.yml](cpu.yml) | Push / pull request; hosted Linux, Python 3.10 and 3.12 | Constraints-checked editable installation without GPU dependencies, lint/format, CPU contracts, source audit, wheel and sdist build |
| [gpu.yml](gpu.yml) | Manual dispatch; self-hosted `linux`, `x64`, `blackwell` runner | Installation/GPU smoke, GPU tests, large scaled-path test, benchmark smoke run; uploads JSON results |

The GPU workflow assumes the maintainer has reserved an appropriate GPU and
installed a compatible CUDA environment. Its ordinary invocation skips optional
native DSO tests unless `VC_ATTN_NATIVE_LIBRARY` is supplied by the environment.
CPU CI does not establish kernel correctness or GPU performance.

See [test commands](../../tests/README.md) and
[recorded validation](../../docs/performance.md).

[Repository](../../README.md) · [Parent directory](../DIRECTORY.md)
