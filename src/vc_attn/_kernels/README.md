# Frozen CuTe kernel snapshots

Each subdirectory is an isolated namespace for a pinned attention revision.
Keeping snapshots separate makes historical comparisons reproducible without
swapping source files at runtime.

| Snapshot | Role |
|---|---|
| [baseline/](baseline/README.md) | Ordinary BF16 and FP8 performance/numerical reference |
| [v1/](v1/README.md) | First historical VC snapshot; extracted tuning constants and cleanup |
| [v2/](v2/README.md) | Shared launch/register rules and softcap/batch-rank fixes |
| [v3/](v3/README.md) | Public forward API, dispatch/packaging and batch-descale changes |
| [scaled/](scaled/README.md) | Fused ExpCast dispatch with external descales |
| [v4/](v4/README.md) | Default: original-scan ExpCast and SM103 ordinary-FP8 scheduling |

Use the version selector in [api.py](../api.py), not direct imports in model
code. The [source manifest](../source_manifest.json) records original and
namespaced file hashes. Package initializers are minimal to avoid unrelated
imports and global initialization patches. Existing numerical snapshots are
frozen; add a new version for algorithm changes instead of rewriting a baseline.

These directory READMEs describe the extraction; they are not part of the
original hashed kernel files. See [version behavior](../../../docs/appendix/versions.md).

[Repository](../../../README.md) · [Parent directory](../README.md)

Historical source comments may mention older dependency releases. For this
package, follow the root installation guide and `requirements-tested.txt`,
including CuTe DSL 4.6.0 and quack 0.6.1.
