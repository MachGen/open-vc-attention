# Python source tree

This project uses a `src` package layout. The installable
[vc_attn/](vc_attn/README.md) package contains the inference API, benchmark CLI,
frozen kernel snapshots, framework adapters and optional native sources.

Install from the repository root before importing:

```bash
python -m pip install -e '.[dev]'
```

[pyproject.toml](../pyproject.toml) defines dependencies, package data and the
`vc-attn-bench` / `vc-attn-info` commands. Read
[the package map](vc_attn/README.md) to find an implementation and
[CONTRIBUTING](../CONTRIBUTING.md) before editing a frozen snapshot.

[Repository](../README.md)
