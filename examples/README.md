# Usage examples

| File | Purpose |
|---|---|
| [quickstart.py](quickstart.py) | Creates synthetic BF16 Q/K/V, calls the convenience and explicit-preparation APIs, and checks equal outputs |
| [minimax_sglang.json](minimax_sglang.json) | Starting configuration for the pipeline benchmark: model/server settings and fixed sampling settings |

After installation, run from the repository root on a supported Blackwell GPU:

```bash
python examples/quickstart.py
```

Edit the MiniMax configuration for your accessible checkpoint and supported
workload, then follow the [SGLang guide](../docs/sglang.md). It is a configuration
example, not a bundled model or evidence of completed end-to-end validation.
The JSON keeps text-encoder and VAE attention on their specified compatible backends.

[Repository](../README.md)
