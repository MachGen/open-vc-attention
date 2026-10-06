# External framework registration

This directory contains artifacts that register VC Attention with an external
framework. [sglang/](sglang/README.md) holds a narrow registration patch and its
source preimage hashes.

The Python adapter implementation lives in
[src/vc_attn/integrations/](../src/vc_attn/integrations/README.md). Keeping it in
the package lets registration point to a versioned installed implementation.
Model execution, weights and distributed communication remain in the framework.

Use [docs/sglang.md](../docs/sglang.md) for the complete installation and
measurement procedure.

[Repository](../README.md)
