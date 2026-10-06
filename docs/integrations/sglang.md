# SGLang diffusion integration

Install Open-VC in the same Python environment as SGLang. Check out SGLang revision `0318a8d0af86ba14a05ca093aa43fafe446da23e`, inspect the registration patch, then apply it:

```bash
python integrations/sglang/install.py /path/to/sglang
python integrations/sglang/install.py /path/to/sglang --apply
export OPEN_VC_ATTN_IMPLEMENTATION=open-vc
export OPEN_VC_ATTN_MODE=expcast
```

Select `OPEN_VC_ATTN` in the diffusion transformer's attention backend configuration. The registration resolves to `open_vc_attn.integrations.sglang.OpenVCAttentionBackend`; it does not change SGLang's global default. Keep other model components on their supported backends. The adapter accepts dense noncausal MHA, head dimension 128, FP16/BF16; it rejects unsupported causal/GQA/dropout settings. It includes quantization in the operator call.

Packed calls require explicit host sequence boundaries. Each sequence is processed independently so quantization scales do not cross segment boundaries. Trailing padding must be declared; ring rotation is unsupported. Sequence-parallel communication remains SGLang's responsibility.

Use `--revert` to remove the patch. Both directions validate file hashes against the manifest; drift requires a reviewed new patch. Patch application and isolated adapter tests do not establish full model or multi-GPU inference correctness. Validate the complete model workload before deployment.
