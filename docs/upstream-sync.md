# VC-attn synchronization

The default `v4` snapshot is extracted from `MachGenPlatform`'s `feat/VC-attn`
at **`8aa761eac845d734dc5dbe48a6196c1fe1b0a7cf`**, fetched on 2026-09-30.
Only namespace imports change during extraction. All 33 extracted files match
the upstream source after that substitution; all eight quantizer functions
match the upstream AST. Every runtime file changed on the feature branch is
covered by the extraction. Original and transformed hashes are in
[`source_manifest.json`](../src/vc_attn/source_manifest.json).

## Commit coverage

The audited branch range starts after `f06014eb6`, before VC consolidation.

| Upstream commit | Changes and standalone disposition |
|---|---|
| `0924a347f` | ExpCast, NVFP4 and V-Smooth consolidation; retained in snapshot sources and advanced preparation helpers |
| `7fcd99433` | V-Smooth / ExpCast optimization; retained |
| `7d52e40dc` | Tensor Core V-mean restoration; retained |
| `f4636337a` | Inline ExpCast rescaling; retained |
| `7b3cb28db` | Deferred correction register allocation; retained |
| `77fad93a0` | Softmax-warp epilogue; retained |
| `632789fae` | Historical DSL validation notes; upstream evidence, not a new package measurement |
| `d902dc122` | B300 ExpCast and V-Smooth schedules; retained |
| `d2617dfd1` | Branch merge; included in ancestry |
| `da40905c3` | Historical B200 regression evidence; no additional runtime patch |
| `4245ca87a` | Scaled ExpCast fast path and correct block-K descales; retained in `scaled` and `v4` |
| `ab627b4db` | Platform H3 end-to-end benchmark records; no kernel changes |
| `8aa761eac` | Previously missing: original-scan unscaled ExpCast and both SM103 ordinary-FP8 scheduling changes; now in `v4` |

The new commit changes three runtime files: `interface.py`, `flash_fwd_sm100.py`
and `softmax.py`. It enables packed V / inline rescale for eligible unscaled
FP8 ExpCast with `mid_window_blocks=None`. For eligible SM103 ordinary FP8,
it interleaves score scaling with exp2 and delays the correction wait until
after row-sum computation. The latter two changes remain disabled for SM100,
ExpCast, skipping, V-Smooth and mid-window scans.

The current public API and diffusion adapter use `v4` with mid-window 4;
the default benchmark uses `vc_v4_mid4`. Eligible B200 calls select source-level
fusedpipe, with the experimental D patch disabled. `vc_v4` retains its
original-scan identity and `fp8_v4` selects ordinary FP8. Report rendering
prefers `vc_v4_mid4` when present, with historical `vc_v4` / `vc_scaled` fallbacks.
The archived performance tables continue to describe their recorded revisions.

The extraction and GPU-validation results below describe the 2026-09-30
snapshot. Later standalone changes include the B200 packing gate, fusedpipe and
experimental D integration; see the per-file transformations in the manifest.
Current D policy is documented in the [API guide](integration.md#scan-order).

## Integration and scope

Platform request routing, H3 weight recipes, video fixtures, capture scripts and
end-to-end generation runners are application-specific and are not copied into
the standalone package. Their attention controls are exposed through `mode`,
`version`, explicit preparation and the diffusion adapter. Historical upstream
model measurements do not establish standalone model-quality or E2E validation.
Retained low-level NVFP4, V-Smooth, backward and sparse helpers do not expand
the documented public API contract.

## Recheck against upstream

Fetch the feature branch in a local upstream clone, then run:

```bash
python tools/audit_upstream.py /path/to/upstream \
  --revision origin/feat/VC-attn --version v4
python tools/audit_release.py
python -m pytest tests/cpu
VC_ATTN_TEST_LARGE=1 python -m pytest tests/test_v4_gpu.py tests/test_gpu.py
```

The upstream audit rejects a revision mismatch, missing changed runtime files,
source/hash differences and quantizer-function differences. It also prints
the exact commit range so a later branch update cannot pass as this revision.

GPU tests include the upstream original-scan partial-tile case, LSE fallback
and packing selection, bitwise scaled-path comparisons against the preserved
snapshot for both scan orders, and SM103 ordinary-FP8 comparisons with and
without descales/LSE. These are numerical regression checks, not performance
claims.

## Validation (2026-09-30)

- Upstream audit: all 33 extracted files, eight quantizer functions and nine
  changed runtime files verified; no differences beyond namespace substitution.
- CPU suite: 13 passed, four subtests passed.
- B200 (SM100): 11 passed, seven skipped (four SM103-only cases and three
  optional native-library cases).
- B300 (SM103): 15 passed, three optional native-library cases skipped.
- Both GPU runs enabled the large-sequence cases and used PyTorch 2.11.0+cu130,
  CUTLASS DSL 4.6.0 and quack 0.6.1. The GPUs had resident allocations from
  other processes, so these runs establish numerical correctness only.
- Ruff lint/format, release hash audit, and wheel/source-distribution build
  passed. Historical snapshots retain their original hashes.

These checks do not claim a new 1.9x measurement. The added B200 fast-path
eligibility concerns unscaled original-scan inputs; scaled ExpCast retains
the previous implementation, including its scan-order performance distinction.
