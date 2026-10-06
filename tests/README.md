# Tests

`cpu/` checks public contracts without Torch or CUDA. `gpu/` checks numerical behavior on Blackwell; large-sequence tests require `OPEN_VC_ATTN_TEST_LARGE=1`. `packaging/` checks built wheel and source distribution contents. See CONTRIBUTING.md.

The large B200 suite covers:

- fused preparation byte equality with general preparation, and three-kernel profiling;
- prepacked-input validation and graph input refresh;
- V repair's denominator invariance and accuracy;
- benchmark CLI smoke runs, with tiny samples explicitly labeled as not isolated.
