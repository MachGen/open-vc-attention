# v3: minimal modified_utils namespace

This directory contains only [__init__.py](__init__.py), a minimal package
marker retained in the isolated `v3` source tree. It has no public API
and performs no compiler patching or helper initialization.

The active implementation is in [the parent CuTe directory](../README.md).
Preserve the initializer's recorded source hash when maintaining this snapshot;
put new shared integration behavior in the standalone API or adapter layer.

[Repository](../../../../../../../README.md) · [Parent directory](../README.md)
