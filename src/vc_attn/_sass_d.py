"""Fail-closed, in-memory scheduling patch for a verified SM100 executable family.

This is not a general SASS assembler. Four verified scan immediates must equal
the caller's compile-time window; all other code bits and ELF metadata must match
before ten instruction slots are changed. Different compiler output is left
untouched. No compiler cache, file, CUDA context, driver function or process-global
hook is modified here.
"""

from __future__ import annotations

import hashlib
import json
import struct

PATCH_REVISION = "sm100-score-two-body-window-v2"
_TEXT_SIZE = 87040
_ORIGINAL_TEXT_SHA = "3f8c9a6f08115780f331d674a3effa77410d88bd2fe9ee91c9fffeed74367591"
_PATCHED_TEXT_SHA = "040efd30afd2544024a7a36046ad8382d4687d2d3d6ce6da9fee4d60b0b6755f"
_METADATA_SHA = "9ab3d51a5555fc0a52621a3623fe36ddd0fa20d1a1802bfd5765ac6e23946e42"
# These ULEA.HI.SX32 instructions compute the same window offset for the
# producer and consumers. Only their 32-bit immediate operands may vary.
_WINDOW_PCS = (0x1740, 0x5FC0, 0x7CA0, 0xA570)
# 128-bit integers; the ELF stores each instruction little-endian.
_WORDS = (
    (0xA750, "000fe400082c00000000000a002479ee", "000e6400082c00000000000a002479ee"),
    (0xA790, "000fe200000e000000000000000972ca", "002fe4000780002600000025244d7276"),
    (0xA7A0, "000fe200000000000000000000007918", "000fc4000780002900000028274c7276"),
    (0xA7C0, "000fe4000780002600000025244d7276", "000fe200000e000000000000000972ca"),
    (0xA7D0, "000fc4000780002900000028274c7276", "000fe200000000000000000000007918"),
    (0xDEA0, "000fe400082c000000000006002479ee", "000ea400082c000000000006002479ee"),
    (0xDEE0, "000fe200000e000000000000000572ca", "004fe4000780002600000025244c7276"),
    (0xDEF0, "000fe200000000000000000000007918", "000fc4000780002900000028274b7276"),
    (0xDF10, "000fe4000780002600000025244c7276", "000fe200000e000000000000000572ca"),
    (0xDF20, "000fc4000780002900000028274b7276", "000fe200000000000000000000007918"),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _need(condition: bool) -> None:
    if not condition:
        raise ValueError("Unsupported ELF layout")


def _elf(data: bytes) -> dict:
    """Strict ELF64-LE reader; extended numbering and overlapping payloads fail."""
    _need(isinstance(data, bytes) and 64 <= len(data) <= 256 * 1024 * 1024)
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    _need(h[0][:7] == b"\x7fELF\x02\x01\x01" and h[3] == 1 and h[8] == 64)
    _need(h[11] == 64 and 0 < h[12] <= 4096 and 0 < h[13] < h[12])
    _need(h[6] >= 64 and h[6] + h[11] * h[12] <= len(data))
    _need(h[10] <= 64 and (h[10] == 0 or (h[9] == 56 and h[5] >= 64)))
    _need(h[5] + h[9] * h[10] <= len(data))
    raw = [struct.unpack_from("<IIQQQQIIQQ", data, h[6] + i * 64) for i in range(h[12])]
    _need(raw[0] == (0,) * 10)
    names = raw[h[13]]
    _need(names[1] == 3 and names[4] + names[5] <= len(data))
    strings = data[names[4] : names[4] + names[5]]
    sections = []
    occupied = [(0, 64), (h[6], h[6] + h[11] * h[12])]
    if h[10]:
        occupied.append((h[5], h[5] + h[9] * h[10]))
    extent = max(end for _, end in occupied)
    for i, s in enumerate(raw):
        _need(s[0] < len(strings) and s[6] < len(raw))
        end = strings.find(b"\0", s[0])
        _need(end >= 0)
        name = strings[s[0] : end].decode("ascii")
        _need(s[8] == 0 or s[8] & (s[8] - 1) == 0)
        _need(s[8] <= 1 << 30)
        _need(s[1] == 8 or s[4] + s[5] <= len(data))
        if s[1] != 8 and s[5]:
            occupied.append((s[4], s[4] + s[5]))
            extent = max(extent, s[4] + s[5])
        sections.append(
            dict(
                index=i,
                name=name,
                type=s[1],
                flags=s[2],
                address=s[3],
                offset=s[4],
                size=s[5],
                link=s[6],
                info=s[7],
                alignment=s[8],
                entry_size=s[9],
            )
        )
    _need(len({s["name"] for s in sections}) == len(sections))
    occupied.sort()
    _need(all(a[1] <= b[0] for a, b in zip(occupied, occupied[1:])))
    programs = []
    for i in range(h[10]):
        p = struct.unpack_from("<IIQQQQQQ", data, h[5] + i * 56)
        _need(p[2] + p[5] <= len(data) and p[5] <= p[6])
        extent = max(extent, p[2] + p[5])
        programs.append(p)
    return dict(header=h, sections=sections, programs=programs, extent=extent)


def _metadata_fingerprint(data: bytes, elf: dict, entry: str) -> str:
    """Only names/string-table storage offsets vary; executable metadata is exact.

    File offsets may shift with names. Program-segment locations are instead
    bound to their section indices or to the ELF program-header table. Symbol
    tables, constants, relocation data and Mercury bytes are hashed unchanged.
    An unfamiliar harmless difference deliberately falls back too.
    """
    h = elf["header"]
    rows = []
    for s in elf["sections"]:
        row = {k: v for k, v in s.items() if k not in ("name", "offset")}
        row["name"] = s["name"].replace(entry, "<kernel>")
        if s["type"] == 3:
            # Only this kernel's spelling may differ. All other strings and
            # their order remain exact, as do type/link/index/flags/alignment.
            row.pop("size")
            payload = data[s["offset"] : s["offset"] + s["size"]]
            row["payload"] = _sha(payload.replace(entry.encode("ascii"), b"<kernel>"))
        elif s["name"] == ".text." + entry:
            row["payload"] = "verified-separately"
        elif s["type"] != 8:
            row["payload"] = _sha(data[s["offset"] : s["offset"] + s["size"]])
        rows.append(row)
    programs = []
    for p in elf["programs"]:
        anchor = (
            "program_headers"
            if p[2] == h[5]
            else [s["index"] for s in elf["sections"] if s["offset"] == p[2]]
        )
        _need(bool(anchor))
        programs.append([p[0], p[1], anchor, *p[3:]])
    canonical = dict(header=[h[0].hex(), *h[1:5], h[7], *h[8:]], sections=rows, programs=programs)
    return _sha(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode())


def _normalized_text(text: bytes, mid_window_blocks: int) -> bytes:
    """Bind all four literals to the caller, then compare the complete code."""
    _need(type(mid_window_blocks) is int and 0 <= mid_window_blocks <= 0x7FFFFFFF)
    _need(len(text) == _TEXT_SIZE)
    normalized = bytearray(text)
    for pc in _WINDOW_PCS:
        _need(struct.unpack_from("<I", text, pc + 4)[0] == mid_window_blocks)
        struct.pack_into("<I", normalized, pc + 4, 4)
    return bytes(normalized)


def _patch(data: bytes, *, mid_window_blocks: int = 4) -> tuple[bytes, dict] | None:
    elf = _elf(data)
    h = elf["header"]
    _need(h[1:3] == (2, 190) and h[7] == 0x06006402 and elf["extent"] == len(data))
    texts = [s for s in elf["sections"] if s["name"].startswith(".text.")]
    _need(len(texts) == 1)
    text = texts[0]
    entry = text["name"][6:]
    _need(text["size"] == _TEXT_SIZE and text["type"] == 1 and text["flags"] == 6)
    begin, end = text["offset"], text["offset"] + text["size"]
    before_sha = _sha(data[begin:end])
    normalized_before_sha = _sha(_normalized_text(data[begin:end], mid_window_blocks))
    _need(normalized_before_sha in (_ORIGINAL_TEXT_SHA, _PATCHED_TEXT_SHA))
    _need(_metadata_fingerprint(data, elf, entry) == _METADATA_SHA)
    already = normalized_before_sha == _PATCHED_TEXT_SHA
    result = bytearray(data) if not already else None
    for pc, old, new in _WORDS:
        expected = int(new if already else old, 16).to_bytes(16, "little")
        _need(data[begin + pc : begin + pc + 16] == expected)
        if result is not None:
            result[begin + pc : begin + pc + 16] = int(new, 16).to_bytes(16, "little")
    patched = bytes(result) if result is not None else data
    _need(len(patched) == len(data))
    _need(_sha(_normalized_text(patched[begin:end], mid_window_blocks)) == _PATCHED_TEXT_SHA)
    return patched, dict(
        applied=not already,
        status="already_patched" if already else "applied",
        patch_revision=PATCH_REVISION,
        mid_window_blocks=mid_window_blocks,
        kernel_name=entry,
        original_cubin_sha256=_sha(data),
        patched_cubin_sha256=_sha(patched),
        original_text_sha256=before_sha,
        patched_text_sha256=_sha(patched[begin:end]),
        normalized_original_text_sha256=normalized_before_sha,
        normalized_patched_text_sha256=_PATCHED_TEXT_SHA,
    )


def patch_cubin(data: bytes, *, mid_window_blocks: int = 4) -> bytes | None:
    """Patch the exact window family, return already-patched bytes, or None.

    Omitting the keyword retains the original window-4 contract. Explicit None,
    invalid windows, different code layouts and different metadata fail closed.
    """
    try:
        result = _patch(data, mid_window_blocks=mid_window_blocks)
        return result[0] if result is not None else None
    except (ValueError, TypeError, struct.error, UnicodeError, OverflowError):
        return None


def patch_host_object(data: bytes, *, mid_window_blocks: int = 4) -> tuple[bytes, dict] | None:
    """Patch the sole embedded CUDA ELF in an x86-64 relocatable CuTe object.

    Only read-only allocated PROGBITS may contain the cubin. Multiple CUDA ELFs,
    unknown object layouts or any failed cubin identity check return None.
    Outer symbols, relocations and ABI bytes are never changed.
    """
    try:
        host = _elf(data)
        _need(host["header"][1:3] == (1, 62) and host["extent"] == len(data))
        embedded = []
        for section in host["sections"]:
            if section["type"] != 1 or section["flags"] & 7 != 2:
                continue
            start, stop = section["offset"], section["offset"] + section["size"]
            cursor = start
            while True:
                pos = data.find(b"\x7fELF", cursor, stop)
                if pos < 0:
                    break
                cursor = pos + 4
                _need(len(embedded) < 2)
                if pos + 20 > stop or struct.unpack_from("<H", data, pos + 18)[0] != 190:
                    continue
                inner = _elf(data[pos:stop])
                size = inner["extent"]
                _need(pos + size <= stop)
                embedded.append((pos, size))
        _need(len(embedded) == 1)
        offset, size = embedded[0]
        result = _patch(data[offset : offset + size], mid_window_blocks=mid_window_blocks)
        _need(result is not None)
        cubin, info = result
        patched = data[:offset] + cubin + data[offset + size :]
        _need(len(patched) == len(data))
        info.update(
            original_object_sha256=_sha(data),
            patched_object_sha256=_sha(patched),
            cubin_offset=offset,
            cubin_size=size,
        )
        return patched, info
    except (ValueError, TypeError, struct.error, UnicodeError, OverflowError):
        return None
