"""Public-facing English PDF; no GPU work and no edits to runtime source."""

import ast
import hashlib
import html
import json
import math
import re
import statistics
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "docs/reports"
SCRATCH = ROOT / "build/report-pdf/en"
OUT = WORK / "VC-Attention-Fusedpipe-D-Technical-Report.pdf"
REPO = ROOT
T = json.loads((WORK / "captured-summary.json").read_text())
LONG = json.loads((WORK / "VC-Attention-Long-Sequence-Results.json").read_text())
SHA = "cbb2da8e9b41d146e1593a3c2946b4608f6cb2ff"
BASE = "https://github.com/MachGen/vc-attention/blob/" + SHA + "/"
KERNEL = "src/vc_attn/_kernels/v4/flash_attn/cute/flash_fwd_sm100.py"
NAVY, TEAL, INK, MUTED, LIGHT, RULE = map(
    colors.HexColor, ["#15354C", "#087F8C", "#20323C", "#60727E", "#EDF5F7", "#D3DFE5"]
)
W, H = A4
CW = W - 88
S = {
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.6, leading=14.1, textColor=INK, spaceAfter=8
    ),
    "small": ParagraphStyle(
        "small", fontName="Helvetica", fontSize=8.2, leading=11.7, textColor=MUTED, spaceAfter=6
    ),
    "h1": ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=NAVY, spaceAfter=13
    ),
    "h2": ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=11.8,
        leading=16.5,
        textColor=TEAL,
        spaceBefore=8,
        spaceAfter=6,
    ),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.1, leading=11.5, textColor=INK),
    "th": ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=8.0, leading=11.3, textColor=colors.white
    ),
    "code": ParagraphStyle(
        "code", fontName="Courier", fontSize=8.0, leading=11.1, textColor=INK, spaceAfter=4
    ),
}
story = []


def p(text, kind="body"):
    story.append(Paragraph(text, S[kind]))


def h(text):
    p(text, "h2")


def page(n, title):
    if story:
        story.append(PageBreak())
    p(f"TECHNICAL REPORT / {n:02d}", "small")
    p(title, "h1")


def table(headers, rows, widths=None):
    data = [[Paragraph(html.escape(str(x)), S["th"]) for x in headers]]
    data += [
        [Paragraph(html.escape(str(x)).replace("\n", "<br/>"), S["cell"]) for x in row]
        for row in rows
    ]
    ws = (
        [CW / len(headers)] * len(headers)
        if widths is None
        else [CW * x / sum(widths) for x in widths]
    )
    t = Table(data, colWidths=ws, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
            ]
        )
    )
    story.extend([t, Spacer(1, 8)])


def code(text, size=8.0):
    st = ParagraphStyle("c", parent=S["code"], fontSize=size, leading=size * 1.39)
    block = Preformatted(text.strip("\n"), st)
    t = Table([[block]], colWidths=[CW])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([t, Spacer(1, 7)])


def note(text):
    t = Table([[Paragraph(text, S["body"])]], colWidths=[CW])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.extend([t, Spacer(1, 7)])


def source(label, path, line, desc):
    p(
        f'[{label}] <link href="{BASE}{path}#L{line}" color="#087F8C">{desc} (line {line})</link>',
        "small",
    )


class Pipeline(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = CW, 107

    def draw(self):
        c = self.canv
        for y, label, items in [
            (
                69,
                "Before",
                [
                    (59, 127, "Wait / read scores"),
                    (198, 95, "Read K scale"),
                    (305, 190, "Max / ExpCast"),
                ],
            ),
            (
                25,
                "Fusedpipe",
                [
                    (59, 95, "Read K scale"),
                    (166, 127, "Wait / read scores"),
                    (305, 190, "Max / ExpCast"),
                ],
            ),
        ]:
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 8)
            c.drawString(0, y + 11, label)
            for x, w, t in items:
                c.setFillColor(LIGHT)
                c.roundRect(x, y, w, 29, 4, fill=1, stroke=0)
                c.setFillColor(INK)
                c.drawCentredString(x + w / 2, y + 11, t)
                if x + w < 495:
                    c.setStrokeColor(TEAL)
                    c.line(x + w + 2, y + 14, x + w + 10, y + 14)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7.5)
        c.drawString(0, 3, "Logical issue order only. Box widths do not represent measured cycles.")


page(1, "VC Attention<br/>Fusedpipe and D Scheduling")
p("B200 / SM100 | Technical report | 4 October 2026", "h2")
p(
    "Two incremental scheduling optimizations for an existing dense FP8 attention kernel. Fusedpipe exposes independent reads in CuTe source. D changes dependency control and local max scheduling in the compiled SASS. This report separates their matched measurements from the larger benefit of the complete VC attention algorithm."
)
table(
    ["Layer", "Change", "Real-capture increment"],
    [
        [
            "Existing VC algorithm",
            "FP8, ExpCast, Tensor Core denominator and inline output rescaling form the common base.",
            "Not decomposed here",
        ],
        [
            "Fusedpipe",
            "Read the current K descale earlier; replace four 32-column output correction fragments with two preloaded 64-column fragments.",
            "1.013014x versus pre-change VC",
        ],
        [
            "D",
            "Assign an independent dependency slot to the first score load; move two existing max leaves before the full-read completion point.",
            "1.002647x versus fusedpipe",
        ],
    ],
    [1.1, 3.1, 1.6],
)
note(
    "Both optimizations schedule existing work more effectively. Neither adds warps, removes attention interactions, changes the softmax reduction tree, nor introduces a new approximation. Byte equality is established against the matching FP8 control, not against BF16."
)
h("Scope of the main results")
p(
    "One noncausal sequence, equal query/KV head counts, D=128 and S=73,426 on NVIDIA B200. Inputs are privately captured MiniMax-H3 attention tensors from step 24, layers 0 and 24. H=7 selects heads 0-6 from H=56 captures; it is not a live context-parallel execution. All Q/K/V descales are supplied, mid_window_blocks=4, and all skipping is disabled."
)
p(
    "<b>Timing is kernel-only:</b> one attention kernel replayed under CUDA-event timing. Input quantization, V packing, compilation, allocation, communication and the rest of model inference are excluded. The two incremental ratios above come from separate 20-round paired cohorts and must not be multiplied into a new measured result."
)
p(
    "<b>Release policy:</b> since 7ae8e21, the public default is mid4 with eligible B200 fusedpipe; D is disabled. Historical measurements below retain revision cbb2da8. Pages 13-14 document the public reproduction tools and recovered environment.",
    "small",
)
if LONG is not None:
    p(
        "Section 10 adds a separate four-case, fixed-seed synthetic sweep at S=188,214 and 262,144. It includes freshly paired BF16 timings. D has a small median regression in one case; the new sweep does not establish a uniform D benefit. Raw paired samples and seeds are supplied in the companion JSON.",
        "small",
    )
else:
    p("Longer-sequence results, when accepted, are reported separately in Section 10.", "small")

page(2, "01  Mathematics and pipeline context")
p(
    "Let r_ij be a raw FP8 Q-K dot product. dQ_i denotes the Q block descale for row i, dK_b the current K block descale, and sigma=1/sqrt(D). D is the per-head Q/K dimension, 128 here. For this scaled path, the ideal restored logit is sigma * dQ_i * dK_b * r_ij."
)
code(
    "c_i     = sigma * dQ_i * log2(e)\ntilemax = dK_b * max_j(r_ij)\nalpha_i = 2 ** (c_i * (M_i - M_i_new))\nA_new   = alpha_i * A_old + P_block @ V_fp8\nl_new   = alpha_i * l_old + sum(P_block)\nO_final = dV * A_final / l_final"
)
p(
    'A is the unnormalized output numerator and l is the denominator. Kernel code also calls the TMEM numerator accumulator O; "O rescale" below refers to this accumulator, not a previously normalized final output. The equations describe online-softmax scaling. Actual P uses the existing FP8 ExpCast encoder, Tensor Core denominator and max deadband policy, so this is not exact BF16 softmax.'
)
h("Why numerator and denominator must both be rescaled")
p(
    "When the running maximum changes, prior probabilities are expressed relative to an older exponent origin. Multiplying both accumulators by the same alpha preserves their relationship. Fusedpipe changes when the values are loaded and how columns are grouped; it retains the same alpha and per-element multiply."
)
table(
    ["Role", "Existing work", "Change in this report"],
    [
        [
            "Softmax warps 0-7",
            "Read score TMEM, reduce row maxima, encode P and perform the established inline correction work.",
            "Internal scheduling only",
        ],
        [
            "Other CTA roles",
            "Load, MMA and service roles; the CTA has 12 warps / 384 threads.",
            "No new warps or participation changes",
        ],
        [
            "Producer-consumer handoff",
            "S_full publishes scores; P_full_O_rescaled publishes first P and corrected O.",
            "Protocol retained",
        ],
    ],
    [1.35, 2.9, 1.55],
)
p(
    "K descale scales the tile maximum and the raw-score encoder multiplier. The ExpCast bias is first formed using the base code scale and the selected row maximum; K descale is not multiplied into that bias a second time. These operation and rounding choices are unchanged.",
    "small",
)
source("S1", KERNEL, 807, "Warp roles and register allowances")
source("S2", KERNEL, 3907, "ExpCast scale and bias construction")
source(
    "S3",
    "src/vc_attn/_kernels/v4/flash_attn/cute/softmax.py",
    213,
    "Existing maximum and rescale update",
)

page(3, "02  Fusedpipe: read K descale earlier")
p(
    "The original path waits for QK scores, reads and releases the score buffer, applies masking, and then reads the current K block descale. Fusedpipe moves that same immutable scalar read before the S_full wait. Its batch, KV-head and block indices are already available."
)
story.append(Pipeline())
code(
    "# Before\nwait(S_full[stage])\nscores = load_score_TMEM()\nwait_score_loads_and_arrive_score_empty()\napply_mask(scores)\nk_scale = K_descale[batch, kv_head, n_block]\n\n# Fusedpipe\nk_scale = K_descale[batch, kv_head, n_block]\nwait(S_full[stage])\nscores = load_score_TMEM()\nwait_score_loads_and_arrive_score_empty()\napply_mask(scores)"
)
h("What can overlap")
p(
    "The scalar load does not depend on the QK result. It can progress while the softmax warp waits for S_full. In canonical mid4 SASS, LDG.E at 0xa6f0 precedes the score-ready TRYWAIT at 0xa710. This is the current n_block, not speculative prefetch of the next block."
)
h("Why the lifetime remains valid")
p(
    "The index, value, number of reads and consumption sites are unchanged. Score TMEM loads remain after S_full, and score_empty remains after score-load completion. The descale is an input scalar, not a producer-owned score buffer being overwritten by MMA."
)
note(
    "This transformation moves a logical load; it neither adds preprocessing nor reduces its logical data payload. Cache and transaction behavior affect physical traffic. The measured fusedpipe increment includes both source changes and any resulting compiler scheduling differences; it is not an isolated measurement of this LDG."
)
source("S4", KERNEL, 3665, "Eligibility, early descale load and unchanged S_full wait")
source("S5", KERNEL, 3792, "Original descale load retained as a fallback")

page(4, "03  Fusedpipe: preload two O halves")
p(
    "The original correction helper handles four 32-column fragments as read-multiply-write chains. The new helper issues two independent 64-column loads before the first dependent multiply. The second read can progress while the first half is multiplied and stored."
)
code(
    "# Before\nfor cols in [0:32, 32:64, 64:96, 96:128]:\n    x = load_TMEM(O[cols])\n    store_TMEM(O[cols], x * alpha)\n\n# Fusedpipe\na = load_TMEM(O[0:64])\nb = load_TMEM(O[64:128])\nstore_TMEM(O[0:64],   a * alpha)\nstore_TMEM(O[64:128], b * alpha)\n\n# Unchanged: denominator remains LAST\nl = load_TMEM(denominator)\nstore_TMEM(denominator, l * alpha)\nfence_TMEM_stores()"
)
table(
    ["Canonical PC", "Operation", "Meaning"],
    [
        ["b210 / b230", "LDTM.x64 / LDTM.x64", "Both O reads precede consumption"],
        ["b250", "First FMUL2", "First dependent O multiply"],
        ["b470 / b660", "STTM.x64 / STTM.x64", "Store the two corrected halves"],
        ["b670 / b680 / b690", "LDTM / FMUL / STTM", "Denominator remains after both O halves"],
        ["b6a0 / b6e0", "FENCE / ARRIVE", "Publish only after correction stores"],
    ],
    [1.05, 1.8, 2.75],
)
p(
    "For each logical row that executes correction, the payload remains 128 FP32 O elements plus one FP32 denominator: <b>516 bytes read and 516 bytes written</b>. These are logical TMEM bytes, not HBM transactions. No additional Q/K/V TMA transfers or TMEM allocation are introduced."
)
p(
    "First tiles do not read uninitialized O. The original warp ballot still controls rescaling; no-rescale warps do not perform new unconditional O reads. The first P segment is stored before correction but is not published to MMA until the correction fence. Source pseudocode is not a promise of every final SASS issue order.",
    "small",
)
source("S6", KERNEL, 4734, "Two-half helper; denominator still read last")
source("S7", KERNEL, 4027, "Conditional correction and P publication")

page(5, "04  D: use the first score segment sooner")
p(
    "D modifies two repeated softmax bodies in the fusedpipe binary. Each body changes five existing 16-byte instruction slots: ten slots and 74 changed bytes in total. The issue PCs of all four score LDTM.x32 instructions remain fixed. Two existing max leaves move into earlier slots, while a uniform-register move and a NOP move later."
)
table(
    ["Body 1 PC", "Original fusedpipe", "D"],
    [
        ["a750", "LDTM.x32 R36, tmem[UR10]", "Same load; add write dependency B1"],
        [
            "a760-a780",
            "Other three score loads; final load writes B0",
            "All instruction words unchanged",
        ],
        ["a790", "R2UR UR9, R0", "FMNMX3 R77,R36,R37,R38,!PT; wait B1"],
        ["a7a0", "NOP", "FMNMX3 R76,R39,R40,R41,!PT"],
        ["a7b0", "score_empty ARRIVE; wait B0", "Same PC and complete instruction word"],
        ["a7c0", "First max leaf, producing R77", "R2UR UR9,R0"],
        ["a7d0", "Second max leaf, producing R76", "NOP"],
    ],
    [0.85, 2.1, 2.85],
)
h("Load granularity and wait granularity are different")
p(
    "The first x32 load produces R36-R67. The next three produce R4-R35, R164-R195 and R132-R163. The two moved max leaves consume only R36-R41. Assigning a separate dependency slot to the first load allows those leaves to wait for that segment rather than for all four segments."
)
code(
    "Before: LD0 LD1 LD2 LD3 | R2UR NOP | WAIT_ALL/ARRIVE | MAX0 MAX1\nD:      LD0 LD1 LD2 LD3 | WAIT_0/MAX0 MAX1 | WAIT_ALL/ARRIVE\n                                                | R2UR NOP",
    7.4,
)
p(
    'B0/B1/B2 identify hardware dependency scoreboard slots, not shared-memory mbarrier indices. "Write B1" records an asynchronous result dependency; it does not write application data to memory. The original full-read wait still constrains score_empty publication.'
)
note(
    "D does not insert max instructions between score loads, omit a score load or remove completion. It makes two existing max leaves available after the first segment is ready, while later segments may still be returning. This is a dependency-level scheduling explanation, not a measured per-instruction cycle attribution."
)
source("S8", "src/vc_attn/_sass_d.py", 25, "Complete ten-slot patch table")

page(6, "05  D: second body and safety checks")
table(
    ["Body 2 PC", "Original fusedpipe", "D"],
    [
        ["dea0", "LDTM.x32 R36, tmem[UR6]", "Same load; add write dependency B2"],
        ["deb0-ded0", "Other score loads; final load writes B1", "All instruction words unchanged"],
        ["dee0", "R2UR UR5, R0", "FMNMX3 R76,R36,R37,R38,!PT; wait B2"],
        ["def0", "NOP", "FMNMX3 R75,R39,R40,R41,!PT"],
        ["df00", "score_empty ARRIVE; wait B1", "Same PC and complete instruction word"],
        ["df10 / df20", "Original positions of the two max leaves", "R2UR UR5,R0 / NOP"],
    ],
    [0.85, 2.1, 2.85],
)
h("Dependency slots cannot be copied blindly between bodies")
p(
    "In the second body, B1 already protects the full score read. B0 also protects the descale load at 0xde40 and is waited on before its related overwrite at 0xdf70. D therefore uses retired B2. Its availability was checked through the entry waitmask at 0xde60 and the slow-wait return path, not just the adjacent instructions."
)
table(
    ["Review item", "Established boundary"],
    [
        [
            "Arithmetic",
            "The same source registers, !PT modifier and ordered max tree remain. Later combining nodes at a880 / dfd0 are unchanged.",
        ],
        [
            "Register conflicts",
            "The reordered instruction pairs have no RAW, WAR or WAW conflict; max results do not overwrite raw-score operands.",
        ],
        [
            "Delayed uniform moves",
            "UR9 / UR5 are still initialized before their first later consumer at b1b0 / e8f0.",
        ],
        [
            "Dependency retirement",
            "Review includes entry and slow-wait control flow, B1 in body 1, B2 in body 2 and the preserved B0 dependency.",
        ],
        [
            "Completion and metadata",
            "All 33 ARRIVE PCs and instruction words, including associated 0x3904 metadata records, remain unchanged.",
        ],
    ],
    [1.3, 4.5],
)
p(
    "Keeping ARRIVE at the same PC does not mean it executes in the same cycle. Moving useful work near that wait is the intended optimization. Code length, branch targets, register-allocation metadata and the memory-access set do not grow."
)
p(
    "Full machine-code differences, control-flow/liveness review and byte-equal outputs on matching FP8 inputs support the checked binary family. They are not a proof for arbitrary compiler output. A new layout must pass the complete runtime guard rather than a short instruction-pattern match.",
    "small",
)

page(7, "06  Matched incremental performance")
p(
    "B200, S=73,426, D=128. L00 / L24 identify layer 0 / 24 of step 24. Each row is a separate 20-round paired test. Speedup = median(control) / median(candidate). The aggregate is the equally weighted geometric mean of four case ratios; wins count rounds in which the candidate was faster."
)
h("Fusedpipe versus pre-change experimental VC: 1.013014x")
table(
    ["Case", "Control ms", "Fusedpipe ms", "Speedup", "Wins"],
    [
        [
            r["case_id"],
            f"{r['control_ms']:.6f}",
            f"{r['candidate_ms']:.6f}",
            f"{r['speedup_control_over_candidate']:.6f}x",
            f"{r['wins']}/20",
        ]
        for r in T["fusedpipe20"]["cases"]
    ],
    [1.1, 1.2, 1.3, 1.2, 0.7],
)
p(
    "An independent 12-round cohort measured 1.014248x with 48/48 wins; the 20-round cohort had 77/80 wins. The repeated aggregate speedup is approximately 1.30%-1.42% above 1x for these cases. Cohorts are not pooled, and the better batch is not selected per shape.",
    "small",
)
h("D versus already-fusedpipe: 1.002647x")
table(
    ["Case", "Fusedpipe ms", "With D ms", "Speedup", "Wins"],
    [
        [
            r["case_id"],
            f"{r['control_ms']:.6f}",
            f"{r['candidate_ms']:.6f}",
            f"{r['speedup_control_over_candidate']:.6f}x",
            f"{r['wins']}/20",
        ]
        for r in T["d20"]["cases"]
    ],
    [1.1, 1.3, 1.2, 1.2, 0.7],
)
p(
    "The first 12-round cohort measured 1.002887x and 37/48 wins; confirmation had 56/80 wins. The 0.26%-0.29% repeated aggregate signal is small. It does not establish a uniform benefit at every shape or time interval.",
    "small",
)
h("What the measurements do and do not attribute")
p(
    "CUDA events measure five replays of one attention kernel per timing sample, after two warm calls and a one-second warmup. Preparation and validation are outside timing. Included cases completed with clean ownership guards; clocks were not locked. Profiling duration is not used as performance data."
)
p(
    "Fusedpipe is measured as one combined source change; these results do not assign separate percentages to K prefetch and O preloading. D already has fusedpipe in its control. Ratios from the two cohorts must not be multiplied and presented as a newly measured combined speedup.",
    "small",
)

page(8, "07  BF16 context, resources and correctness")
h("Historical whole-VC comparison: approximately 2.137x")
p(
    "The pinned BF16 reference uses original BF16 QKV and its default scan order. VC uses preprocessed FP8 inputs, all descales and mid4. BF16 medians are from a separate 20-round cohort on 2 October; D medians are from its 3 October confirmation. This table is contextual arithmetic, not a newly paired BF16/D benchmark."
)
table(
    ["Case", "BF16 ms", "VC + D ms", "Historical ratio"],
    [
        [
            r["case_id"],
            f"{r['bf16_ms']:.6f}",
            f"{r['d_ms']:.6f}",
            f"{r['speedup_bf16_over_d']:.6f}x",
        ]
        for r in T["historical_bf16_vs_d"]["cases"]
    ],
    [1.1, 1.4, 1.4, 1.5],
)
p(
    "The geometric mean is 2.137153x. This includes the entire VC algorithm and earlier optimizations. It is neither the isolated D gain nor an end-to-end model speedup. The BF16 source revision is 2cae9072801704491b37f14037b4baa32c3958dc; provenance is retained in the public vendored-source manifest.",
    "small",
)
h("Registers: allocation and liveness are distinct")
table(
    ["Metric", "Fusedpipe -> D", "Interpretation"],
    [
        [
            "REG / STACK metadata",
            "168 / 8 -> 168 / 8",
            "Initial resource fields, not dynamic live values",
        ],
        ["Softmax USETMAXREG", "232 -> 232", "Role-specific allowance, not measured consumption"],
        [
            "Conservative CFG live peak",
            "205 -> 205",
            "Static analysis, not a hardware register-use counter",
        ],
        [
            "Static LDL / STL count",
            "3 / 2 -> 3 / 2",
            "Existing epilogue spills; no new hot-loop spill",
        ],
        [
            "Text bytes / instruction slots",
            "87,040 / 5,440 -> unchanged",
            "Existing slots are rearranged",
        ],
    ],
    [1.8, 1.65, 2.35],
)
p(
    "For the matched frozen pre-change VC and fusedpipe binaries, conservative static GPR peak rises from 169 to 205, with unchanged initial allocation and role allowance. The larger lifetime is consistent with both O halves being live, but the difference is not a dynamic measurement of 36 extra registers and does not by itself establish occupancy changes. SHARED=1024 excludes dynamic shared memory.",
    "small",
)
h("Correctness scope")
p(
    "Scheduling candidates are compared byte-for-byte with matching FP8 controls. The scan-window release validation passed ten cases and 120 complete-output comparisons. Equality for these changes does not imply FP8 equals BF16, exhaust all inputs, or establish model-quality equivalence."
)
source("S9", "src/vc_attn/source_manifest.json", 1, "Public baseline and implementation provenance")

page(9, "08  Applying D to the actual callable")
p(
    "D is applied per compiled function after cute.compile. The adapter exports a host object, identifies its embedded CUDA ELF, validates it, rewrites approved instruction words in memory, and binds the same ABI through BinaryExecutionEngine and TVM FFI. It retains the required objects and lifetimes. It does not install a global driver hook or mutate the original cubin or compiler on-disk cache; the API cache stores the wrapped callable."
)
code(
    "cute.compile(...)\n    -> export host object\n    -> locate the single embedded CUDA ELF\n    -> validate architecture, text family and metadata\n    -> rewrite 10 existing instruction slots\n    -> bind the same function ABI\n    -> cache the wrapped callable\n\nunsupported layout -> keep the original callable"
)
table(
    ["Guard", "Constraint"],
    [
        [
            "Architecture and adapter",
            "SM100 flags; ELF64 little-endian; CuTe DSL 4.6.0 and the supported FFI wrapper",
        ],
        [
            "Complete native code",
            "A single target text section, 87,040 bytes; complete normalized pre/post SHA-256 validation",
        ],
        [
            "Scan-window fields",
            "Only four proven immediate fields may vary; each must equal the caller-supplied window",
        ],
        [
            "Execution metadata",
            "Complete canonical fingerprint, with only explicitly defined naming normalization",
        ],
        [
            "Mutation boundaries",
            "All ten old words must match; post-patch length and target hash must match",
        ],
    ],
    [1.45, 4.35],
)
p(
    "The revision is sm100-score-two-body-window-v2. Instance metadata records the actual and normalized text hashes, window and applied status. A Python version label alone is not proof of the loaded code; external AOT loaders or bypass paths need their own loaded-artifact verification."
)
note(
    "Unsupported layouts retain the original callable. Errors from actual kernel execution are not silently swallowed as a patch mismatch. The export_to_c wrapper also revalidates the patch with its bound scan window."
)
source(
    "S10", "src/vc_attn/_sass_d.py", 142, "Window normalization and complete native-code validation"
)
source("S11", "src/vc_attn/_sass_runtime.py", 1, "Per-callable loading adapter")

page(10, "09  Eligibility and specialization boundaries")
p(
    "Sequence length alone is not an adequate eligibility rule. A parameter can select a different compiler branch, affect alignment or ABI, change resource allocation, or remain runtime data within one specialization. The complete code and metadata guard remains authoritative."
)
table(
    ["Factor", "How it may change the compiled layout"],
    [
        [
            "Scan window",
            "A compile-time scan constant. Common values change four immediate fields; 0/1 produce different layouts and currently fall back for D.",
        ],
        [
            "Sequence lengths, heads, batch",
            "Can cross Q-stage, packing, split-KV or stride/layout dispatch thresholds; within a branch they may remain runtime parameters.",
        ],
        [
            "D / Dv, tiles, stages, warps",
            "Change unrolled work, registers, TMEM/SMEM layouts and synchronization participants.",
        ],
        [
            "Dtype and descale presence/rank",
            "Select different arithmetic and loading paths; scaled and unscaled binaries are not interchangeable.",
        ],
        [
            "Mask, score modifier, skip, LSE",
            "Add control flow, output or side effects. GQA, paged KV and alternate quantization can select other specializations.",
        ],
        [
            "Architecture and compiler",
            "Target architecture, DSL, ptxas and compilation options may change instructions and dependency control.",
        ],
    ],
    [1.55, 4.25],
)
h("Current checked family")
p(
    "CPU native-family checks cover windows 2, 3, 4, 7, 8, 16, 31 and 1024. D falls back for 0, 1 and None. The fusedpipe source gate accepts non-None nonnegative windows, so 0/1 can use fusedpipe without D. The source path is limited to the checked SM100 schedule, E4M3 Q/K/V, all descales with rank-3 K descales, Q-stage 2, 128x128 tiles, D/Dv=128, inline correction and first-P64 handoff; skipping and custom score/mask modifiers are excluded."
)
p(
    "CPU mid8 shape evidence covers [32769,7,128], [73426,7,128], [73426,56,128] and [188214,7,128]. This is not their full Cartesian product, and compilation evidence alone is not a long-sequence performance result."
)
p(
    "The ten-case window release compared two public revisions. At windows 2/8/16/1024, enabling both fusedpipe and D is a combined change, not isolated D. At None and 4, both sides selected identical native text; their small timing differences are regression noise, not a new optimization gain.",
    "small",
)
source("S12", KERNEL, 3668, "Exact source-level fusedpipe gate")
source(
    "S13",
    "src/vc_attn/_kernels/v4/flash_attn/cute/interface.py",
    1283,
    "Runtime D attempt followed by strict binary validation",
)

page(11, "10  Longer-sequence extension")
if LONG is None:
    note(
        "No new long-sequence timing results are included in this edition. The previously verified results remain the main evidence. This section is reserved for a separately audited extension; no performance is inferred from a successful CPU compilation."
    )
    h("Required comparison boundary")
    p(
        "Report the actual S, H and D for each case and identify whether tensors are real model captures, synthetic inputs or head slices. Preserve a common prepared FP8 input, packed V and descales for both sides of each scheduling comparison. Never pad or repeat a private capture and describe it as a longer real inference."
    )
    h("Required evidence before a row is accepted")
    table(
        ["Check", "Required record"],
        [
            [
                "Implementation identity",
                "Source revision, native-code identity, configuration and whether D was actually applied",
            ],
            [
                "Correctness",
                "Finite output and complete byte equality against the same-input scheduling control",
            ],
            [
                "Timing",
                "Complete paired rounds, medians, paired wins, replay/warmup settings and kernel-only boundary",
            ],
            [
                "Execution validity",
                "Completed result and clean ownership guard; interrupted or contaminated runs excluded in full",
            ],
            [
                "Interpretation",
                "Isolated D, isolated fusedpipe or combined release comparison explicitly labeled",
            ],
        ],
        [1.4, 4.4],
    )
    p(
        "Longer sequences increase attention work approximately quadratically for this dense algorithm, but the measured scheduling gain need not grow with S. Tile count, head parallelism, rescale frequency and the compiled specialization can all affect the comparison. No gain is extrapolated here."
    )
else:
    assert LONG["schema"] == "vc-longseq-public-summary-v1"
    assert LONG["status"] == "complete" and LONG["accepted_case_count"] == 4
    assert not LONG["missing_shapes"]
    assert {tuple(r["shape"]) for r in LONG["rows"]} == {
        tuple(x) for x in LONG["expected_shape_matrix"]
    }
    pairs = [
        ("v4_over_fusedpipe", "v4", "fusedpipe"),
        ("fusedpipe_over_d", "fusedpipe", "d"),
        ("bf16_over_d", "bf16", "d"),
    ]
    for r in LONG["rows"]:
        assert (
            r["all_output_byte_checks_passed"]
            and r["single_kernel_each"]
            and r["bf16_same_run_and_paired"]
        )
        assert r["ownership_checks"] > 0 and r["rounds"] == LONG["settings"]["rounds"] == 12
        for label in ["v4", "fusedpipe", "d", "bf16"]:
            samples = r["raw_samples_ms"][label]
            assert len(samples) == 12 and all(math.isfinite(x) and x > 0 for x in samples)
            assert math.isclose(statistics.median(samples), r["median_ms"][label], rel_tol=1e-12)
        for key, a, b in pairs:
            cmp = r["comparisons"][key]
            assert math.isclose(
                cmp["ratio_of_medians"], r["median_ms"][a] / r["median_ms"][b], rel_tol=1e-12
            )
            assert cmp["wins"] == sum(
                x > y for x, y in zip(r["raw_samples_ms"][a], r["raw_samples_ms"][b])
            )
    seeds = sorted({r["input"]["seed"] for r in LONG["rows"]})
    assert len(seeds) == 1
    seed = seeds[0]
    p(
        f"<b>Fixed-seed synthetic inputs, separate from the real-capture results.</b> Independent CPU torch.randn BF16 Q/K/V use seeds {seed}, {seed + 1} and {seed + 2}. One noncausal sequence, D=128, equal Q/KV heads, all descales, mid4 and no skipping. These are newly generated long tensors, not padded or repeated model captures."
    )
    h("Four matched implementations: median latency in milliseconds")
    table(
        ["[S,H,D]", "Pre-fusedpipe VC", "Fusedpipe", "Fusedpipe + D", "BF16"],
        [
            [
                str(r["shape"]),
                *[f"{r['median_ms'][v]:.6f}" for v in ["v4", "fusedpipe", "d", "bf16"]],
            ]
            for r in LONG["rows"]
        ],
        [1.7, 1.3, 1.2, 1.35, 1.2],
    )
    h("Separate ratios, each computed within the same run")
    table(
        ["[S,H,D]", "Fusedpipe increment", "D increment", "Whole VC vs BF16"],
        [
            [
                str(r["shape"]),
                *[
                    f"{r['comparisons'][key]['ratio_of_medians']:.6f}x\n{r['comparisons'][key]['wins']}/12 wins"
                    for key, _, _ in pairs
                ],
            ]
            for r in LONG["rows"]
        ],
        [1.5, 1.45, 1.45, 1.5],
    )
    p(
        "Ratios are pre-fusedpipe/fusedpipe, fusedpipe/D and BF16/D, respectively. Wins count faster candidate rounds; a ratio of medians and a paired win count describe different aspects of the same samples. The BF16 numerator is freshly measured in the same run, not reused from the historical table.",
        "small",
    )
    gm = [
        math.exp(sum(math.log(r["comparisons"][key]["ratio_of_medians"]) for r in LONG["rows"]) / 4)
        for key, _, _ in pairs
    ]
    p(
        f"Equal-weight geometric means across these four synthetic cases: <b>{gm[0]:.6f}x</b> for fusedpipe, <b>{gm[1]:.6f}x</b> for D, and <b>{gm[2]:.6f}x</b> for complete VC + D versus BF16. These are kernel-only measurements, not an inference-pipeline speedup or a prediction for real-model inputs."
    )
    losses = [
        r for r in LONG["rows"] if r["comparisons"]["fusedpipe_over_d"]["ratio_of_medians"] < 1
    ]
    assert len(losses) == 1 and losses[0]["shape"] == [262144, 56, 128]
    loss = losses[0]
    slower = (loss["median_ms"]["d"] / loss["median_ms"]["fusedpipe"] - 1) * 100
    note(
        f"<b>D does not improve every case.</b> At [262144,56,128], D takes {slower:.3f}% longer by the ratio of medians, despite winning 8/12 paired rounds. These summaries can disagree because they aggregate samples differently. This is a tiny regression in one cohort, not evidence of a stable general regression or improvement."
    )
    p(
        "This is one fixed-seed synthetic sweep with no independent repeat cohort. In particular, the small D aggregate must not be described as a stable universal benefit. The real-capture cohort and its historical BF16 ratio remain separate.",
        "small",
    )
    page(12, "10  Long-sequence method and evidence")
    table(
        ["Item", "Recorded boundary"],
        [
            [
                "Input generation",
                "Independent CPU torch.randn BF16 Q/K/V; Q/K/V seeds 20261004 / 20261005 / 20261006. Equal Q/KV head counts; S=188214 or 262144 and H=7 or 56.",
            ],
            [
                "Quantized configuration",
                "All Q/K/V descales supplied, ExpCast, packed V, inline O rescale, Tensor Core denominator, mid4 and no skipping.",
            ],
            [
                "Timing",
                "12 complete paired rounds per case; five kernel replays per sample, two warm calls, one-second warmup. Each implementation occupies each order position three times.",
            ],
            [
                "Excluded work",
                "QKV generation and quantization, V packing, allocation, compilation, validation, profiling, model execution and communication.",
            ],
            [
                "Correctness and ownership",
                "All four accepted cases passed complete scheduling-output byte comparisons and single-kernel checks. Interruptions and foreign-process contamination invalidate an entire attempt.",
            ],
            [
                "Clock and repetition limits",
                "Clocks were not locked. One accepted cohort per shape; no independent repeat cohort.",
            ],
        ],
        [1.4, 4.4],
    )
    h("Exactly which implementation was tested")
    p(
        "The four routes are a matched pre-fusedpipe VC control, fusedpipe, its exact D binary derivative, and a pinned BF16 reference freshly timed in the same run. The controls are frozen experimental kernels, not unmodified current public checkouts. The historical H7 dispatch adaptation is shared by the relevant experimental controls."
    )
    p(
        "Only three attention kernel records were needed for BF16, pre-fusedpipe VC and fusedpipe. The four long shapes share the respective native functions with different runtime dimensions; D derives from the fusedpipe artifact. The fusedpipe text and D guard match the canonical 87,040-byte family. Correct execution at S=262144 is new evidence from this typed-AOT replay path, not proof that every public-API configuration is supported."
    )
    p(
        "CPU artifact verification, isolated host/internal symbols and invocation of a single-kernel callable identify the tested D derivative. This sweep has no new device-code dump. Complete FP8 scheduling-output equality is required; equality to BF16 and model-quality equivalence are not claimed.",
        "small",
    )
    h("Public supplementary results")
    code("VC-Attention-Long-Sequence-Results.json")
    p(
        "The companion JSON contains every accepted raw paired timing sample, generation seeds and input hashes, configurations, per-case medians, wins, ratios, order-position counts and artifact hashes. It includes no private model input paths or infrastructure hostnames. Rejected attempts are excluded from every table and aggregate; their count is recorded."
    )

page(13, "11  Public reproduction workflow")
p(
    "The public tools in tools/report_reproduction rebuild isolated controls from this checkout. They create a private source tree and CPU AOT artifacts; the installed inference package and its defaults are unchanged."
)
table(
    ["Route", "Construction and identity"],
    [
        [
            "pre_fusedpipe",
            "Set only use_fusedpipe=False in the staged v4 source. Keep mid4, all descales and the shared B200 packing gate.",
        ],
        [
            "fusedpipe",
            "Clone the same v4 source under an isolated namespace; keep the fusedpipe schedule enabled.",
        ],
        ["bf16", "Use the pinned baseline snapshot with original BF16 inputs."],
        [
            "d (explicit opt-in)",
            "Derive from the exact fusedpipe object. Check canonical text/metadata and isolate host symbols; fail if the fingerprint differs.",
        ],
    ],
    [1.2, 4.8],
)
p(
    "The report control pre_fusedpipe is not the public CLI backend vc_v4: that historical CLI name selects the original scan, whereas both report FP8 controls use mid4. Native equivalence is identified by artifacts, not labels.",
    "small",
)
code(
    "python tools/report_reproduction/build.py --output build/report\npython tools/report_reproduction/benchmark.py \\\n  --build build/report --shape 32769x7x128 \\\n  --output results/report-smoke.json\npython tools/report_reproduction/summarize.py \\\n  results/report-smoke.json",
    7.8,
)
h("Long-sequence matrix and D")
p(
    "Repeat the runner for S=188214/262144 and H=7/56, D=128. Defaults are seeds 20261004/5/6, 12 balanced rounds, 5 replays, 2 warm calls and 1 second of warmup. Pass --include-d to both build and benchmark only for the experimental fourth route. A D guard rejection is an error, never a relabeled fusedpipe fallback."
)
h("Exact timer boundary")
p(
    "Input generation, quantization and V packing happen before capture. The runner captures the native AOT call, checks direct/graph output bytes and profiles exactly one attention kernel outside timing. Complete post-timing byte checks, artifact hashes and raw paired samples are retained. The ordinary vc-attn-bench scope includes V packing and remains a separate operator benchmark."
)
p(
    "See docs/reports/reproduction.md for complete commands, prerequisites, expected output and failure handling. Fresh measurements form a new cohort; the published historical tables are unchanged.",
    "small",
)

page(14, "12  Environment and validation boundaries")
h("Recovered historical long-sequence environment")
table(
    ["Component", "Recorded value"],
    [
        ["GPU / driver", "NVIDIA B200 / SM100; driver 590.48.01"],
        ["Python / PyTorch", "Python 3.12.3; PyTorch 2.11.0+cu130; CUDA runtime 13.0"],
        ["DSL / Triton / quack", "CuTe DSL 4.6.0; Triton 3.6.0; quack 0.6.1"],
        ["CUDA Python / linker", "cuda-python 13.0.3; GCC 13.3.0; GNU objcopy 2.42"],
        [
            "Clock policy",
            "Clocks were not locked. Per-case start/end clock, pstate and power snapshots are now public.",
        ],
    ],
    [1.7, 4.3],
)
p(
    "historical-environment.json recovers these fields from the original accepted records and retains their hashes. The historical ptxas binary identity and complete compiler environment were not recorded. They remain explicitly unknown; current tool versions are not substituted for historical facts."
)
h("What a new build records")
p(
    "manifest.json records package versions, available compiler/linker identities, discovered bundled ptxas candidates, compile options, runtime library hashes, source transformations, ABI/cache metadata, artifact hashes and explicit D patch proof. Candidate compiler discovery does not itself prove which binary the DSL invoked. Rebuild and run in the same pinned environment."
)
h("Correctness, numerical approximation and ownership")
p(
    "FP8 scheduling controls must be finite and byte-equal; direct and graph calls must match, including after timing. BF16 is a timing reference with its own finite-output and direct/graph checks; FP8 equality to BF16 is not claimed. The runner checks foreign CUDA PIDs around preparation and every timed sample; reserve the GPU externally. Shared-GPU opt-in results are explicitly marked non-isolated."
)
h("Current release and experimental scope")
p(
    "The default API uses mid4 and eligible source-level fusedpipe, with D disabled. D remains an experimental offline replay option and may reject a different compiler family. This report does not establish general real-Uint8 FFI ABI compatibility, model-quality equivalence, backward support or end-to-end speedup. Current reproduction validation is recorded separately in reproduction-validation.json."
)
p(
    "Private captured-input tensors are not distributed. The synthetic long-sequence matrix can be regenerated and input-hash checked. Its original single-seed cohort and small D effect do not establish a universal benefit.",
    "small",
)

page(15, "Appendix  Canonical patch and fingerprints")
p(
    "Canonical mid4, text-relative PCs. Each OLD/NEW value is a complete 128-bit instruction word, stored little-endian in ELF. All unlisted slots remain unchanged. The four scan immediate fields are normalized only for family validation; the caller's actual scan window is preserved.",
    "small",
)
tree = ast.parse((REPO / "src/vc_attn/_sass_d.py").read_text())
words = next(
    ast.literal_eval(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    and any(isinstance(x, ast.Name) and x.id == "_WORDS" for x in n.targets)
)
for pc, old, new in words:
    code(f"{pc:04x}  OLD {old}\n      NEW {new}", 7.5)
h("Normalized SHA-256 fingerprints")
code(
    "before  3f8c9a6f08115780f331d674a3effa774\n        10d88bd2fe9ee91c9fffeed74367591\nafter   040efd30afd2544024a7a36046ad8382d\n        4687d2d3d6ce6da9fee4d60b0b6755f\nmeta    9ab3d51a5555fc0a52621a3623fe36dd\n        d0fa20d1a1802bfd5765ac6e23946e42",
    7.5,
)
p(
    "Scan immediates: PCs 1740, 5fc0, 7ca0 and a570; only bits 32-63 of each word participate in window normalization. These PCs identify the canonical binary family, not universal addresses for other compiler versions.",
    "small",
)


def furniture(c, doc):
    c.saveState()
    c.setStrokeColor(RULE)
    c.setLineWidth(0.5)
    c.line(44, H - 40, W - 44, H - 40)
    c.setFont("Helvetica", 7.7)
    c.setFillColor(MUTED)
    c.drawString(44, H - 31, "VC ATTENTION / KERNEL ENGINEERING")
    c.drawRightString(W - 44, H - 31, "B200 / SM100")
    c.line(44, 38, W - 44, 38)
    c.drawString(44, 25, "Reproduction edition | 2026-10-04 | historical cbb2da8")
    c.drawRightString(W - 44, 25, f"{doc.page:02d}")
    c.restoreState()


SCRATCH.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    rightMargin=44,
    leftMargin=44,
    topMargin=55,
    bottomMargin=51,
    title="VC Attention: Fusedpipe and D Scheduling",
    author="VC Attention engineering analysis",
    subject="B200 scheduling, matched kernel benchmarks and guarded runtime integration",
)
doc.build(story, onFirstPage=furniture, onLaterPages=furniture)
reader = PdfReader(OUT)
text = "\n\n".join(f"PAGE {i + 1}\n" + p.extract_text() for i, p in enumerate(reader.pages))
(SCRATCH / "extracted.txt").write_text(text)
forbidden = [r"/" + r"Users/", r"/" + r"mnt/", r"work/" + r"vc-"]
assert not any(re.search(pattern, text, re.I) for pattern in forbidden), (
    "Private/irrelevant content in PDF"
)
expected_pages = 15
assert len(reader.pages) == expected_pages, f"Unexpected overflow: {len(reader.pages)} pages"
assert all(
    f"TECHNICAL REPORT / {i:02d}" in reader.pages[i - 1].extract_text()
    for i in range(1, expected_pages + 1)
), "Page split/overflow"
meta = {
    "pdf": str(OUT.relative_to(ROOT)),
    "pages": len(reader.pages),
    "bytes": OUT.stat().st_size,
    "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    "long_sequence_included": LONG is not None,
    "source_revision": SHA,
    "numeric_input_sha256": hashlib.sha256(
        (WORK / "captured-summary.json").read_bytes()
    ).hexdigest(),
    "long_sequence_sha256": hashlib.sha256(
        (WORK / "VC-Attention-Long-Sequence-Results.json").read_bytes()
    ).hexdigest()
    if LONG is not None
    else None,
    "privacy_text_check": "pass",
    "render_review": "pending",
}
(SCRATCH / "build-audit.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
