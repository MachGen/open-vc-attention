"""Generate every computed number, table and chart in the technical report.

Run by tools/reports/build.py inside a temporary build directory; outputs go to
./fig, ./tables and ./tables/numbers.json relative to the working directory.

Measured values are read from the committed benchmark records in benchmarks/results/b200/
(comparison.json: BF16, VC and Open-VC; repair.json: V residual repair budgets), produced by
open-vc-attn-bench. Everything else is computed here (FLOP model, utilization, simulations).
"""

import json
import math
import os
from pathlib import Path

import matplotlib
import ml_dtypes
import numpy as np

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT_FIG, OUT_TAB = "fig", "tables"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
E4, F4 = ml_dtypes.float8_e4m3fn, ml_dtypes.float4_e2m1fn
LOG2E = 1 / math.log(2)
D = 128
PEAK_BF16, PEAK_FP8 = 2.25e15, 4.5e15  # B200 dense
FA4_PUB = 1.605e15  # FA4 blog, B200 BF16
MUFU_GB200, MUFU_GB300 = 4943e9, 10024e9  # exp2 FP32 op/s, NVIDIA blog


def flops(S, H, Sk=None):
    return 4 * S * (Sk or S) * H * D


# ---------------------------------------------------------------- measured data
RESULTS = Path(__file__).resolve().parents[2] / "benchmarks/results/b200"
CMP = json.loads((RESULTS / "comparison.json").read_text())
REP = json.loads((RESULTS / "repair.json").read_text())
ROWS = CMP["rows"]
S0 = ROWS[0]["shape"][0]
BK = ("bf16", "vc", "open-vc")
# V-Smooth's k-means grouping runs on the first quarter of denoising steps (VC-Attention's
# protocol); timed calls reuse the grouping, so it is amortized separately.
GROUP_FRACTION = CMP["v_smooth"]["grouping_step_fraction"]


def vc_group_ms(r):
    """Extra time of a VC complete call that groups over one that reuses the grouping."""
    timed = r["vc_complete_call_events_ms"]
    return timed["fresh_grouping"]["median_ms"] - timed["reused_grouping"]["median_ms"]


def vc_avg_ms(r):
    """Complete call averaged over denoising steps: the grouping increment on a quarter."""
    return r["complete_call"]["median_ms"]["vc"] + GROUP_FRACTION * vc_group_ms(r)


N = {}  # LaTeX macros


def pf(S, H, ms):
    return flops(S, H) / (ms * 1e-3) / 1e15


def k(S):
    return f"{S:,}".replace(",", "{,}")


def rng2(vals, fmt="{:.2f}"):
    return fmt.format(min(vals)), fmt.format(max(vals))


# ---------------------------------------------------------------- tables: performance
with open(f"{OUT_TAB}/results.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}llrrrrr@{}}\\toprule\n"
        "Heads & Scope & BF16 (ms) & VC (ms) & Open-VC (ms) & VC speedup & Open-VC speedup\\\\\\midrule\n"
    )
    for i, r in enumerate(ROWS):
        H = r["shape"][1]
        for j, (key, label) in enumerate(
            (("attention", "attention kernel"), ("complete_call", "complete call"))
        ):
            m, sp = r[key]["median_ms"], r[key]["speedup_vs_bf16"]
            f.write(
                f"{H if j == 0 else ''} & {label} & {m['bf16']:.3f} & {m['vc']:.3f} & {m['open-vc']:.3f} & "
                f"{sp['vc']:.2f}$\\times$ & \\textbf{{{sp['open-vc']:.2f}}}$\\times$\\\\\n"
            )
        if i < len(ROWS) - 1:
            f.write("\\midrule\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(f"{OUT_TAB}/throughput.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}lrrrrr@{}}\\toprule\n"
        "Heads & BF16 (PF/s) & VC (PF/s) & Open-VC (PF/s) & Open-VC FP8 util. & Open-VC / FA4 published\\\\\\midrule\n"
    )
    for r in ROWS:
        H = r["shape"][1]
        m = r["attention"]["median_ms"]
        p = {b: pf(S0, H, m[b]) for b in BK}
        f.write(
            f"{H} & {p['bf16']:.2f} & {p['vc']:.2f} & {p['open-vc']:.2f} & {p['open-vc'] / 4.5 * 100:.0f}\\% & "
            f"{p['open-vc'] / 1.605:.2f}$\\times$\\\\\n"
        )
    f.write("\\bottomrule\\end{tabular}\n")
with open(f"{OUT_TAB}/accuracy.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n"
        "& \\multicolumn{3}{c}{VC: ExpCast + V-Smooth} & \\multicolumn{3}{c}{Open-VC}\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(l){5-7} Heads & Rel.\\ $L_2$ & RMSE & Max abs. & Rel.\\ $L_2$ & RMSE & Max abs.\\\\\\midrule\n"
    )
    for r in ROWS:
        a = r["accuracy_vs_bf16"]
        f.write(
            f"{r['shape'][1]} & "
            + " & ".join(
                f"{a[b]['relative_l2'] * 100:.3f}\\% & {a[b]['rmse']:.4f} & {a[b]['max_abs']:.2f}"
                for b in ("vc", "open-vc")
            )
            + "\\\\\n"
        )
    f.write("\\bottomrule\\end{tabular}\n")


def col(key, b):
    return [r[key]["speedup_vs_bf16"][b] for r in ROWS]


def pfs(b):
    return [pf(S0, r["shape"][1], r["attention"]["median_ms"][b]) for r in ROWS]


def rel(b):
    return [r["accuracy_vs_bf16"][b]["relative_l2"] * 100 for r in ROWS]


ovs_k = [r["attention"]["median_ms"]["vc"] / r["attention"]["median_ms"]["open-vc"] for r in ROWS]
ovs_c = [
    r["complete_call"]["median_ms"]["vc"] / r["complete_call"]["median_ms"]["open-vc"] for r in ROWS
]
vc_avg = [r["complete_call"]["median_ms"]["bf16"] / vc_avg_ms(r) for r in ROWS]
group_pct = [vc_group_ms(r) / r["attention"]["median_ms"]["vc"] * 100 for r in ROWS]
h56 = next(r for r in ROWS if r["shape"][1] == 56)
prep = h56["complete_call"]["median_ms"]["open-vc"] - h56["attention"]["median_ms"]["open-vc"]
N.update(sTok=k(S0))
for name, vals, fmt in (
    ("spK", col("attention", "open-vc"), "{:.2f}"),
    ("spC", col("complete_call", "open-vc"), "{:.2f}"),
    ("vcK", col("attention", "vc"), "{:.2f}"),
    ("vcC", col("complete_call", "vc"), "{:.2f}"),
    ("vcAvg", vc_avg, "{:.2f}"),
    ("vcGroupPct", group_pct, "{:.0f}"),
    ("vcGroupAvgPct", [g * GROUP_FRACTION for g in group_pct], "{:.1f}"),
    ("ovK", ovs_k, "{:.2f}"),
    ("ovC", ovs_c, "{:.2f}"),
    ("pf", pfs("open-vc"), "{:.2f}"),
    ("vcPf", pfs("vc"), "{:.2f}"),
    ("bfPf", pfs("bf16"), "{:.2f}"),
    ("util", [p / 4.5 * 100 for p in pfs("open-vc")], "{:.0f}"),
    ("bfUtil", [p / 2.25 * 100 for p in pfs("bf16")], "{:.0f}"),
    ("vsFa", [p / 1.605 for p in pfs("open-vc")], "{:.2f}"),
    ("belowFa", [(1 - p / 1.605) * 100 for p in pfs("bf16")], "{:.0f}"),
    ("rel", rel("open-vc"), "{:.2f}"),
    ("vcRel", rel("vc"), "{:.2f}"),
):
    lo, hi = rng2(vals, fmt)
    N[name + "Lo"], N[name + "Hi"] = lo, hi
vs_gain = [
    (r["accuracy_vs_bf16"]["open-vc"]["relative_l2"] - r["accuracy_vs_bf16"]["vc"]["relative_l2"])
    / r["accuracy_vs_bf16"]["open-vc"]["relative_l2"]
    * 100
    for r in ROWS
]
N["vsGainLo"], N["vsGainHi"] = rng2(vs_gain, "{:.1f}")
vc_prep = h56["complete_call"]["median_ms"]["vc"] - h56["attention"]["median_ms"]["vc"]
N.update(
    vcGroupMs=f"{vc_group_ms(h56):.1f}",
    vcGroupPct=f"{vc_group_ms(h56) / h56['attention']['median_ms']['vc'] * 100:.0f}",
    vcGroupAvgPct=f"{vc_group_ms(h56) / h56['attention']['median_ms']['vc'] * 100 * GROUP_FRACTION:.1f}",
    vcPrepMs=f"{vc_prep:.2f}",
    groupStepPct=f"{GROUP_FRACTION * 100:.0f}",
)
N.update(
    prepMs=f"{prep:.2f}",
    prepShare=f"{prep / h56['complete_call']['median_ms']['open-vc'] * 100:.1f}",
    relFiftySix=f"{h56['accuracy_vs_bf16']['open-vc']['relative_l2'] * 100:.2f}",
    vcRelFiftySix=f"{h56['accuracy_vs_bf16']['vc']['relative_l2'] * 100:.2f}",
)
N.update(
    mufuCeil=f"{MUFU_GB200 * 4 * D / 1e15:.2f}",
    mufuCeilBthree=f"{MUFU_GB300 * 4 * D / 1e15:.1f}",
    bfNeedExp=f"{PEAK_BF16 / (4 * D) / 1e12:.1f}",
    fpNeedExp=f"{PEAK_FP8 / (4 * D) / 1e12:.1f}",
    bfMufuFrac=f"{PEAK_BF16 / (4 * D) / MUFU_GB200 * 100:.0f}",
    smThreeH=k(math.ceil(1048576 / 7)),
)


# ---------------------------------------------------------------- ExpCast numerics
def dec(codes):
    return np.asarray(codes, np.uint8).view(E4).astype(np.float64)


def expcast(u):
    return np.clip(np.rint(8 * u * LOG2E + 119.65), 0, 120).astype(np.uint8)


def rne(u):
    return (256 * np.exp(u)).astype(E4).view(np.uint8)


u = np.linspace(-14, 0, 2_000_001)
ce, cr = expcast(u), rne(u)
vt = 256 * np.exp(u)
ve, vr = dec(ce), dec(cr)
nm = ce >= 8
re_e, re_r = ve[nm] / vt[nm] - 1, vr[nm] / vt[nm] - 1
m97 = u >= -9.7
N.update(
    byteMatch=f"{np.mean(ce[m97] == cr[m97]) * 100:.1f}",
    ecMin=f"{re_e.min() * 100:.1f}",
    ecMax=f"{re_e.max() * 100:.1f}",
    ecRms=f"{np.sqrt(np.mean(re_e**2)) * 100:.2f}",
    rnMax=f"{np.abs(re_r).max() * 100:.1f}",
    rnRms=f"{np.sqrt(np.mean(re_r**2)) * 100:.2f}",
    ecCut=f"{(0.5 - 119.65) / (8 * LOG2E):.2f}",
    rnCut=f"{math.log(2**-10 / 256):.2f}",
    ecCutProb=f"{math.exp((0.5 - 119.65) / (8 * LOG2E)) * 1e5:.1f}",
    clipDelta=f"{(120.5 - 119.65) / (8 * LOG2E):.3f}",
    overDelta=f"{(126.5 - 119.65) / (8 * LOG2E):.2f}",
)


# worked examples
def ex(uv):
    c1 = int(expcast(np.array([uv]))[0])
    c2 = int(rne(np.array([uv]))[0])
    return (
        256 * math.exp(uv),
        8 * uv * LOG2E + 119.65,
        c1,
        float(dec([c1])[0]),
        c2,
        float(dec([c2])[0]),
    )


for name, uv in (("One", -1.0), ("Two", -0.17)):
    t, raw, c1, v1, c2, v2 = ex(uv)
    N.update(
        {
            f"ex{name}U": f"{uv:g}",
            f"ex{name}True": f"{t:.2f}",
            f"ex{name}Raw": f"{raw:.2f}",
            f"ex{name}Code": str(c1),
            f"ex{name}Val": f"{v1:g}",
            f"ex{name}RneCode": str(c2),
            f"ex{name}RneVal": f"{v2:g}",
        }
    )
# subnormal decode ratios
sub = {cc: float(dec([cc])[0]) / 2 ** (8 + (cc - 119.65) / 8) for cc in (1, 7)}
N.update(subLo=f"{sub[1]:.2f}", subHi=f"{sub[7]:.2f}")

# tail-mass toy model: mass of the exact softmax below each flush threshold (5 seeds x 64 rows)
NK, rows_t, seeds = 75600, 64, 5
u_rne0 = math.log(2**-10 / 256)
u_ec0 = (0.5 - 119.65) / (8 * LOG2E)
tail = []
for sigma in (1, 2, 3, 4):
    me_l, mr_l = [], []
    for sd in range(seeds):
        rng = np.random.default_rng(100 + sd)
        me = mr = 0.0
        for _ in range(rows_t):
            s_ = sigma * rng.standard_normal(NK)
            uu = s_ - s_.max()
            p = np.exp(uu)
            P = p / p.sum()
            me += P[uu < u_ec0].sum() / rows_t
            mr += P[uu < u_rne0].sum() / rows_t
        me_l.append(me * 100)
        mr_l.append(mr * 100)
    tail.append((sigma, np.mean(me_l), min(me_l), max(me_l), np.mean(mr_l), min(mr_l), max(mr_l)))
with open(f"{OUT_TAB}/tail.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}ccc@{}}\\toprule\n"
        "Score spread $\\sigma$ (nats) & ExpCast: mass flushed to zero & exp + cast: mass flushed to zero\\\\\\midrule\n"
    )
    for sg, me, mel, meh, mr, mrl, mrh in tail:
        f.write(
            f"{sg} & {me:.2f}\\% \\;({mel:.2f}--{meh:.2f}) & {mr:.3f}\\% \\;({mrl:.3f}--{mrh:.3f})\\\\\n"
        )
    f.write("\\bottomrule\\end{tabular}\n")
N.update(
    tailThreeEc=f"{tail[2][1]:.2f}",
    tailThreeEcLo=f"{tail[2][2]:.2f}",
    tailThreeEcHi=f"{tail[2][3]:.2f}",
    tailThreeRne=f"{tail[2][4]:.2f}",
)


# ---------------------------------------------------------------- V-Smooth model
def fp8_pc(V):
    s = np.abs(V).max(0, keepdims=True) / 448
    return (V / s).astype(E4).astype(np.float64) * s


def nvfp4(V, blk=16):
    n, d = V.shape
    g = np.abs(V).max() / (6 * 448)
    Vb = V.reshape(n // blk, blk, d)
    sb = (np.abs(Vb).max(1, keepdims=True) / 6 / g).astype(E4).astype(np.float64) * g
    sb[sb == 0] = 1
    return ((Vb / sb).astype(F4).astype(np.float64) * sb).reshape(n, d)


rng = np.random.default_rng(1)
Ns, Bs = 32768, 128
vs_rows = []
for frac in (0.08, 0.36, 0.80):
    lab = rng.integers(0, 64, Ns)
    C = rng.standard_normal((64, D)) * math.sqrt(frac / (1 - frac))
    V = (C[lab] + rng.standard_normal((Ns, D)))[np.argsort(lab, kind="stable")]
    mu = V.reshape(Ns // Bs, Bs, D).mean(1, keepdims=True)
    R = (V.reshape(Ns // Bs, Bs, D) - mu).reshape(Ns, D)
    left = np.sum(R**2) / np.sum(V**2)
    r8 = np.sum((fp8_pc(R) - R) ** 2) / np.sum((fp8_pc(V) - V) ** 2)
    r4 = np.sum((nvfp4(R) - R) ** 2) / np.sum((nvfp4(V) - V) ** 2)
    vs_rows.append((frac, left, r8, r4))
Vg = rng.standard_normal((Ns, D))
N.update(
    fpEightElem=f"{np.sqrt(np.sum((fp8_pc(Vg) - Vg) ** 2) / np.sum(Vg**2)) * 100:.1f}",
    fpFourElem=f"{np.sqrt(np.sum((nvfp4(Vg) - Vg) ** 2) / np.sum(Vg**2)) * 100:.1f}",
)
with open(f"{OUT_TAB}/vsmooth_sim.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}cccc@{}}\\toprule\n"
        "Mean-component share of $V$ energy & Energy left after demeaning & FP8 error ratio & NVFP4 error ratio\\\\\\midrule\n"
    )
    for frac, left, r8, r4 in vs_rows:
        f.write(f"{frac:.2f} & {left:.3f} & {r8:.3f} & {r4:.3f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
base = 1.81 / (1 - 0.08)
N.update(
    predCube=f"{(1 - 0.12) / (1 - 0.08):.3f}",
    measCube=f"{1.74 / 1.81:.3f}",
    predKm=f"{(1 - 0.36) / (1 - 0.08):.3f}",
    measKm=f"{1.28 / 1.81:.3f}",
    impliedBase=f"{base:.2f}",
    fullVs=f"{(1 - 1.28 / base) * 100:.0f}",
)
vsh = []
for share in (0.82, 0.60, 0.40):
    mse = share * (1 - 1.28 / base)
    ours = float(N["relFiftySix"])
    vsh.append(
        (
            share,
            mse * 100,
            (1 - math.sqrt(1 - mse)) * 100,
            -20 * math.log10(math.sqrt(1 - mse)),
            ours * math.sqrt(1 - mse),
        )
    )
with open(f"{OUT_TAB}/vsmooth_est.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}ccccc@{}}\\toprule\n"
        f"V share of output MSE & Output MSE change & Rel.\\ $L_2$ change & SQNR gain & Our {N['relFiftySix']}\\% would become\\\\\\midrule\n"
    )
    for share, mse, rl, db, new in vsh:
        f.write(
            f"{share * 100:.0f}\\% & $-${mse:.1f}\\% & $-${rl:.1f}\\% & {db:.1f} dB & {new:.2f}\\%\\\\\n"
        )
    f.write("\\bottomrule\\end{tabular}\n")
N.update(
    vsRlHi=f"{vsh[0][2]:.1f}",
    vsRlLo=f"{vsh[2][2]:.1f}",
    vsRlMid=f"{vsh[1][2]:.1f}",
    vsNewLo=f"{vsh[1][4]:.1f}",
    vsNewHi=f"{vsh[2][4]:.1f}",
)

# ---------------------------------------------------------------- repair table
REPAIR = [
    (
        b["budget"] * 100,
        b["relative_l2_vs_bf16"] * 100,
        b["attention_median_ms"],
        b["complete_call_median_ms"],
    )
    for b in REP["budgets"]
]
with open(f"{OUT_TAB}/repair.tex", "w") as f:
    f.write(
        "\\begin{tabular}{@{}rrrrrr@{}}\\toprule\n"
        "Budget & Rel.\\ $L_2$ vs BF16 & Attention (ms) & Total (ms) & Error reduction & Total-time increase\\\\\\midrule\n"
    )
    for b, e, a_, t in REPAIR:
        f.write(
            f"{b:g}\\% & {e:.3f}\\% & {a_:.2f} & {t:.2f} & {(REPAIR[0][1] - e) / REPAIR[0][1] * 100:.2f}\\% & {(t - REPAIR[0][3]) / REPAIR[0][3] * 100:.2f}\\%\\\\\n"
        )
    f.write("\\bottomrule\\end{tabular}\n")


def rep_at(pct):
    b, e, a_, t = next(x for x in REPAIR if abs(x[0] - pct) < 1e-9)
    return (REPAIR[0][1] - e) / REPAIR[0][1] * 100, (t - REPAIR[0][3]) / REPAIR[0][3] * 100, e


N.update(
    repBaseErr=f"{REPAIR[0][1]:.3f}",
    repHalfErr=f"{rep_at(0.5)[2]:.3f}",
    repHalfGain=f"{rep_at(0.5)[0]:.2f}",
    repHalfTime=f"{rep_at(0.5)[1]:.2f}",
    repEightErr=f"{rep_at(8)[2]:.3f}",
    repEightGain=f"{rep_at(8)[0]:.2f}",
    repEightTime=f"{rep_at(8)[1]:.2f}",
    repBf=f"{h56['attention']['median_ms']['bf16']:.2f}",
)
sel = round(0.005 * S0)
N.update(
    repSel=str(sel),
    repRows=str(128 * math.ceil(sel / 128)),
    repOver=f"{128 * math.ceil(sel / 128) / S0 * 100:.3f}",
)
N.update(wanTok=k(21 * 30 * 52), wanSevenTwenty=k(21 * 45 * 80))

with open(f"{OUT_TAB}/numbers.tex", "w") as f:
    for kk, vv in N.items():
        f.write(f"\\newcommand{{\\n{kk}}}{{{vv}}}\n")
with open(f"{OUT_TAB}/numbers.json", "w") as f:
    json.dump(N, f, indent=1)

# ================================================================ charts
INK, INK2, MUTED, GRID, AXIS, SURF = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#ffffff",
)
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.titlesize": 8.5,
        "axes.titlecolor": INK,
        "legend.frameon": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    }
)


def style(ax, grid_axis="y"):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def bar(ax, x, h, w, color):
    ax.add_patch(Rectangle((x - w / 2, 0), w, h, lw=0, fc=color, zorder=2))


# ---- Figure: throughput, three backends per head count, one panel per scope
TEAL = "#17808e"
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
w = 0.26
for ax, (key, title) in zip(
    axes,
    (("attention", "Attention kernel"), ("complete_call", "Complete call (incl. FP8 preparation)")),
):
    style(ax)
    ax.axhline(MUFU_GB200 * 4 * D / 1e15, color=INK2, lw=0.9, zorder=1)
    ax.axhline(1.605, color=MUTED, lw=0.9, ls=(0, (2, 2)), zorder=1)
    for i, r in enumerate(ROWS):
        H = r["shape"][1]
        m = r[key]["median_ms"]
        for j, (b, color) in enumerate((("bf16", ORANGE), ("vc", TEAL), ("open-vc", BLUE))):
            x = i + (j - 1) * (w + 0.02)
            bar(ax, x, pf(S0, H, m[b]), w, color)
        top = pf(S0, H, m["open-vc"])
        ax.text(
            i + w + 0.02,
            top + 0.06,
            f"{r[key]['speedup_vs_bf16']['open-vc']:.2f}×",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=7.0,
        )
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels([f"H = {r['shape'][1]}" for r in ROWS], fontsize=7)
    ax.set_xlim(-0.6, len(ROWS) - 0.4)
    ax.set_ylim(0, 3.2)
    ax.set_title(title, loc="left", pad=5, fontsize=7.8)
axes[0].set_ylabel("Throughput (PFLOP/s)")
axes[0].set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

handles = [plt.Rectangle((0, 0), 1, 1, fc=c, lw=0) for c in (ORANGE, TEAL, BLUE)] + [
    Line2D([0], [0], color=INK2, lw=0.9),
    Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (2, 2))),
]
fig.legend(
    handles,
    [
        "BF16 (upstream FA4)",
        "VC: ExpCast + V-Smooth",
        "Open-VC",
        "MUFU-only exp ceiling (≈2.53)",
        "FA4 published BF16 (1.61)",
    ],
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 1.0),
    fontsize=6.6,
    handlelength=1.4,
    columnspacing=1.1,
)
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.2)
fig.savefig(f"{OUT_FIG}/throughput.pdf")
plt.close(fig)

# ---- Figure: ExpCast code map and error
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
uu = np.linspace(-13.5, 0, 40001)
style(a1, "both")
a1.semilogy(uu, 256 * np.exp(uu), color=MUTED, lw=1.0, label="exact $256\\,e^{u}$")
v_r = dec(rne(uu))
v_e = dec(expcast(uu))
a1.semilogy(
    uu[v_r > 0], v_r[v_r > 0], color=ORANGE, lw=1.4, drawstyle="steps-mid", label="exp + RNE cast"
)
a1.semilogy(uu[v_e > 0], v_e[v_e > 0], color=BLUE, lw=1.4, drawstyle="steps-mid", label="ExpCast")
a1.axvline(float(N["ecCut"]), color=AXIS, lw=0.8)
a1.axvline(float(N["rnCut"]), color=AXIS, lw=0.8)
a1.text(
    float(N["ecCut"]) + 0.2,
    1.3e-3,
    "ExpCast flushes below −10.32",
    color=INK2,
    fontsize=6.5,
    va="bottom",
)
a1.text(
    float(N["rnCut"]) + 0.15,
    0.06,
    "exp + cast flushes\nbelow −12.48",
    color=INK2,
    fontsize=6.5,
    va="bottom",
)
a1.set_xlim(-13.5, 0.2)
a1.set_ylim(1e-3, 600)
a1.set_xlabel("score gap $u = S - m$ (nats)")
a1.set_ylabel("decoded probability ($\\times 2^{8}$)")
a1.set_title("(a) Probability encoding across the range", loc="left")
hl, ll = a1.get_legend_handles_labels()
style(a2, "both")
u2 = np.linspace(-3 * math.log(2), 0, 60001)
t2 = 256 * np.exp(u2)
a2.plot(u2, (dec(rne(u2)) / t2 - 1) * 100, color=ORANGE, lw=1.0, label="exp + RNE cast")
a2.plot(u2, (dec(expcast(u2)) / t2 - 1) * 100, color=BLUE, lw=1.0, label="ExpCast")
a2.axhline(0, color=AXIS, lw=0.8)
a2.set_xlim(u2[0], 0)
a2.set_ylim(-9, 9)
a2.set_xlabel("score gap $u$ (nats), three binades")
a2.set_ylabel("relative error (%)")
a2.set_title("(b) Per-element error near the peak", loc="left")
a2.set_ylim(-9.5, 9.5)
fig.legend(
    hl,
    ll,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.0),
    fontsize=7,
    handlelength=1.6,
    columnspacing=1.6,
)
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
fig.savefig(f"{OUT_FIG}/expcast.pdf")
plt.close(fig)

# ---- Figure: V repair cost/benefit (single series, no legend)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
style(ax, "both")
xs = [(t - REPAIR[0][3]) / REPAIR[0][3] * 100 for _, _, _, t in REPAIR]
ys = [(REPAIR[0][1] - e) / REPAIR[0][1] * 100 for _, e, _, _ in REPAIR]
ax.plot(xs, ys, color=BLUE, lw=2, solid_capstyle="round", zorder=2)
ax.scatter(xs, ys, s=30, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
for (b, *_), x, y in zip(REPAIR, xs, ys):
    ax.annotate(
        f"{b:g}%",
        (x, y),
        xytext=(4, -10 if b in (0.5, 1) else 5),
        textcoords="offset points",
        fontsize=7,
        color=INK2,
    )
ax.set_xlabel("total-time increase (%)")
ax.set_ylabel("relative error reduction (%)")
ax.set_xlim(-0.8, max(xs) * 1.12 + 0.5)
ax.set_ylim(0, max(ys) * 1.18 + 0.2)
fig.tight_layout()
fig.savefig(f"{OUT_FIG}/repair.pdf")
plt.close(fig)

print("macros:", len(N))
[print(f"  {kk} = {vv}") for kk, vv in N.items()]
print("tail:", [tuple(round(float(x), 3) for x in t) for t in tail])
print("vs_rows:", [tuple(round(x, 3) for x in r) for r in vs_rows])
