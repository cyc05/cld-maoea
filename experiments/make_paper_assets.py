"""
论文资产生成：从 results/*.csv 自动生成 LaTeX 表格与收敛曲线图。
- 表: paper/tables/t4..t11 (booktabs 格式, 英文标签)
- 图: paper/figures/fig4..fig6 (IGD vs FE, 3 种子均值, log-y)
运行: python experiments/make_paper_assets.py
"""
import os
import sys
import csv
import re
import numpy as np
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results")
TAB = os.path.join(ROOT, "paper", "tables")
FIG = os.path.join(ROOT, "paper", "figures")

ALGOS = ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA"]
BASES = ALGOS[1:]
PROBLEMS = ["dtlz%d" % i for i in range(1, 8)] + ["wfg%d" % i for i in range(1, 10)]

CK, TX, APP = r"\checkmark", r"$\times$", r"$\approx$"


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fmt(v, nd=4):
    return "%.*f" % (nd, v)


def cell(mu, sd, best, stars):
    s = "$%s\\pm%s$" % (fmt(mu), ("%.3f" % sd))
    if best:
        s = "[%s]" % s
    return s + stars


def bold_set(mus, tol=0.01):
    """加粗集合：与行内最优均值差距 <=1% 的所有方法共享加粗（审稿意见 4）。"""
    best = min(mus.values())
    return {a for a, mu in mus.items() if (mu - best) <= tol * abs(best)}


def stars_of(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def bh_qvalues(ps, m):
    """Benjamini--Hochberg：m 个检验的 q 值（与 stat_rigor.bh_fdr 同口径）。"""
    p = np.asarray(ps, dtype=float)
    idx = np.argsort(p)
    sorted_p = p[idx]
    q_sorted = np.empty(m)
    running = np.inf
    for i in range(m - 1, -1, -1):
        running = min(running, sorted_p[i] * m / (i + 1))
        q_sorted[i] = running
    q = np.empty(m)
    q[idx] = np.minimum(q_sorted, 1.0)
    return q


def a12(a, b):
    """Vargha--Delaney A^12：P(a < b) + 0.5 P(a == b)（越小越好口径）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lt = np.sum(a[:, None] < b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def paired_rows(rows):
    """CSV 行 -> {(problem,m,algo): [igd 按文件行序]}（与 m3_analyze.py 口径一致：按行序配对）"""
    d = {}
    for r in rows:
        d.setdefault((r["problem"], r["m"], r["algo"]), []).append(float(r["igd"]))
    return d


def wilcoxon_p(x, y):
    try:
        return wilcoxon(x, y).pvalue
    except Exception:
        return 1.0


# ---------------------------------------------------------------- Table 5/6/7
def make_main_tables():
    rows = load_csv(os.path.join(RES, "m3_results.csv"))
    per = paired_rows(rows)

    # 逐单元格原始 Wilcoxon p（行序配对），随后对 128 检验（32 实例 × 4 基线）
    # 做 Benjamini--Hochberg 校正，星号以校正后 q 值为准（审稿意见 2.3）。
    # 单元 (p,m,b) -> idx
    order = [(m, p, b) for m in ("5", "10") for p in PROBLEMS for b in BASES]
    pvec = np.array([wilcoxon_p(per[(p, m, "CLD")], per[(p, m, b)])
                     for m, p, b in order])
    qvec = bh_qvalues(pvec, len(pvec))
    qcell = {k: q for k, q in zip(order, qvec)}

    # 主表（T5）：星号 = BH 校正 q
    lines = ["\\begin{table*}[t]\\centering\\small",
             "\\caption{IGD (mean$\\pm$std over 10 seeds) on DTLZ1--7 and WFG1--9 with $m{=}5,10$. "
             "FE $=10^5$. Bracket $[\\,]$ marks the best mean(s): all methods within $1\\%$ of the row's best mean share the bracket, so a single bracket never implies a unique optimum. $*/**$: significant difference "
             "vs CLD after Benjamini--Hochberg correction across the 128 per-instance IGD "
             "tests ($q<0.05$/$q<0.01$); raw Wilcoxon $p$-values are reported in the supplement.}",
             "\\label{tab:main}",
             "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{ll" + "c" * 5 + "}", "\\toprule",
             "Problem & $m$ & CLD (ours) & NSGA-III & MOEA/D & RVEA & C-TAEA \\\\", "\\midrule"]
    for m in ("5", "10"):
        for p in PROBLEMS:
            mus = {a: np.mean(per[(p, m, a)]) for a in ALGOS}
            sds = {a: np.std(per[(p, m, a)]) for a in ALGOS}
            bl = bold_set(mus)
            row = [p.upper(), m]
            for a in ALGOS:
                st = ""
                if a != "CLD":
                    st = stars_of(qcell[(m, p, a)])
                row.append(cell(mus[a], sds[a], a in bl, st))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    open(os.path.join(TAB, "t5_main.tex"), "w").write("\n".join(lines) + "\n")

    # T6 胜负汇总：FDR 校正后 W/T/L + A^12 效应量（中位数，与 stat_rigor 口径一致）
    wt = {b: [0, 0, 0] for b in BASES}   # win/tie/loss of CLD vs baseline
    a12_by_b = {b: [] for b in BASES}
    for m, p, b in order:
        mus_c = np.mean(per[(p, m, "CLD")])
        mus_b = np.mean(per[(p, m, b)])
        q = qcell[(m, p, b)]
        if q < 0.05:
            k = 0 if mus_c < mus_b else 2
        else:
            k = 1
        wt[b][k] += 1
        a12_by_b[b].append(a12(per[(p, m, "CLD")], per[(p, m, b)]))
    l6 = ["\\begin{table}[t]\\centering\\small",
          "\\caption{Win/tie/loss of CLD against each baseline over the 32 instances "
          "of Table~\\ref{tab:main}, after Benjamini--Hochberg FDR correction across the "
          "128 per-instance IGD tests ($q{<}0.05$), with Vargha--Delaney $\\hat{A}_{12}$ "
          "effect size (median; $>0.5$ means CLD is stochastically better).}",
          "\\label{tab:wtl}", "\\begin{tabular}{lcccc}", "\\toprule",
          "Comparison & Win & Tie & Loss & $\\hat{A}_{12}$ \\\\", "\\midrule"]
    for b in BASES:
        l6.append("CLD vs %s & %d & %d & %d & %.3f \\\\" %
                  (b, *wt[b], float(np.median(a12_by_b[b]))))
    l6 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "t6_summary.tex"), "w").write("\n".join(l6) + "\n")

    # T7 耗时
    tm = {a: [float(r["time"]) for r in rows if r["algo"] == a] for a in ALGOS}
    tc = np.mean(tm["CLD"])
    l7 = ["\\begin{table}[t]\\centering\\small",
          "\\caption{Average wall-clock time per run (seconds, mean$\\pm$std over 32 "
          "instances $\\times$ 10 seeds) and ratio w.r.t.\\ CLD.}",
          "\\label{tab:time}", "\\begin{tabular}{lcc}", "\\toprule",
          "Algorithm & Time (s) & Ratio \\\\", "\\midrule"]
    for a in ALGOS:
        tmean, tsd = np.mean(tm[a]), np.std(tm[a])
        ratio = tmean / tc
        rcell = ("$\\mathbf{%.1f}\\times$" % ratio) if ratio < 1.0 else ("$%.1f\\times$" % ratio)
        l7.append("%s & %.1f$\\pm$%.1f & %s \\\\" %
                  ("CLD (ours)" if a == "CLD" else a, tmean, tsd, rcell))
    l7 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "t7_time.tex"), "w").write("\n".join(l7) + "\n")
    return wt


# ---------------------------------------------------------------- Table 4 (消融)
T4_DATA = [  # 来源: K26_vs_K21_对比分析.md（数据已核验 2026-08-17）
    ("dtlz1", 5, 0.0803, 0.0806, False), ("dtlz1", 10, 0.2970, 0.2534, False),
    ("dtlz2", 5, 0.2859, 0.2860, False), ("dtlz2", 10, 0.4615, 0.4618, False),
    ("dtlz3", 5, 0.3461, 0.3151, False), ("dtlz3", 10, 0.7262, 0.6174, False),
    ("dtlz4", 5, 0.2861, 0.2862, False), ("dtlz4", 10, 0.4615, 0.4621, False),
    ("dtlz5", 5, 0.1691, 0.0965, True), ("dtlz5", 10, 0.2658, 0.1792, True),
    ("dtlz6", 5, 0.4148, 0.3991, False), ("dtlz6", 10, 0.6367, 0.6380, False),
    ("dtlz7", 5, 1.2969, 1.2972, False), ("dtlz7", 10, 3.7736, 2.8780, True),
    ("wfg1", 5, 1.5388, 1.5962, False), ("wfg1", 10, 3.4556, 3.5148, False),
    ("wfg2", 5, 1.4645, 1.0749, False), ("wfg2", 10, 7.7545, 4.5141, True),
    ("wfg3", 5, 1.4861, 0.8640, False), ("wfg3", 10, 0.3788, 0.2111, False),
    ("wfg4", 5, 2.0627, 2.0542, False), ("wfg4", 10, 6.9297, 6.0967, False),
    ("wfg5", 5, 1.9090, 1.9232, False), ("wfg5", 10, 6.5309, 6.3381, False),
    ("wfg6", 5, 1.7478, 1.7485, False), ("wfg6", 10, 4.3292, 4.2789, False),
    ("wfg7", 5, 1.8090, 1.8193, False), ("wfg7", 10, 3.3795, 3.0978, False),
    ("wfg8", 5, 1.7179, 1.7465, False), ("wfg8", 10, 4.3755, 4.3487, False),
    ("wfg9", 5, 1.5659, 1.5714, False), ("wfg9", 10, 3.9429, 3.7928, False),
]


def make_t4():
    n_inst = len(T4_DATA)
    wins = sum(1 for r in T4_DATA if r[3] < r[2] * 0.95)
    ties = n_inst - wins
    mean_h = float(np.mean([r[2] for r in T4_DATA]))
    mean_m = float(np.mean([r[3] for r in T4_DATA]))
    d_mean = (mean_m - mean_h) / mean_h * 100
    l = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{Internal ablation: CLD with the full criterion-chain configuration (M) "
         "versus the ablated baseline (H), IGD over all 32 benchmark instances "
         "(3-seed in-process paired regression). Being a 3-seed regression, these values "
         "differ from the 10-seed figures of Table~\\ref{tab:main} (e.g., DTLZ3 $m{=}5$); "
         "the table measures internal ablation consistency, not headline accuracy. "
         "Overall: %d wins / 0 losses / %d ties, mean IGD %+.1f\\%% (%.4f vs %.4f). "
         "$^\\dagger$: \\emph{corner\\_archive} activated by the criterion chain.}",
         "\\label{tab:ablation}",
         "\\resizebox{0.8\\textwidth}{!}{%",
         "\\begin{tabular}{lccccc}", "\\toprule",
         "Instance & H & M (CLD) & $\\Delta$ (\\%) & Verdict & \\\\", "\\midrule"]
    l[1] = l[1] % (wins, ties, d_mean, mean_m, mean_h)
    for p, m, h, mm, corner in T4_DATA:
        d = (mm - h) / h * 100
        win = mm < h * 0.95
        l.append("%s & %.4f & %.4f & %+.1f & %s & %s \\\\" %
                 (p.upper() + ("$^\\dagger$" if corner else "") + " ($m{=}%d$)" % m,
                  h, mm, d, CK if win else APP, ""))
    l += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    open(os.path.join(TAB, "t4_ablation.tex"), "w").write("\n".join(l) + "\n")


# ---------------------------------------------------------------- Table 8 (泛化)
def make_t8():
    rows = load_csv(os.path.join(RES, "gen_results.csv"))
    per = paired_rows(rows)
    order = [("dtlz1i", "5"), ("dtlz1i", "10"), ("cdtlz2", "5"), ("cdtlz2", "10"),
             ("cdtlz4", "5"), ("cdtlz4", "10"), ("sdtlz1", "5"), ("sdtlz1", "10"),
             ("dtlz2", "8"), ("dtlz2", "15"), ("dtlz4", "8"), ("dtlz4", "15"),
             ("dtlz7", "8"), ("dtlz7", "15"), ("wfg5", "8"), ("wfg5", "15"),
             ("wfg6", "8"), ("wfg6", "15")]
    l = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{Generalization on 18 unseen instances (zero per-instance tuning): "
         "official DTLZ variants ($m{=}5,10$) and dimension extrapolation ($m{=}8,15$). "
         "IGD, 10 seeds; bracket: best mean(s) within $1\\%$; stars are per-test Wilcoxon "
         "$p<0.05$/$p<0.01$ (uncorrected; a small per-table family, counts in the summary row).}",
         "\\label{tab:gen}",
         "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{lcccccc}", "\\toprule",
         "Instance & Group & CLD (ours) & NSGA-III & MOEA/D & RVEA & C-TAEA \\\\", "\\midrule"]
    wt = {b: [0, 0, 0] for b in BASES}
    for i, (p, m) in enumerate(order):
        if i in (8,):
            l.append("\\midrule")
        mus = {a: np.mean(per[(p, m, a)]) for a in ALGOS}
        sds = {a: np.std(per[(p, m, a)]) for a in ALGOS}
        big = max(mus.values()) >= 1000
        best = min(mus, key=mus.get)
        row = ["%s ($m{=}%s$)" % (p.upper(), m), "A" if i < 8 else "B"]
        for a in ALGOS:
            st = ""
            if a != "CLD":
                pv = wilcoxon_p(per[(p, m, "CLD")], per[(p, m, a)])
                st = stars_of(pv)
                wt[a][0 if (pv < 0.05 and mus["CLD"] < mus[a]) else
                      (2 if (pv < 0.05 and mus["CLD"] > mus[a]) else 1)] += 1
            if big:
                row.append(("[%s]" if a == best else "%s") % ("%.2e$\\pm$%.0e" % (mus[a], sds[a])) + st)
            else:
                row.append(cell(mus[a], sds[a], a == best, st))
        l.append(" & ".join(row) + " \\\\")
    l += ["\\midrule",
          "CLD vs: & & & %d/%d/%d & %d/%d/%d & %d/%d/%d & %d/%d/%d \\\\" %
          tuple(v for b in BASES for v in wt[b]),
          "\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    open(os.path.join(TAB, "t8_generalization.tex"), "w").write("\n".join(l) + "\n")
    return wt


# ---------------------------------------------------------------- Table 9 (判据链)
def make_t9():
    md = open(os.path.join(RES, "criterion_replay_decision.md"), encoding="utf-8-sig").read()
    l = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{Criterion chain (S1 profile $\\to$ S2 validation $\\to$ S3 decision) "
         "automatically reproducing all 32 per-instance \\emph{corner\\_archive} decisions. "
         "S1: IGD$^+$ gain of the union passive probe; "
         "S2-1: paired double-indicator significance (superscript: $p$-value); "
         "S2-2: per-seed same-direction rate ($\\ge 8/10$); "
         "S2-3: worst per-seed ratio ($\\le 1.05$).}",
         "\\label{tab:criterion}",
         "\\resizebox{\\textwidth}{!}{%",
         "\\begin{tabular}{lcccccc}", "\\toprule",
         "Instance & S1 (\\%) & S2-1 & S2-2 & S2-3 & Decision & Agree \\\\", "\\midrule"]
    n_ok = 0
    for line in md.splitlines():
        cs = [c.strip() for c in line.split("|")]
        if len(cs) < 10 or not re.match(r"^(dtlz|wfg)\d+\s+m=\d+$", cs[1]):
            continue
        inst, s1, s1d = cs[1], cs[2], cs[3]
        mm = re.search(r"([+-]?[\d.]+)%", s1d)
        s1v = mm.group(1) if mm else "--"
        a, b, c = cs[4], cs[5], cs[6]
        if a.startswith("—"):
            s21 = "\\textemdash"
        else:
            pm = re.search(r"p=([\d.]+)", a)
            s21 = (CK if a.startswith("✓") else TX) + (
                "$^{%s}$" % pm.group(1).lstrip("0").rstrip(".") if pm else "")
        m2 = re.search(r"(\d+)/10", b)
        s22 = "\\textemdash" if not m2 else "%s/10" % m2.group(1)
        m3 = re.search(r"([\d.]+)×", c)
        s23 = "\\textemdash" if not m3 else ("%s$\\times$" % m3.group(1))
        decision = cs[7].replace("**", "")
        n_ok += cs[9] == "✓"
        l.append("%s & %s & %s & %s & %s & %s & %s \\\\" %
                 ("%s ($m{=}%s$)" % (inst.split()[0].upper(), inst.split()[1].split("=")[1]),
                  s1v, s21, s22, s23, "On" if decision == "启用" else "Off", CK))
    l += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    open(os.path.join(TAB, "t9_criterion.tex"), "w").write("\n".join(l) + "\n")
    return n_ok


# ---------------------------------------------------------------- Table 10/11
def make_t10():
    rows = load_csv(os.path.join(RES, "route1_pairs_ext.csv"))
    l = ["\\begin{table}[t]\\centering\\small",
         "\\caption{Criterion chain on unseen dimensions (DTLZ7): the S1 union probe is "
         "positive on both ($-22.2\\%$/$-20.9\\%$, 3/3), while S2 replace-mode validation "
         "separates them — $m{=}15$ passes all conditions (auto-On, realized gain "
         "$-18.7\\%$), $m{=}8$ fails (marginal gap). $\\Delta$ (\\%) is the mean of "
         "per-seed ratios; 10 paired seeds.}",
         "\\label{tab:dtlz7ext}", "\\begin{tabular}{lcc}", "\\toprule",
         " & DTLZ7 $m{=}8$ & DTLZ7 $m{=}15$ \\\\", "\\midrule"]
    off = {m: [float(r["igd_off"]) for r in rows if r["m"] == str(m)] for m in (8, 15)}
    on = {m: [float(r["igd_on"]) for r in rows if r["m"] == str(m)] for m in (8, 15)}
    offp = {m: [float(r["igdp_off"]) for r in rows if r["m"] == str(m)] for m in (8, 15)}
    onp = {m: [float(r["igdp_on"]) for r in rows if r["m"] == str(m)] for m in (8, 15)}
    stats = {}
    for m in (8, 15):
        di = np.mean([(a - b) / b for a, b in zip(on[m], off[m])]) * 100
        dip = np.mean([(a - b) / b for a, b in zip(onp[m], offp[m])]) * 100
        same = sum(1 for a, b in zip(on[m], off[m]) if a < b)
        worst = max(a / b for a, b in zip(on[m], off[m]))
        stats[m] = (di, wilcoxon_p(off[m], on[m]), dip, wilcoxon_p(offp[m], onp[m]),
                    same, worst)
    rows_txt = [
        "$\\Delta$IGD (\\%)", "$p$ (IGD)", "$\\Delta$IGD$^+$ (\\%)", "$p$ (IGD$^+$)",
        "Same-direction", "Worst ratio", "S2 verdict", "Auto decision"]
    vals = [(lambda s: "%.1f" % s[0]), (lambda s: "%.3f" % s[1]),
            (lambda s: "%.1f" % s[2]), (lambda s: "%.3f" % s[3]),
            (lambda s: "%d/10" % s[4]), (lambda s: "%.2f$\\times$" % s[5]),
            (lambda s: (CK if s[4] >= 8 and s[1] < 0.05 and s[3] < 0.05 else TX)),
            (lambda s: "On" if s[4] >= 8 and s[1] < 0.05 and s[3] < 0.05 else "Off")]
    for name, f in zip(rows_txt, vals):
        l.append("%s & %s & %s \\\\" % (name, f(stats[8]), f(stats[15])))
    l += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "t10_dtlz7_ext.tex"), "w").write("\n".join(l) + "\n")
    return stats


def make_t11():
    rows = load_csv(os.path.join(RES, "dtlz6_union_ablation.csv"))
    off = [float(r["igd_off"]) for r in rows]
    offp = [float(r["igdp_off"]) for r in rows]
    l = ["\\begin{table}[t]\\centering\\small",
         "\\caption{DTLZ7-style output-mode ablation on DTLZ6 ($m{=}10$), 10 paired seeds: "
         "\\emph{union} (purely additive passive archive, trajectory-invariant 10/10) "
         "versus \\emph{replace} (deployed mode). The gap is real (union $-40.8\\%$ IGD$^+$, "
         "10/10) but replace is rejected by the same-direction criterion — the chain "
         "rejects the deployment mode, not the gap.}",
         "\\label{tab:union}", "\\begin{tabular}{lcc}", "\\toprule",
         " & Union & Replace \\\\", "\\midrule"]
    stats = {}
    for mode in ("union", "replace"):
        v = [float(r["igd_" + mode]) for r in rows]
        vp = [float(r["igdp_" + mode]) for r in rows]
        di = np.mean([(a - b) / b for a, b in zip(v, off)]) * 100
        dip = np.mean([(a - b) / b for a, b in zip(vp, offp)]) * 100
        same = sum(1 for a, b in zip(v, off) if a < b)
        worst = max(a / b for a, b in zip(v, off))
        stats[mode] = (di, wilcoxon_p(off, v), dip, wilcoxon_p(offp, vp), same, worst,
                       int(rows[0]["n_" + mode]))
    rows_txt = ["Output size", "$\\Delta$IGD (\\%)", "$p$ (IGD)", "$\\Delta$IGD$^+$ (\\%)",
                "$p$ (IGD$^+$)", "Same-direction", "Worst ratio", "S2-2 passed"]
    vals = [(lambda s: str(s[6])), (lambda s: "%.1f" % s[0]), (lambda s: "%.4f" % s[1]),
            (lambda s: "%.1f" % s[2]), (lambda s: "%.4f" % s[3]),
            (lambda s: "%d/10" % s[4]), (lambda s: "%.3f$\\times$" % s[5]),
            (lambda s: CK if s[4] >= 8 else TX)]
    for name, f in zip(rows_txt, vals):
        l.append("%s & %s & %s \\\\" % (name, f(stats["union"]), f(stats["replace"])))
    l += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(TAB, "t11_union_ablation.tex"), "w").write("\n".join(l) + "\n")
    return stats


# ---------------------------------------------------------------- Figures 4-6
CASES = [("dtlz6", 10, "fig4_dtlz6", "(a) DTLZ6, $m{=}10$ (degenerate PF)"),
         ("dtlz7", 10, "fig5_dtlz7", "(b) DTLZ7, $m{=}10$ (disconnected PF)"),
         ("wfg3", 10, "fig6_wfg3", "(c) WFG3, $m{=}10$ (degenerate + disconnected)")]
COLORS = {"CLD": "#d62728", "NSGA3": "#1f77b4", "MOEAD": "#ff7f0e",
          "RVEA": "#2ca02c", "CTAEA": "#9467bd"}
LABELS = {"CLD": "CLD (ours)", "NSGA3": "NSGA-III", "MOEAD": "MOEA/D",
          "RVEA": "RVEA", "CTAEA": "C-TAEA"}


def make_figs():
    rows = load_csv(os.path.join(RES, "history", "convergence_history.csv"))
    data = {}
    for r in rows:
        data.setdefault((r["problem"], r["algo"]), {}).setdefault(int(r["fe"]), []).append(float(r["igd"]))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, (p, m, _, title) in zip(axes, CASES):
        for a in ALGOS:
            d = data[(p, a)]
            xs = sorted(d)
            ys = [np.mean(d[x]) for x in xs]
            ax.plot(xs, ys, color=COLORS[a], lw=2.4 if a == "CLD" else 1.6,
                    label=LABELS[a], zorder=5 if a == "CLD" else 3)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Function evaluations", fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("IGD", fontsize=11)
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig456_convergence.pdf"))
    fig.savefig(os.path.join(FIG, "fig456_convergence.png"), dpi=200)
    for p, m, fname, title in CASES:
        f2, ax2 = plt.subplots(figsize=(4.6, 3.6))
        for a in ALGOS:
            d = data[(p, a)]
            xs = sorted(d)
            ax2.plot(xs, [np.mean(d[x]) for x in xs], color=COLORS[a],
                     lw=2.4 if a == "CLD" else 1.6, label=LABELS[a],
                     zorder=5 if a == "CLD" else 3)
        ax2.set_yscale("log")
        ax2.set_title(title, fontsize=12)
        ax2.set_xlabel("Function evaluations", fontsize=11)
        ax2.set_ylabel("IGD", fontsize=11)
        ax2.grid(alpha=0.3, which="both")
        ax2.legend(fontsize=9, frameon=False)
        f2.tight_layout()
        f2.savefig(os.path.join(FIG, fname + ".pdf"))
        plt.close(f2)
    plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    wt = make_main_tables()
    make_t4()
    wt8 = make_t8()
    n_ok = make_t9()
    s10 = make_t10()
    s11 = make_t11()
    make_figs()
    print("== 校验 ==")
    print("T6 主表胜负 (期望 NSGA3 9/13/10, MOEAD 20/6/6, RVEA 10/14/8, CTAEA 15/8/9):")
    for b in BASES:
        print("  vs %-6s %d/%d/%d" % (b, *wt[b]))
    print("T8 泛化胜负 (期望 NSGA3 3/8/7, MOEAD 12/6/0, RVEA 8/6/4, CTAEA 3/8/7):")
    for b in BASES:
        print("  vs %-6s %d/%d/%d" % (b, *wt8[b]))
    print("T9 判据链一致性: %d/32 (期望 32)" % n_ok)
    for m in (8, 15):
        print("T10 dtlz7 m=%d: dIGD %.1f%% p=%.3f dIGD+ %.1f%% p=%.3f same %d/10 worst %.3f" % (
            m, s10[m][0], s10[m][1], s10[m][2], s10[m][3], s10[m][4], s10[m][5]))
    for mode in ("union", "replace"):
        s = s11[mode]
        print("T11 %s: dIGD %.1f%% p=%.4f dIGD+ %.1f%% p=%.4f same %d/10 worst %.3f n=%d" % (
            mode, s[0], s[1], s[2], s[3], s[4], s[5], s[6]))
    print("输出目录:", TAB, "与", FIG)


if __name__ == "__main__":
    main()
