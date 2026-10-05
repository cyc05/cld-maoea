"""
两两消融表 t21：M1/M12/M13/M23 vs 完整版 F。
读 results/module_ablation_summary.csv（由 module_ablation.py 产出）。
星号：BH 校正（128 检验 = 32 实例 × 4 变体，IGD 口径）q<0.05；汇总行同口径。
产物：paper/tables/t21_module_abl.tex
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SUM_CSV = os.path.join(ROOT, "experiments", "results", "module_ablation_summary.csv")
TAB = os.path.join(ROOT, "paper", "tables", "t21_module_abl.tex")
CONFS = ["M1", "M12", "M13", "M23"]


def bh(ps):
    p = np.asarray(ps, float)
    idx = np.argsort(p)
    sp = p[idx]
    q = np.empty(len(p))
    run = np.inf
    for i in range(len(p) - 1, -1, -1):
        run = min(run, sp[i] * len(p) / (i + 1))
        q[i] = run
    q[idx] = np.minimum(q, 1.0)
    return q


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = defaultdict(dict)
    with open(SUM_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            data[(r["problem"], int(r["m"]), r["conf"])] = r

    order = [(p, m) for m in (5, 10) for p in
             (["dtlz%d" % i for i in range(1, 8)] + ["wfg%d" % i for i in range(1, 10)])]

    # BH（128 检验，IGD）
    keys = [(p, m, c) for (p, m) in order for c in CONFS]
    ps = [float(data[k]["p_igd"]) for k in keys]
    qcell = {k: q for k, q in zip(keys, bh(ps))}

    out = ["\\begin{table*}[t]\\centering\\small",
           "\\caption{Pairwise module ablation: each variant against the full configuration F "
           "(M1+M2+M3; F is Table~\\ref{tab:main}'s CLD), mean $\\Delta$IGD (\\%) over 10 seeds "
           "($\\Delta<0$ favors F). M1: closed-loop dominance control; M2: directional "
           "differentiation; M3: R2 greedy truncation with protection. An ``off'' module uses "
           "its standard substitute---$\\phi\\equiv0.5$ (standard dominance, no gate/escape) for "
           "M1, uniform $\\phi$ mating for M2, and crowding-distance truncation for M3; the "
           "corner-archive/escape overrides of the archived configuration are held fixed. "
           "$^{*}$: significant vs F after Benjamini--Hochberg correction across the $128$ "
           "tests ($q{<}0.05$). The last row aggregates over instances: F's win/tie/loss "
           "(corrected) and the \\emph{median} per-instance $\\Delta$ (the mean is dominated by "
           "the DTLZ1/3 linear-front collapses of the M3-off variants).}",
           "\\label{tab:moduleabl}",
           "\\resizebox{0.85\\textwidth}{!}{%",
           "\\begin{tabular}{l" + "c" * len(CONFS) + "}",
           "\\toprule",
           "Instance & " + " & ".join(CONFS) + " \\\\",
           "\\midrule"]

    for i, (p, m) in enumerate(order):
        if i == 16:
            out.append("\\midrule")
        cells = []
        for c in CONFS:
            r = data[(p, m, c)]
            star = "$^{*}$" if qcell[(p, m, c)] < 0.05 else ""
            cells.append("%+.1f%s" % (float(r["d_igd"]), star))
        out.append("%s $(m{=}%d)$ & %s \\\\" % (p.upper(), m, " & ".join(cells)))

    # 汇总行：BH 口径 W/T/L + 平均 Δ + 中位 A12
    wtl = {c: [0, 0, 0] for c in CONFS}
    ds = {c: [] for c in CONFS}
    a12s = {c: [] for c in CONFS}
    for (p, m) in order:
        for c in CONFS:
            r = data[(p, m, c)]
            ds[c].append(float(r["d_igd"]))
            a12s[c].append(float(r["a12_igd"]))
            q = qcell[(p, m, c)]
            if q < 0.05:
                wtl[c][0 if float(r["d_igd"]) < 0 else 2] += 1
            else:
                wtl[c][1] += 1
    out.append("\\midrule")
    out.append("w/t/l vs F & " + " & ".join(
        "%d/%d/%d (med. %+.1f\\%%)" % (*wtl[c], np.median(ds[c])) for c in CONFS) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    with open(TAB, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("已写", TAB)
    for c in CONFS:
        print("  %-4s vs F: %d/%d/%d med d%+.1f%% mean d%+.1f%% medianA12 %.3f" %
              (c, *wtl[c], np.median(ds[c]), np.mean(ds[c]), np.median(a12s[c])))


if __name__ == "__main__":
    main()