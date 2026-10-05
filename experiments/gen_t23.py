"""
表 t23：准则链决策移植到 NSGA-III/RVEA 的效果。
读 results/transplant_summary.csv → paper/tables/t23_transplant.tex。
星号：BH 校正（8 检验 = 4 实例 × 2 基线，逐指标）q<0.05。
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SUM = os.path.join(ROOT, "experiments", "results", "transplant_summary.csv")
TAB = os.path.join(ROOT, "paper", "tables", "t23_transplant.tex")
INST = [("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10), ("wfg2", 10)]
ALGOS = ["NSGA3", "RVEA"]


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
    d = defaultdict(dict)
    with open(SUM, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d[(r["problem"], int(r["m"]), r["algo"])] = r

    keys = [(n, m, a) for (n, m) in INST for a in ALGOS]
    q_igd = {k: q for k, q in zip(keys, bh([float(d[k]["p_igd"]) for k in keys]))}
    q_igdp = {k: q for k, q in zip(keys, bh([float(d[k]["p_igdp"]) for k in keys]))}

    out = ["\\begin{table}[t]\\centering\\small",
           "\\caption{Criterion-chain transfer: the four archived structural decisions "
           "(corner archive on, direction policy, \\texttt{replace} mode, budget $0.1$) transplanted onto "
           "NSGA-III and RVEA as a passive output merge that leaves their search dynamics untouched "
           "(same 10 seeds, $\\mathrm{FE}{=}10^5$). Mean $\\Delta$ of the transplanted output vs the plain "
           "baseline ($\\Delta<0$ = improvement); $^{*}$: significant after Benjamini--Hochberg correction "
           "across the 8 tests per metric ($q{<}0.05$).}",
           "\\label{tab:transplant}",
           "\\begin{tabular}{llcccc}", "\\toprule",
           " & & \\multicolumn{2}{c}{NSGA-III} & \\multicolumn{2}{c}{RVEA} \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
           "Instance & & $\\Delta$IGD (\\%) & $\\Delta$IGD$^{+}$ (\\%) & "
           "$\\Delta$IGD (\\%) & $\\Delta$IGD$^{+}$ (\\%) \\\\",
           "\\midrule"]
    for (n, m) in INST:
        cells = []
        for a in ALGOS:
            r = d[(n, m, a)]
            s1 = "$^{*}$" if q_igd[(n, m, a)] < 0.05 else ""
            s2 = "$^{*}$" if q_igdp[(n, m, a)] < 0.05 else ""
            cells.append("%+.1f%s & %+.1f%s" % (float(r["d_igd"]), s1,
                                                float(r["d_igdp"]), s2))
        out.append("%s $(m{=}%d)$ & & %s \\\\" % (n.upper(), m, " & ".join(cells)))
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(TAB, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("已写", TAB)


if __name__ == "__main__":
    main()