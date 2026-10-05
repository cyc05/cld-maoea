"""
新基线表 t22：AGEMOEA / AGEMOEA2 / UNSGA3 与 CLD 的逐实例 IGD。
星号：BH 校正（96 检验 = 32 实例 × 3 基线，IGD 口径）q<0.05。
产物：paper/tables/t22_newbaselines.tex
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NB = os.path.join(ROOT, "experiments", "results", "new_baselines.csv")
SB = os.path.join(ROOT, "experiments", "results", "new_baselines_summary.csv")
M3 = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
TAB = os.path.join(ROOT, "paper", "tables", "t22_newbaselines.tex")
BASES = ["AGEMOEA", "AGEMOEA2", "UNSGA3"]
ALGOS = ["CLD"] + BASES


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
    per = defaultdict(list)
    with open(M3, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                per[(r["problem"], int(r["m"]), "CLD")].append(float(r["igd"]))
    with open(NB, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            per[(r["problem"], int(r["m"]), r["algo"])].append(float(r["igd"]))

    summ = defaultdict(dict)
    with open(SB, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            summ[(r["problem"], int(r["m"]), r["algo"])] = r

    order = [(p, m) for m in (5, 10) for p in
             (["dtlz%d" % i for i in range(1, 8)] + ["wfg%d" % i for i in range(1, 10)])]
    keys = [(p, m, b) for (p, m) in order for b in BASES]
    ps = [float(summ[k]["p_igd"]) for k in keys if k in summ]
    qcell = {}
    if len(ps) == len(keys):
        qcell = {k: q for k, q in zip(keys, bh(ps))}

    out = ["\\begin{table*}[t]\\centering\\small",
           "\\caption{Comparison with three newer \\textsc{pymoo}-implemented algorithms "
           "(AGE-MOEA, AGE-MOEA2, U-NSGA-III): IGD (mean$\\pm$std, 10 seeds), "
           "$\\mathrm{FE}{=}10^5$, same reference-direction and operator protocol as "
           "Table~\\ref{tab:main}. Bracket $[\\,]$ marks the best mean(s) within $1\\%$; $*/**$: significant "
           "vs CLD after Benjamini--Hochberg correction across the tests ($q{<}0.05$/"
           "$q{<}0.01$). $^{\\dagger}$: AGEMOEA2 raised a numerical exception in its "
           "survival normalization on DTLZ4 ($m{=}5$) for $2/10$ seeds; the cell aggregates "
           "the remaining $8$ seeds and the crashed runs are reported as n/a (a robustness "
           "property of the baseline itself).}",
           "\\label{tab:newbase}",
           "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{ll" + "c" * len(ALGOS) + "}",
           "\\toprule",
           "Problem & $m$ & " + " & ".join(a if a != "CLD" else "CLD (ours)" for a in ALGOS) + " \\\\",
           "\\midrule"]

    wtl = {b: [0, 0, 0] for b in BASES}
    a12s = {b: [] for b in BASES}
    for i, (p, m) in enumerate(order):
        if i == 16:
            out.append("\\midrule")
        mus = {a: np.mean(per[(p, m, a)]) for a in ALGOS}
        sds = {a: np.std(per[(p, m, a)]) for a in ALGOS}
        best = min(mus, key=mus.get)
        row = ["%s" % p.upper(), str(m)]
        for a in ALGOS:
            st = ""
            if a != "CLD" and (p, m, a) in qcell:
                q = qcell[(p, m, a)]
                st = "**" if q < 0.01 else ("*" if q < 0.05 else "")
                dm = (mus["CLD"] - mus[a])
                if q < 0.05:
                    wtl[a][0 if dm < 0 else 2] += 1
                else:
                    wtl[a][1] += 1
                a12s[a].append(float(summ[(p, m, a)]["a12_igd"]))
            s = "%.4f$\\pm$%.3f" % (mus[a], sds[a])
            if a == best:
                s = "[%s]" % s
            row.append(s + st)
        out.append(" & ".join(row) + " \\\\")
    out.append("\\midrule")
    out.append("CLD w/t/l, $\\hat A_{12}$ & & " + " & ".join(
        "%d/%d/%d, %.3f" % (*wtl[b], np.median(a12s[b])) for b in BASES) +
        " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    with open(TAB, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("已写", TAB)
    for b in BASES:
        print("  %-8s: %d/%d/%d medianA12 %.3f" %
              (b, *wtl[b], np.median(a12s[b])))


if __name__ == "__main__":
    main()