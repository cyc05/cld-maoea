"""
审稿意见 2.1：等 wall-clock 对比的正式表格 t20。

对每个基线（NSGA3/RVEA）：
  固定预算（FE=1e5，来自 m3_results.csv）：CLD vs 基线@1e5（与 t6 同口径，BH 校正）
  等墙钟预算（来自 equal_budget.csv）：基线 FE 提升到 ≈ CLD 墙钟，CLD@1e5 vs 基线@等FE

产物：paper/tables/t20_equal_budget.tex + 控制台汇总
运行：python experiments/gen_t20.py
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

M3 = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
EQ = os.path.join(ROOT, "experiments", "results", "equal_budget.csv")
TAB = os.path.join(ROOT, "paper", "tables", "t20_equal_budget.tex")

PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]


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


def a12(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def load_seeds(path):
    d = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]), r["algo"])
            d[key][int(r["seed"])] = (float(r["igd"]), float(r["igd_plus"]),
                                      int(r.get("fe", -1)))
    return d


def wtl(res):
    w = t = l = 0
    for dm, p in res:
        if p < 0.05:
            if dm < 0:
                w += 1
            else:
                l += 1
        else:
            t += 1
    return w, t, l


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    m3 = load_seeds(M3)
    eq = load_seeds(EQ)

    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Accuracy of CLD against the two lighter baselines under a \\emph{fixed} "
        "evaluation budget ($\\mathrm{FE}{=}10^5$; the fixed-FE counts repeat Table~"
        "\\ref{tab:wtl}) and under an \\emph{equal wall-clock} budget, where each baseline "
        "run's budget is raised to $\\lceil \\bar T_{\\text{CLD}}/\\bar T_{\\text{base}}\\rceil"
        "\\times10^5$ per instance so that one run matches CLD's $15.7$\\,s. All counts are "
        "win/tie/loss of CLD (Benjamini--Hochberg across the 32 per-instance tests, $q{<}0.05$); "
        "$\\hat{A}_{12}$ is the median stochastic-dominance effect ($>0.5$ favors CLD); mean FE "
        "is the equal-time per-run budget in $10^3$ evaluations. CLD itself always runs at "
        "$10^5$ evaluations.}",
        "\\label{tab:eqbudget}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Baseline & fixed FE w/t/l & $\\hat{A}_{12}$ & equal-time IGD w/t/l & equal-time IGD$^{+}$ w/t/l & mean FE \\\\",
        "\\midrule",
    ]

    body = []
    for base in ("NSGA3", "RVEA"):
        # ---- 固定 FE ----
        fixed_res, fixed_a12 = [], []
        for pname in PROBLEMS:
            for m in M_VALUES:
                c, b = m3.get((pname, m, "CLD")), m3.get((pname, m, base))
                if not c or not b:
                    continue
                seeds = sorted(set(c) & set(b))
                cv = np.array([c[s][0] for s in seeds])
                bv = np.array([b[s][0] for s in seeds])
                p = wilcoxon(cv, bv).pvalue if (cv != bv).any() else 1.0
                fixed_res.append((cv.mean() - bv.mean(), p))
                fixed_a12.append(a12(cv, bv))
        fq = bh([x[1] for x in fixed_res])
        f_wtl = wtl([(x[0], q) for x, q in zip(fixed_res, fq)])
        f_a12 = float(np.median(fixed_a12))

        et = {}
        for metric in (0, 1):
            res, a12s, fes = [], [], []
            for pname in PROBLEMS:
                for m in M_VALUES:
                    c, b = m3.get((pname, m, "CLD")), eq.get((pname, m, base))
                    if not c or not b:
                        continue
                    seeds = sorted(set(c) & set(b))
                    cv = np.array([c[s][metric] for s in seeds])
                    bv = np.array([b[s][metric] for s in seeds])
                    p = wilcoxon(cv, bv).pvalue if (cv != bv).any() else 1.0
                    res.append((cv.mean() - bv.mean(), p))
                    a12s.append(a12(cv, bv))
                    fes.append(b[seeds[0]][2])
            qq = bh([x[1] for x in res])
            et[metric] = (wtl([(x[0], q) for x, q in zip(res, qq)]),
                          float(np.median(a12s)), int(np.mean(fes) / 1000))
        body.append("%-7s & %d/%d/%d & %.3f & %d/%d/%d & %d/%d/%d & %d \\\\" % (
            "NSGA-III" if base == "NSGA3" else "RVEA",
            *f_wtl, f_a12, *et[0][0], *et[1][0], et[0][2]))
        print("%-7s fixed %d/%d/%d (a12 %.3f) | equal IGD %d/%d/%d IGD+ %d/%d/%d | meanFE %dk" % (
            base, *f_wtl, f_a12, *et[0][0], *et[1][0], et[0][2]))

    out += body + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(TAB, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("已写", TAB)


if __name__ == "__main__":
    main()