"""
M7 结果统计分析：CLD-M7（problem_type 区分 d⊥）vs M6/M5/M4。
运行: python experiments/m7_analyze.py
"""
import os
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
M4_CSV = os.path.join(ROOT, "experiments", "results", "m4_results.csv")
M5_CSV = os.path.join(ROOT, "experiments", "results", "m5b_results.csv")
M6_CSV = os.path.join(ROOT, "experiments", "results", "m6_results.csv")
M7_CSV = os.path.join(ROOT, "experiments", "results", "m7_results.csv")
BASELINES = ["NSGA3", "MOEAD", "RVEA", "CTAEA"]
ALGOS = ["CLD-M7", "CLD-M6", "CLD-M5", "CLD-M4", "CLD-M3"] + BASELINES


def load():
    data = defaultdict(lambda: defaultdict(list))
    with open(M3_CSV, newline="") as f:
        for r in csv.DictReader(f):
            algo = "CLD-M3" if r["algo"] == "CLD" else r["algo"]
            data[(r["problem"], int(r["m"]))][algo].append(
                (float(r["igd"]), float(r["igd_plus"]), float(r["time"])))
    for csv_path, tag in [(M4_CSV, "CLD-M4"), (M5_CSV, "CLD-M5"),
                           (M6_CSV, "CLD-M6"), (M7_CSV, "CLD-M7")]:
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    if r["algo"] == "CLD":
                        data[(r["problem"], int(r["m"]))][tag].append(
                            (float(r["igd"]), float(r["igd_plus"]), float(r["time"])))
    return data


def sig_mark(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def wilp(a, b):
    n = min(len(a), len(b))
    if n < 5:
        return 1.0
    try:
        return wilcoxon(a[:n], b[:n]).pvalue
    except Exception:
        return 1.0


def main():
    data = load()
    keys = sorted(data.keys(), key=lambda k: (k[1], k[0]))
    win_count = defaultdict(lambda: [0, 0, 0])

    print(f"{'problem':8s} {'m':>2s}  " + "  ".join(f"{a:>18s}" for a in ALGOS))
    for (name, m) in keys:
        means = {}
        for a in ALGOS:
            igds = [x[0] for x in data[(name, m)].get(a, [])]
            means[a] = (np.mean(igds), np.std(igds), igds) if igds \
                else (np.nan, np.nan, [])
        marks = {}
        cld = means["CLD-M7"]
        for a in ["CLD-M6", "CLD-M5", "CLD-M4", "CLD-M3"] + BASELINES:
            p = wilp(cld[2], means[a][2])
            marks[a] = sig_mark(p)
            if np.isnan(cld[0]) or np.isnan(means[a][0]):
                continue
            if cld[0] < means[a][0] and p < 0.05:
                win_count[a][0] += 1
            elif cld[0] > means[a][0] and p < 0.05:
                win_count[a][2] += 1
            else:
                win_count[a][1] += 1
        best = min((kv for kv in means.items() if not np.isnan(kv[1][0])),
                   key=lambda kv: kv[1][0])[0]
        row = f"{name:8s} {m:2d}  "
        for a in ALGOS:
            mu, sd, _ = means[a]
            cell = "N/A" if np.isnan(mu) else f"{mu:.4f}±{sd:.3f}"
            if a == best:
                cell = "[" + cell + "]"
            if a in marks:
                cell += marks[a]
            row += f"  {cell:>18s}"
        print(row)

    print("\n=== CLD-M7 vs 各算法 胜/平/负（IGD 均值 + Wilcoxon α=0.05）===")
    for a in ["CLD-M6", "CLD-M5", "CLD-M4", "CLD-M3"] + BASELINES:
        w, t, l = win_count[a]
        print(f"  vs {a:7s}: {w}胜 {t}平 {l}负")

    print("\n=== 平均运行时间（秒）===")
    time_sum = defaultdict(list)
    for key, algos in data.items():
        for a, runs in algos.items():
            time_sum[a].extend(x[2] for x in runs)
    for a in ALGOS:
        if time_sum[a]:
            print(f"  {a:7s}: {np.mean(time_sum[a]):8.1f}s "
                  f"± {np.std(time_sum[a]):.1f}")

    # IGD+ 汇总
    print("\n=== CLD-M7 vs 各算法 胜/平/负（IGD+ 均值 + Wilcoxon α=0.05）===")
    win_p = defaultdict(lambda: [0, 0, 0])
    for (name, m) in keys:
        cldp = [x[1] for x in data[(name, m)].get("CLD-M7", [])]
        if not cldp:
            continue
        for a in ["CLD-M6", "CLD-M5", "CLD-M4", "CLD-M3"] + BASELINES:
            op = [x[1] for x in data[(name, m)].get(a, [])]
            if not op:
                continue
            p = wilp(cldp, op)
            if np.mean(cldp) < np.mean(op) and p < 0.05:
                win_p[a][0] += 1
            elif np.mean(cldp) > np.mean(op) and p < 0.05:
                win_p[a][2] += 1
            else:
                win_p[a][1] += 1
    for a in ["CLD-M6", "CLD-M5", "CLD-M4", "CLD-M3"] + BASELINES:
        w, t, l = win_p[a]
        print(f"  vs {a:7s}: {w}胜 {t}平 {l}负")


if __name__ == "__main__":
    main()
