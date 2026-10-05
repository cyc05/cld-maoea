"""
M3 结果统计分析：均值±std、Wilcoxon 符号秩检验（CLD vs 各基线）、运行时间对比。
运行: python experiments/m3_analyze.py
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
ALGOS = ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA"]
BASELINES = ["NSGA3", "MOEAD", "RVEA", "CTAEA"]


def load(csv_path):
    data = defaultdict(lambda: defaultdict(list))   # (problem, m) -> algo -> [(igd, igdp, time)]
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]))
            data[key][r["algo"]].append(
                (float(r["igd"]), float(r["igd_plus"]), float(r["time"])))
    return data


def sig_mark(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RESULT_CSV,
                    help="结果 CSV 路径（默认 m3_results.csv；MaF 用 "
                         "experiments/results/maf_results.csv）")
    args = ap.parse_args()
    csv_path = args.csv

    data = load(csv_path)
    keys = sorted(data.keys(), key=lambda k: (k[1], k[0]))
    win_count = defaultdict(lambda: [0, 0, 0])  # metric -> [win, tie, lose] vs CLD

    print(f"{'problem':8s} {'m':>2s}  " + "  ".join(f"{a:>18s}" for a in ALGOS))
    for (name, m) in keys:
        row = f"{name:8s} {m:2d}  "
        means = {}
        for a in ALGOS:
            igds = [x[0] for x in data[(name, m)].get(a, [])]
            means[a] = (np.mean(igds), np.std(igds), igds)
        # Wilcoxon: CLD vs baseline
        marks = {}
        cld_igds = means["CLD"][2]
        for a in BASELINES:
            other = means[a][2]
            n = min(len(cld_igds), len(other))
            if n >= 5:
                try:
                    p = wilcoxon(cld_igds[:n], other[:n]).pvalue
                except Exception:
                    p = 1.0
                marks[a] = sig_mark(p)
                # 胜负判定（均值 + 显著性）
                if means["CLD"][0] < means[a][0] and p < 0.05:
                    win_count[a][0] += 1
                elif means["CLD"][0] > means[a][0] and p < 0.05:
                    win_count[a][2] += 1
                else:
                    win_count[a][1] += 1
            else:
                # 小样本筛查口径（如 3 种子）：按均值判，Wilcoxon 无功效不计显著性
                marks[a] = ""
                if means["CLD"][0] < means[a][0]:
                    win_count[a][0] += 1
                elif means["CLD"][0] > means[a][0]:
                    win_count[a][2] += 1
                else:
                    win_count[a][1] += 1
        best = min(means.items(), key=lambda kv: kv[1][0])[0]
        for a in ALGOS:
            mu, sd, _ = means[a]
            cell = f"{mu:.4f}±{sd:.3f}"
            if a == best:
                cell = "[" + cell + "]"
            if a in marks:
                cell += marks[a]
            row += f"  {cell:>18s}"
        print(row)

    print("\n=== CLD vs 各基线 胜/平/负（IGD 均值 + Wilcoxon α=0.05）===")
    for a in BASELINES:
        w, t, l = win_count[a]
        print(f"  vs {a:6s}: {w}胜 {t}平 {l}负")

    print("\n=== 平均运行时间（秒）===")
    time_sum = defaultdict(list)
    for key, algos in data.items():
        for a, runs in algos.items():
            time_sum[a].extend(x[2] for x in runs)
    for a in ALGOS:
        print(f"  {a:6s}: {np.mean(time_sum[a]):8.1f}s ± {np.std(time_sum[a]):.1f}")


if __name__ == "__main__":
    main()
