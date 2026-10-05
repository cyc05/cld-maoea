"""
统计严谨性补强：对主表（m3_results.csv）补充
  1) Benjamini--Hochberg FDR 多重比较校正（逐指标，32 实例 × 4 基线 = 128 检验/指标）
  2) Vargha--Delaney A^12 效应量（CLD vs 各基线，逐实例 + 汇总）

口径：IGD 与 IGD+ 均"越小越好"，A^12 = P(CLD < baseline) + 0.5*P(CLD == baseline)；
      A^12 > 0.5 表示 CLD 随机占优。

运行: python experiments/stat_rigor.py [--csv experiments/results/m3_results.csv]
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
BASELINES = ["NSGA3", "MOEAD", "RVEA", "CTAEA"]
METRICS = ["igd", "igd_plus"]


def load(csv_path):
    # (problem, m) -> algo -> {metric: [values]}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]))
            data[key][r["algo"]]["igd"].append(float(r["igd"]))
            data[key][r["algo"]]["igd_plus"].append(float(r["igd_plus"]))
    return data


def bh_fdr(pvals):
    """Benjamini--Hochberg：返回与输入等长的 q 值数组（NaN 保留）。"""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    m = np.sum(~np.isnan(p))
    if m == 0:
        return q
    idx = np.argsort(p)
    sorted_p = p[idx]
    # 累积最小：q_(i) = min_{j>=i} p_(j) * m / j
    q_sorted = np.empty(m)
    running = np.inf
    for i in range(m - 1, -1, -1):
        running = min(running, sorted_p[i] * m / (i + 1))
        q_sorted[i] = running
    q_sorted = np.minimum(q_sorted, 1.0)
    q[idx] = q_sorted
    return q


def a12(a, b):
    """Vargha--Delaney A^12：a 相对 b 的随机占优（越小越好口径）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lt = np.sum(a[:, None] < b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RESULT_CSV)
    args = ap.parse_args()

    data = load(args.csv)
    keys = sorted(data.keys(), key=lambda k: (k[1], k[0]))

    # 逐指标：收集全部 p 值与 A^12
    for metric in METRICS:
        pvec = []      # (label, p)
        a12_by_base = defaultdict(list)
        for (name, m) in keys:
            cld = np.array(data[(name, m)]["CLD"][metric])
            if len(cld) < 5:
                continue
            for b in BASELINES:
                other = np.array(data[(name, m)][b][metric])
                n = min(len(cld), len(other))
                try:
                    p = wilcoxon(cld[:n], other[:n]).pvalue
                except Exception:
                    p = 1.0
                pvec.append((f"{name} m={m} vs {b}", p))
                a12_by_base[b].append(a12(cld[:n], other[:n]))

        ps = np.array([x[1] for x in pvec])
        qs = bh_fdr(ps)

        print("\n" + "=" * 78)
        print(f"指标 {metric.upper()}：BH-FDR 校正（{len(ps)} 检验）")
        print("=" * 78)

        # 逐基线：FDR 后胜/平/负（胜 = CLD 均值更优 且 q<0.05；负 = 基线更优 且 q<0.05）
        # 审稿意见 4：补 median 差异与分布统计，而非仅依赖均值
        for b in BASELINES:
            win = tie = lose = 0
            med_pct = []     # 逐实例 median 相对差异 %
            for (label, p), q in zip(pvec, qs):
                if not label.endswith(b):
                    continue
                # 重新定位实例与方向
                name_m = label.split(" vs ")[0]
                name, mstr = name_m.split(" m=")
                m = int(mstr)
                cld = np.array(data[(name, m)]["CLD"][metric])
                other = np.array(data[(name, m)][b][metric])
                mc, mo = cld.mean(), other.mean()
                med_pct.append((np.median(cld) - np.median(other))
                               / (abs(np.median(other)) + 1e-12) * 100)
                if mc < mo and q < 0.05:
                    win += 1
                elif mc > mo and q < 0.05:
                    lose += 1
                else:
                    tie += 1
            aa = np.array(a12_by_base[b])
            mp = np.array(med_pct)
            print(f"  vs {b:6s}: FDR后 {win:2d}胜 {tie:2d}平 {lose:2d}负 | "
                  f"medianΔ% mean={mp.mean():+.2f} med={np.median(mp):+.2f} | "
                  f"A^12 = {aa.mean():.3f} [median {np.median(aa):.3f}, "
                  f"min {aa.min():.3f}, max {aa.max():.3f}] "
                  f"(>0.5 CLD 占优)")

        # 显著性翻转统计：原始 p<0.05 但 FDR 后 q>=0.05 的数量（报告哪些实例/基线失去显著性）
        flipped = [(label, p, q) for (label, p), q in zip(pvec, qs)
                   if p < 0.05 and q >= 0.05]
        print(f"\n  原始 p<0.05 但 FDR 后不显著的数量：{len(flipped)}/{int(np.sum(ps < 0.05))}")
        for label, p, q in flipped:
            print(f"    {label}: p={p:.4f} -> q={q:.4f}")


if __name__ == "__main__":
    main()