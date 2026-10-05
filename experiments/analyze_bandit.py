"""
审稿意见回应：PD 闭环控制 vs FRRMAB 在线 bandit 选择的对比分析。

对比设计（同一被控量上的范式对比）：
  * CLD       ：phi 由 PD 闭环控制器连续调节（本文方法）；
  * CLDBandit ：phi 改由 FRRMAB（滑窗 FRR + UCB）在离散档位 {0.15,...,0.5}
                间在线选择，其余机制（方向分化/逃逸/R2 截断/archive）完全相同。
两者唯一差异是"phi 由谁在线决定"，因此该对比直接检验：对于支配强度这一
在线自适应参数，PD 闭环控制是否不亚于通用的 bandit 在线选择。

口径：10 个评估种子配对，IGD 与 IGD+（越小越好），Wilcoxon 符号秩 + A^12。

运行：python experiments/analyze_bandit.py [--csv experiments/results/m3_results.csv]
产物：experiments/results/bandit_compare.csv + stdout 汇总
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "results", "bandit_compare.csv")
METRICS = ["igd", "igd_plus"]


def load(csv_path):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]))
            data[key][r["algo"]]["igd"][int(r["seed"])] = float(r["igd"])
            data[key][r["algo"]]["igd_plus"][int(r["seed"])] = float(r["igd_plus"])
    return data


def a12(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :]); eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=RESULT_CSV)
    args = ap.parse_args()
    from scipy.stats import wilcoxon

    data = load(args.csv)
    keys = sorted([k for k, v in data.items()
                   if "CLD" in v and "CLDBandit" in v],
                  key=lambda k: (k[1], k[0]))

    for metric in METRICS:
        print("\n" + "=" * 82)
        print(f"CLD（PD 闭环）vs CLDBandit（FRRMAB 在线选 phi）— 指标 {metric.upper()}"
              f"（越小越好；负差 = PD 更优）")
        print("=" * 82)
        print(f"{'实例':<14}{'PD mean':<11}{'Bandit mean':<12}{'Δ%':<9}{'p':<8}"
              f"{'A12(PD<Bandit)':<15}{'判定'}")
        print("-" * 82)
        win = tie = lose = 0
        a12s, rows = [], []
        for (name, m) in keys:
            c = data[(name, m)]["CLD"][metric]
            bd = data[(name, m)]["CLDBandit"][metric]
            common = sorted(set(c) & set(bd))
            if len(common) < 5:
                continue
            cv = np.array([c[s] for s in common])
            bv = np.array([bd[s] for s in common])
            try:
                p = wilcoxon(cv, bv).pvalue if (cv != bv).any() else 1.0
            except Exception:
                p = 1.0
            d = (cv.mean() - bv.mean()) / (abs(bv.mean()) + 1e-12) * 100
            a = a12(cv, bv)          # P(PD < Bandit)，>0.5 表示 PD 占优
            a12s.append(a)
            if cv.mean() < bv.mean() and p < 0.05:
                verdict, win = "PD 优", win + 1
            elif cv.mean() > bv.mean() and p < 0.05:
                verdict, lose = "Bandit 优", lose + 1
            else:
                verdict, tie = "平", tie + 1
            print(f"{name} m={m:<4}{cv.mean():<11.4f}{bv.mean():<12.4f}"
                  f"{d:<+8.1f}%{p:<8.3f}{a:<15.3f}{verdict}")
            rows.append(dict(problem=name, m=m, metric=metric,
                             pd_mean=cv.mean(), bandit_mean=bv.mean(),
                             delta_pct=d, p=p, a12=a, verdict=verdict))
        a12s = np.array(a12s)
        print("-" * 82)
        print(f"汇总（PD 相对 Bandit）：胜 {win} / 平 {tie} / 负 {lose} | "
              f"A^12 mean={a12s.mean():.3f} median={np.median(a12s):.3f}")
        print("（胜 = PD 闭环显著优于 bandit 在线选择；胜多 → PD 闭环不弱于在线 bandit）")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n产物：{OUT_CSV}")


if __name__ == "__main__":
    main()
