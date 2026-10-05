# -*- coding: utf-8 -*-
"""汇总静态辨识曲线（论文表 t17 数据源）"""
import csv
import numpy as np
from collections import defaultdict

d = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("experiments/results/plant_gain_static.csv")):
    d[(r["problem"], int(r["m"]))][float(r["phi"])].append(float(r["rho_tail"]))

RHO = 0.15
PHIS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
header = "inst".ljust(16) + " ".join(f"p{p:.2f}" for p in PHIS) + "  L_local"
print(header)
print("-" * len(header))
rows = []
for inst in sorted(d):
    mus = {p: np.mean(v) for p, v in d[inst].items()}
    sds = {p: np.std(v) for p, v in d[inst].items()}
    line = f"{inst[0]} m={inst[1]:<3}".ljust(16)
    line += " ".join(f"{mus[p]:.2f}" for p in PHIS)
    pts = sorted(mus.items())
    loc = None
    for (p1, r1), (p2, r2) in zip(pts, pts[1:]):
        if (r1 - RHO) * (r2 - RHO) <= 0 and p2 > p1:
            loc = (r2 - r1) / (p2 - p1)
            break
    line += "  " + ("--" if loc is None else f"{loc:.2f}")
    print(line)
    rows.append((inst, mus, sds, loc))

# 保存 LaTex 片段数据
with open("experiments/results/plant_gain_summary.txt", "w", encoding="utf-8") as f:
    for inst, mus, sds, loc in rows:
        f.write(f"{inst[0]},m={inst[1]}," +
                ",".join(f"{mus[p]:.4f}" for p in PHIS) + "," +
                ("NA" if loc is None else f"{loc:.3f}") + "\n")
print("\nsaved experiments/results/plant_gain_summary.txt")
