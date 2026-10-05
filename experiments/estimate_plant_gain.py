# -*- coding: utf-8 -*-
"""
理论分析配套：植物增益 L 的经验估计（PD 环稳定性余量验证）。

植物模型: rho_t = g(phi_{t-1}) + w_t（phi_t 在 t+1 代应用，单步滞后）。
线性化斜率 L = g'(phi*) 的估计：在活跃门控期（gamma=1、非逃逸）取
相邻两代的差分对 (Delta phi_{t-1}, Delta rho_t)，L_hat = median(Delta rho / Delta phi)。

稳定性定理（论文 Prop. 1，精确 Jury 条件）: k_d L < 1 且 L(k_i + 2k_d) < 2。
默认 (k_i, k_d) = (0.5, 0.1) 时 L* = min{1/k_d, 2/(k_i+2k_d)} ≈ 2.857。

运行: python experiments/estimate_plant_gain.py
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(ROOT, "experiments", "results", "history", "phi_rho_trace.csv")

KI, KD = 0.5, 0.1
DEAD_ZONE = 0.05
MIN_DPHI = 0.01  # 过滤过小的 phi 变化（信噪比不足）


def solve_L_star(ki=KI, kd=KD):
    """精确 Jury 稳定性边界: L < min{1/k_d, 2/(k_i + 2k_d)}。"""
    return min(1.0 / kd, 2.0 / (ki + 2.0 * kd))


def load_traces(path):
    """(problem, m, seed) -> 按代排序的 (phi, rho, gamma, escape) 列表"""
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]), int(r["seed"]))
            rows[key].append((float(r["phi"]), float(r["rho"]),
                              float(r["gamma"]), float(r["escape"])))
    return rows


def estimate(rows):
    """逐 (problem, m) 汇总活跃期差分比，返回 {inst: ratios}, {inst: L_hat}"""
    inst_ratios = defaultdict(list)
    for (name, m, seed), tr in sorted(rows.items()):
        tr.sort(key=lambda x: x[0])  # 已按代序；fe 不在行内，保持读入顺序即可
        # 行 t: (phi_t, rho_t, gamma_t, esc_t)；rho_t 由 phi_{t-1} 产生
        for t in range(2, len(tr)):
            phi_applied_prev = tr[t - 2][0]          # phi_{t-2}
            phi_applied = tr[t - 1][0]               # phi_{t-1}（产生 rho_t）
            dphi = phi_applied - phi_applied_prev
            drho = tr[t][1] - tr[t - 1][1]
            gamma_prev, esc_prev = tr[t - 1][2], tr[t - 1][3]
            if gamma_prev != 1.0 or esc_prev != 0.0:
                continue                              # 仅活跃控制期
            if abs(dphi) < MIN_DPHI:
                continue                              # 过小变化（含截断饱和）
            e_prev = tr[t - 1][1] - 0.15
            if abs(e_prev) <= DEAD_ZONE:
                continue                              # 死区：控制量为零
            inst_ratios[(name, m)].append(drho / dphi)
    hats = {}
    for inst, ratios in inst_ratios.items():
        hats[inst] = float(np.median(ratios))
    return inst_ratios, hats


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.path.exists(TRACE):
        print(f"轨迹文件不存在: {TRACE}")
        return
    L_star = solve_L_star()
    print(f"稳定性边界 L*(k_i={KI}, k_d={KD}) = {L_star:.3f}\n")
    rows = load_traces(TRACE)
    inst_ratios, hats = estimate(rows)
    print(f"{'instance':>14} | {'n_diff':>6} | {'L_hat(median)':>13} | "
          f"{'IQR':>17} | {'margin L*/L':>11} | stable")
    print("-" * 78)
    all_hats = []
    for inst in sorted(hats, key=lambda k: (k[1], k[0])):
        arr = np.asarray(inst_ratios[inst])
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        lh = hats[inst]
        all_hats.append(lh)
        margin = L_star / lh if lh > 0 else float("inf")
        print(f"{inst[0]:>9} m={inst[1]:<3} | {len(arr):>6} | {lh:>13.3f} | "
              f"[{q1:>7.3f},{q3:>7.3f}] | {margin:>11.2f} | "
              f"{'yes' if lh < L_star else 'NO'}")
    print("-" * 78)
    if all_hats:
        lh_max = max(all_hats)
        print(f"max L_hat = {lh_max:.3f}  →  margin = {L_star / lh_max:.2f}x  "
              f"({'全部在稳定域内' if lh_max < L_star else '超出稳定域！'})")

    out_csv = os.path.join(ROOT, "experiments", "results", "plant_gain_estimates.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem", "m", "n_diffs", "L_hat", "q1", "q3", "L_star", "margin"])
        for inst in sorted(hats, key=lambda k: (k[1], k[0])):
            arr = np.asarray(inst_ratios[inst])
            q1, q3 = np.quantile(arr, [0.25, 0.75])
            lh = hats[inst]
            w.writerow([inst[0], inst[1], len(arr), f"{lh:.4f}",
                        f"{q1:.4f}", f"{q3:.4f}", f"{L_star:.3f}",
                        f"{L_star / lh:.2f}" if lh > 0 else "inf"])
    print(f"\n明细已存 {out_csv}")


if __name__ == "__main__":
    main()
