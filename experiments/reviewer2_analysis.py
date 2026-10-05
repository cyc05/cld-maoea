"""
第二轮审稿意见的统计分析（复用现有数据，无需重跑实验）。

产出三组支撑数据：
  [意见4] IGD 与 IGD+ 在 S2 配对中的逐种子相关性（Spearman/Pearson）——
          量化 S2 条件1"双指标同时显著"的实际保守度。
  [意见5] S2 决策对种子选择的 bootstrap 稳定性——对每个 S2 候选实例，
          从 10 个评估种子有放回重采样，重算三条件决策，报告启用比例；
          回应"32/32 是否为特定种子集的巧合"。
  [意见2] 每 1000 次函数评价的平均墙钟时间（ms/1e3 evals）——
          将计算代价从"总墙钟"归一化到"每 FE"，配合复杂度讨论。

运行：python experiments/reviewer2_analysis.py
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_CSV = os.path.join(ROOT, "experiments", "results", "route1_pairs.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")

S2_P, S2_SAMEDIR, S2_HARD = 0.05, 0.8, 1.05
N_BOOT = 2000
rng = np.random.default_rng(20260903)


def load_pairs():
    pairs = defaultdict(list)
    with open(PAIR_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pairs[(r["problem"], int(r["m"]))].append(
                (float(r["igd_off"]), float(r["igdp_off"]),
                 float(r["igd_on"]), float(r["igdp_on"])))
    return pairs


def s2_decision(rows):
    """三条件判定（与 criterion_replay 一致）。rows: list of 4-tuple.
    bootstrap 内用 method='approx'（正态近似）避免 exact 枚举的高开销。"""
    from scipy.stats import wilcoxon
    arr = np.array(rows)
    igd_o, igdp_o, igd_n, igdp_n = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    d_igd = (igd_n - igd_o) / np.abs(igd_o) * 100
    d_igdp = (igdp_n - igdp_o) / np.abs(igdp_o) * 100
    try:
        p1 = (wilcoxon(igd_o, igd_n, method="approx").pvalue
              if (igd_o != igd_n).any() else 1.0)
        p2 = (wilcoxon(igdp_o, igdp_n, method="approx").pvalue
              if (igdp_o != igdp_n).any() else 1.0)
    except Exception:
        p1 = p2 = 1.0
    ok1 = (d_igd.mean() < 0 and d_igdp.mean() < 0 and p1 < S2_P and p2 < S2_P)
    ok2 = (d_igdp < 0).mean() >= S2_SAMEDIR
    ok3 = (igd_n / igd_o).max() <= S2_HARD
    return ok1 and ok2 and ok3


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pairs = load_pairs()

    # ---------------- 意见4：IGD vs IGD+ 相关性 ----------------
    from scipy.stats import spearmanr, pearsonr
    print("=" * 70)
    print("[意见4] S2 候选实例上 ΔIGD% 与 ΔIGD+% 的逐种子相关性")
    print("=" * 70)
    all_sp, all_pe = [], []
    for (name, m), rows in sorted(pairs.items()):
        arr = np.array(rows)
        d_igd = (arr[:, 2] - arr[:, 0]) / np.abs(arr[:, 0]) * 100
        d_igdp = (arr[:, 3] - arr[:, 1]) / np.abs(arr[:, 1]) * 100
        sp = spearmanr(d_igd, d_igdp).statistic
        pe = pearsonr(d_igd, d_igdp).statistic
        all_sp.append(sp); all_pe.append(pe)
        print(f"  {name} m={m:<3} Spearman={sp:+.3f}  Pearson={pe:+.3f}")
    print(f"  跨实例均值: Spearman={np.mean(all_sp):+.3f}  "
          f"Pearson={np.mean(all_pe):+.3f}")
    print("  → 两指标高度正相关时，条件1的'双指标同时显著'接近单一检验，")
    print("    实际激活错误率远低于名义 5%（保守），而非 0.05² 的独立情形。")

    # ---------------- 意见5：S2 决策 bootstrap 稳定性 ----------------
    print("\n" + "=" * 70)
    print("[意见5] S2 决策对种子选择的 bootstrap 稳定性（%d 次重采样）" % N_BOOT)
    print("=" * 70)
    print(f"  {'实例':<13}{'评估种子决策':<12}{'bootstrap启用比例':<18}{'稳定性'}")
    for (name, m), rows in sorted(pairs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows = list(rows)
        n = len(rows)
        full = s2_decision(rows)
        boot_on = 0
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, size=n)
            if s2_decision([rows[i] for i in idx]):
                boot_on += 1
        frac = boot_on / N_BOOT
        stab = "稳定" if (frac > 0.95 or frac < 0.05) else ("边缘" if 0.2 < frac < 0.8 else "较稳")
        print(f"  {name} m={m:<4}{'启用' if full else '禁用':<12}"
              f"{frac:<18.3f}{stab}")

    # ---------------- 意见2：每 1000 evals 的墙钟时间 ----------------
    print("\n" + "=" * 70)
    print("[意见2] 每 1000 次函数评价的平均墙钟时间（ms/1e3 evals, FE=1e5）")
    print("=" * 70)
    times = defaultdict(list)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            times[r["algo"]].append(float(r["time"]))
    for algo in ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA"]:
        if algo in times:
            t = np.array(times[algo])
            per1k = t / 100.0   # time/1e5 * 1e3 = time/100
            print(f"  {algo:6s}: {per1k.mean():.2f} ms/1e3 evals "
                  f"(total {t.mean():.1f}s, ratio vs CLD n/a)")


if __name__ == "__main__":
    main()
