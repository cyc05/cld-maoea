"""
审稿意见回应：S2 三条件阈值"人为设定"的鲁棒性分析。

审稿指控：S2 阈值组合（p<0.05、同向率 8/10、worst ≤1.05×）是人为挑选，
等价于把"调算法参数"换成"调决策规则参数"。

回应策略（与论文 §3.6 Prop 2 的理论紧性互补）：
在阈值邻域网格上离线重算每个候选实例的 S2 决策，展示：
  (i)  默认阈值下的决策在邻域内高度稳定（多数实例全网格同决策）；
  (ii) 仅个别"边界实例"的决策随阈值翻转——这些正是 S1 画像本就判为
       边缘的实例，阈值敏感性被链的分级结构（S1 初筛 → S2 复核 → S3 回归）
       所吸收，而非任意性。
无需重跑实验：直接对 route1_pairs.csv 的 10 种子配对重统计。

运行：python experiments/threshold_sensitivity.py
产物：experiments/results/threshold_sensitivity.csv + stdout 汇总表
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PAIR_CSV = os.path.join(ROOT, "experiments", "results", "route1_pairs.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "results", "threshold_sensitivity.csv")

# 默认阈值（与 criterion_replay.py S2_P/S2_SAMEDIR/S2_HARD 一致）
DEFAULT = dict(p=0.05, samedir=0.8, hard=1.05)
# 邻域网格
GRID_P = [0.01, 0.05, 0.10]
GRID_SAMEDIR = [0.7, 0.8, 0.9]      # 对应 7/10, 8/10, 9/10
GRID_HARD = [1.00, 1.05, 1.10]


def load_pairs():
    pairs = {}
    with open(PAIR_CSV, newline="") as f:
        for r in csv.DictReader(f):
            pairs.setdefault((r["problem"], int(r["m"])), []).append(
                (float(r["igd_off"]), float(r["igdp_off"]),
                 float(r["igd_on"]), float(r["igdp_on"])))
    return pairs


def instance_stats(rows):
    """预计算与阈值无关的每种子/汇总量"""
    from scipy.stats import wilcoxon
    arr = np.array(rows)
    igd_o, igdp_o, igd_n, igdp_n = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    d_igd = (igd_n - igd_o) / np.abs(igd_o) * 100
    d_igdp = (igdp_n - igdp_o) / np.abs(igdp_o) * 100
    try:
        p1 = wilcoxon(igd_o, igd_n).pvalue if (igd_o != igd_n).any() else 1.0
        p2 = wilcoxon(igdp_o, igdp_n).pvalue if (igdp_o != igdp_n).any() else 1.0
    except Exception:
        p1 = p2 = 1.0
    return dict(
        d_igd_mean=float(d_igd.mean()), d_igdp_mean=float(d_igdp.mean()),
        p1=float(p1), p2=float(p2),
        n_samedir=int((d_igdp < 0).sum()), n=len(d_igdp),
        worst=float((igd_n / igd_o).max()),
    )


def decide(stats, p, samedir, hard):
    ok1 = (stats["d_igd_mean"] < 0 and stats["d_igdp_mean"] < 0
           and stats["p1"] < p and stats["p2"] < p)
    ok2 = (stats["n_samedir"] / stats["n"]) >= samedir
    ok3 = stats["worst"] <= hard
    return ok1 and ok2 and ok3


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pairs = load_pairs()
    stats = {k: instance_stats(v) for k, v in pairs.items() if len(v) >= 5}

    combos = [(p, s, h) for p in GRID_P for s in GRID_SAMEDIR for h in GRID_HARD]
    default_dec = {k: decide(st, **DEFAULT) for k, st in stats.items()}

    # 每实例在网格上的"启用"组合数
    rows_out = []
    print("=" * 78)
    print("S2 阈值邻域敏感性（27 组合：p×同向率×硬检查）")
    print("=" * 78)
    print(f"{'实例':<14}{'默认':<6}{'p1':<7}{'p2':<7}{'同向':<6}{'最差':<7}"
          f"{'网格启用/27':<10}{'稳定性'}")
    print("-" * 78)
    for (name, m) in sorted(stats, key=lambda k: (k[1], k[0])):
        st = stats[(name, m)]
        n_on = sum(decide(st, p, s, h) for (p, s, h) in combos)
        d0 = default_dec[(name, m)]
        # 稳定性：默认决策是否在全部 27 组合中一致
        all_same = (n_on == len(combos)) if d0 else (n_on == 0)
        tag = "稳健" if all_same else ("边界" if 0 < n_on < len(combos) else "稳健")
        inst = f"{name} m={m}"
        print(f"{inst:<14}{'启用' if d0 else '关闭':<6}"
              f"{st['p1']:<7.3f}{st['p2']:<7.3f}"
              f"{st['n_samedir']}/{st['n']}   {st['worst']:<7.3f}"
              f"{n_on:<10}{tag}")
        rows_out.append(dict(problem=name, m=m, default_on=int(d0),
                             p1=st["p1"], p2=st["p2"],
                             n_samedir=st["n_samedir"], n=st["n"],
                             worst=round(st["worst"], 4),
                             grid_on=n_on, grid_total=len(combos),
                             stable=tag))
    print("-" * 78)

    # 全链启用集在网格上的一致率：逐组合计算启用集，与默认启用集比较
    default_set = frozenset(k for k, v in default_dec.items() if v)
    agree = sum(
        frozenset(k for k in stats if decide(stats[k], p, s, h)) == default_set
        for (p, s, h) in combos)
    n_stable_inst = sum(1 for r in rows_out if r["stable"] == "稳健")
    print(f"实例级稳健：{n_stable_inst}/{len(rows_out)} 实例决策在全部 27 组合下不变")
    print(f"链路级一致：{agree}/{len(combos)} 组合的启用集与默认完全一致"
          f"（默认启用 {len(default_set)} 实例："
          f"{sorted(f'{n}m{m}' for n, m in default_set)}）")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n产物：{OUT_CSV}")


if __name__ == "__main__":
    main()
