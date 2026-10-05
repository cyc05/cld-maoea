"""
数据窥探治愈实验：用与评估种子完全不相交的一批种子重跑 S2 同进程验证，
验证判据链 on/off 决策在 held-out 种子上的稳定性（决策规则是否有过拟合）。

口径与 criterion_replay.py stage2 完全一致（10 种子同进程 off/on 配对，
off 显式 corner_archive=False，on 用 K.26 落地款 direction+replace+budget 0.1；
S2 三条件 = 双指标 Wilcoxon p<0.05 + IGD+ 同向率≥8/10 + 单种子 IGD 恶化≤1.05×）。
仅替换种子集为不相交批次。只覆盖 7 个 S1 阳性候选实例（其余 25 例 S1 即阴性）。

运行: python experiments/s2_disjoint.py [--workers 8]
产物: 打印判定表 + 与 K.26 启用集（wfg2 m=10, dtlz5 m=5/10, dtlz7 m=10）一致率。
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 不相交种子批：与评估集 [42,7,2024,314,99,1234,555,8888,271,640] 无交集
SEEDS_B = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
FE_MAX = 100000
S2_P = 0.05
S2_SAMEDIR = 0.8
S2_HARD = 1.05

INSTANCES = [
    ("wfg3", 5), ("wfg2", 10), ("dtlz6", 10),
    ("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10), ("dtlz7", 5),
]
K26_ENABLED = {("wfg2", 10), ("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10)}


def run_pair_task(task):
    """同进程 off→on 配对（与 criterion_replay.run_pair_task 同款）。"""
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA

    name, m, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    pf = get_pf(name, m)

    res = {}
    for tag, arc in (("off", False), ("on", True)):
        alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                       problem_type=ptype, problem_name=name,
                       corner_archive=arc, archive_policy="direction",
                       archive_out_mode="replace", archive_budget=0.1)
        _, F = alg.run(fe_max=FE_MAX)
        res[tag] = (float(IGD(pf)(F)), float(IGDPlus(pf)(F)))
    return name, m, seed, res["off"][0], res["off"][1], res["on"][0], res["on"][1]


def main():
    import argparse
    from scipy.stats import wilcoxon
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(n, m, s) for (n, m) in INSTANCES for s in SEEDS_B]
    print(f"disjoint-seed S2：{len(INSTANCES)} 候选实例 × {len(SEEDS_B)} 种子"
          f"（同进程配对），workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_pair_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                name, m, seed, io, po, inn, pn = fut.result()
                results.setdefault((name, m), []).append((seed, io, po, inn, pn))
                print(f"[done] {name} m={m} seed={seed} IGD {io:.4f}->{inn:.4f} "
                      f"IGD+ {po:.4f}->{pn:.4f}", flush=True)
            except Exception as e:
                print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)

    print("\n" + "=" * 96)
    print(f"| 实例 | S2① 双指标显著 | S2② 同向率 | S2③ 硬检查 | 自动决策 | K.26 | 一致 |")
    print("|" + "-" * 94 + "|")
    agree = n = 0
    for (name, m) in INSTANCES:
        rows = results.get((name, m), [])
        if len(rows) < len(SEEDS_B):
            print(f"| {name} m={m} | 数据不足 {len(rows)}/{len(SEEDS_B)} | | | | | |")
            continue
        arr = np.array([r[1:] for r in sorted(rows)])
        igd_o, igdp_o, igd_n, igdp_n = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        d_igd = (igd_n - igd_o) / np.abs(igd_o) * 100
        d_igdp = (igdp_n - igdp_o) / np.abs(igdp_o) * 100
        try:
            p1 = wilcoxon(igd_o, igd_n).pvalue if (igd_o != igd_n).any() else 1.0
            p2 = wilcoxon(igdp_o, igdp_n).pvalue if (igdp_o != igdp_n).any() else 1.0
        except Exception:
            p1 = p2 = 1.0
        ok1 = (d_igd.mean() < 0 and d_igdp.mean() < 0 and p1 < S2_P and p2 < S2_P)
        rate = float((d_igdp < 0).mean())
        ok2 = rate >= S2_SAMEDIR
        worst = float((igd_n / igd_o).max())
        ok3 = worst <= S2_HARD
        auto = "启用" if (ok1 and ok2 and ok3) else "关闭"
        k26 = "启用" if (name, m) in K26_ENABLED else "关闭"
        agree += (auto == k26); n += 1
        print(f"| {name} m={m} | {'✓' if ok1 else '✗'}（ΔIGD {d_igd.mean():+.1f}% "
              f"p={p1:.3f} / ΔIGD+ {d_igdp.mean():+.1f}% p={p2:.3f}） | "
              f"{'✓' if ok2 else '✗'}（{int((d_igdp < 0).sum())}/{len(d_igdp)}） | "
              f"{'✓' if ok3 else '✗'}（最差 {worst:.2f}×） | **{auto}** | {k26} | "
              f"{'✓' if auto == k26 else '✗'} |")
    print("=" * 96)
    print(f"disjoint 判定与 K.26 一致率：{agree}/{n}")


if __name__ == "__main__":
    main()