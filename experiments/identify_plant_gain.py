# -*- coding: utf-8 -*-
"""
理论分析配套：植物静态增益辨识（闭环断开，fixed_phi 扫描）。

植物模型: rho_{t+1} = g(phi_t) + w_t，phi↑ → CDAS 支配区缩小 → 更难被支配
→ rank-1 占比 ρ↑（g 增）。闭环线性化斜率 L = g'(phi*) 满足精确 Jury 稳定性判据
  k_d L < 1  且  L (k_i + 2 k_d) < 2
默认 (k_i, k_d) = (0.5, 0.1) 时 L* = min{1/k_d, 2/(k_i+2k_d)} ≈ 2.857。

辨识协议（消除反馈内生性）：
  对固定 phi ∈ {0.15,...,0.50} 直接运行（fixed_phi，控制器旁路），
  取运行后期（最后 25% FE）的 ρ 均值作为稳态 g(phi) 的估计；
  3 种子取均值。局部斜率 = 跨越 rho*=0.15 的相邻网格点割线；
  上界斜率 = 操作区间 [0.15, 0.40] 内最大割线斜率。

运行: python experiments/identify_plant_gain.py --workers 8
"""
import os
import sys
import csv
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024]
FE_MAX = 100000
PHI_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
RHO_STAR = 0.15

# 代表性实例：dtlz3(多峰) / dtlz6(欺骗) / dtlz7(不连通) / wfg3(退化)
INSTANCES = [
    ("dtlz3", "dtlz", 5),
    ("dtlz6", "dtlz", 10),
    ("dtlz7", "dtlz", 10),
    ("wfg3", "wfg", 10),
]

KI, KD = 0.5, 0.1


def solve_L_star(ki=KI, kd=KD):
    """精确 Jury 稳定性边界: L < min{1/k_d, 2/(k_i + 2k_d)}。"""
    return min(1.0 / kd, 2.0 / (ki + 2.0 * kd))


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem
    from src.algorithm import CLDMaOEA

    name, ptype, m, phi, seed = task
    problem = make_problem(name, m)
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=ptype, problem_name=name,
                   fixed_phi=phi)
    alg.run(fe_max=FE_MAX)
    rho = np.asarray(alg.history["rho"], dtype=float)
    tail = rho[int(0.75 * len(rho)):]
    return name, m, phi, seed, float(np.mean(tail)), float(np.std(tail))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_CSV = os.path.join(ROOT, "experiments", "results", "plant_gain_static.csv")

    tasks = [(n, t, m, phi, s) for (n, t, m) in INSTANCES
             for phi in PHI_GRID for s in SEEDS]
    print(f"静态辨识：{len(INSTANCES)} 实例 × {len(PHI_GRID)} 个 φ × "
          f"{len(SEEDS)} 种子 = {len(tasks)} 次运行, workers={args.workers}",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = []
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["problem"], int(r["m"]), float(r["phi"]), int(r["seed"]))
                done.add(key)
                results.append((key[0], key[1], key[2], key[3],
                                float(r["rho_tail"]), float(r["rho_std"])))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)
    todo = [t for t in tasks
            if (t[0], t[2], t[3], t[4]) not in
            {(r[0], r[1], r[2], r[3]) for r in results}]
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_task, t): t for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    name, m, phi, seed, mu, sd = fut.result()
                    results.append((name, m, phi, seed, mu, sd))
                    print(f"[done] {name} m={m} phi={phi} seed={seed} "
                          f"rho_tail={mu:.4f}±{sd:.4f}", flush=True)
                except Exception as e:
                    print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["problem", "m", "phi", "seed", "rho_tail", "rho_std"])
            for r in sorted(results):
                w.writerow(r)
        print(f"\n明细已存 {OUT_CSV}")

    # ---- 分析
    L_star = solve_L_star()
    print(f"\n稳定性边界 L*(k_i={KI}, k_d={KD}) = {L_star:.3f}")
    print(f"{'instance':>14} | {'L_local@rho*':>12} | {'L_sup[0.15,0.40]':>16} | "
          f"{'margin(sup)':>11} | stable")
    print("-" * 74)
    summary = {}
    for (name, ptype, m) in INSTANCES:
        pts = []
        for phi in PHI_GRID:
            vals = [r[4] for r in results
                    if r[0] == name and r[1] == m and r[2] == phi]
            if vals:
                pts.append((phi, float(np.mean(vals))))
        pts.sort()
        if len(pts) < 2:
            print(f"{name} m={m}: 数据不足")
            continue
        # 局部斜率：跨越 rho* 的相邻网格割线
        L_local = None
        for (p1, r1), (p2, r2) in zip(pts, pts[1:]):
            if (r1 - RHO_STAR) * (r2 - RHO_STAR) <= 0 and p2 > p1:
                L_local = (r2 - r1) / (p2 - p1)
                break
        # 操作区间上界斜率
        op = [(p, r) for (p, r) in pts if 0.15 <= p <= 0.40]
        L_sup = max((r2 - r1) / (p2 - p1)
                    for (p1, r1), (p2, r2) in zip(op, op[1:])) if len(op) > 1 else float("nan")
        margin = L_star / L_sup if L_sup and L_sup > 0 else float("inf")
        ok = (L_sup is not None and np.isfinite(L_sup) and L_sup < L_star)
        summary[(name, m)] = (L_local, L_sup, margin)
        print(f"{name:>9} m={m:<3} | "
              f"{'—' if L_local is None else f'{L_local:12.3f}'} | "
              f"{L_sup:>16.3f} | {margin:>11.2f} | {'yes' if ok else 'NO'}")

    print("\n说明: L_local = 稳态曲线跨越 ρ*=0.15 处的割线斜率（局部增益）；")
    print("L_sup = 操作区间 φ∈[0.15,0.40] 内最大割线斜率（最坏情况增益）。")
    print(f"定理 1 判据（精确 Jury 条件）: k_d·L < 1 且 L·(k_i + 2k_d) < 2；默认增益下 L* ≈ {L_star:.3f}。")


if __name__ == "__main__":
    main()
