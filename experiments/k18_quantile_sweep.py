"""
K.18 后续：div_eps_quantile 敏感性扫描。
对 8 个逃逸 override 题，扫描 quantile ∈ {0.15, 0.25, 0.35, 0.5}，
记录 M 版 IGD 与自动标定的 div_eps 值，对比 H 基线判断最优分位数。

运行: python experiments/k18_quantile_sweep.py [--workers 8] [--seeds 3]
"""
import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS_ALL = [42, 7, 2024]
FE_MAX = 100000
QUANTILES = [0.15, 0.25, 0.35, 0.5]

# (name, m, H 基线 IGD) —— 来自 k18_verify
CASES = [
    ("dtlz3", 5, 0.3461),
    ("dtlz3", 10, 0.7262),
    ("dtlz1", 10, 0.2970),
    ("wfg2", 5, 1.4952),
    ("wfg2", 10, 7.7947),
    ("wfg3", 5, 1.4695),
    ("wfg7", 10, 3.3494),
    ("wfg9", 10, 3.9429),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from src.algorithm import CLDMaOEA
    name, m, seed, q = task
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=problem_type, problem_name=name,
                   div_eps_quantile=q)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    return name, m, seed, q, igd, float(alg.controller.div_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS_ALL[:args.seeds]

    tasks = [(n, m, s, q)
             for n, m, _ in CASES for q in QUANTILES for s in seeds]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                name, m, seed, q, igd, deps = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results.setdefault((name, m, q), []).append(igd)

    print("\n" + "=" * 96)
    # 对每个题，打印各 quantile 的 IGD 与变化%
    for name, m, h_base in CASES:
        print(f"\n{name} m={m}  (H 基线 {h_base:.4f})")
        print(f"{'quantile':>10} {'M IGD':>18} {'div_eps':>10} {'变化%':>8}  判定")
        best = None
        best_igd = float("inf")
        for q in QUANTILES:
            vals = results.get((name, m, q), [])
            if not vals:
                continue
            mu = np.mean(vals)
            pct = (mu - h_base) / (abs(h_base) + 1e-12) * 100
            if pct < -5:
                verdict = "✓ 改善"
            elif pct > 5:
                verdict = "✗ 退化"
            else:
                verdict = "≈ 持平"
            # div_eps 近似：取该 quantile 下第一次标定值（种子相关，仅示意）
            print(f"{q:>10.2f} {mu:>18.4f} {'--':>10} {pct:>+7.1f}%  {verdict}")
            if mu < best_igd:
                best_igd = mu
                best = q
        print(f"  最优 quantile = {best} (IGD {best_igd:.4f})")


if __name__ == "__main__":
    main()
