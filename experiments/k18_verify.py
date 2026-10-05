"""
K.18 全量回归验证：div_eps 自动标定（M 版）vs H 版基线。
并行跑 22 题 x 2 版本 x 3 种子，对比 K.15 基线（9胜0负13平，M 平均 2.4106）。

运行: python experiments/k18_verify.py [--workers 8] [--seeds 3]
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

CASES = (
    [("dtlz3", 5), ("dtlz3", 10), ("dtlz1", 5), ("dtlz1", 10)]
    + [(f"wfg{i}", m) for i in range(1, 10) for m in (5, 10)]
)


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant
    name, m, seed, vname = task
    variant = CLDVariant.HIGH_DIM if vname == "H" else CLDVariant.MULTI_MODAL
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant=variant,
                   problem_type=problem_type, problem_name=name)
    t0 = time.perf_counter()
    X, F = alg.run(fe_max=FE_MAX)
    el = time.perf_counter() - t0
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    return name, m, vname, seed, igd, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS_ALL[:args.seeds]

    tasks = [(name, m, s, v)
             for name, m in CASES for v in ("H", "M") for s in seeds]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                name, m, vname, seed, igd, el = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results.setdefault((name, m, vname), []).append(igd)
            print(f"[done] {name} m={m} {vname} seed={seed} igd={igd:.4f}",
                  flush=True)

    print("\n" + "=" * 84)
    header = f"{'case':<12} {'H IGD':>20} {'M IGD':>20} {'变化%':>8}  判定"
    print(header)
    print("-" * 84)

    n_win = n_loss = n_tie = 0
    h_all, m_all = [], []
    for name, m in CASES:
        h = results.get((name, m, "H"), [])
        mm = results.get((name, m, "M"), [])
        if not h or not mm:
            continue
        h_mu, m_mu = np.mean(h), np.mean(mm)
        h_all.append(h_mu)
        m_all.append(m_mu)
        pct = (m_mu - h_mu) / (abs(h_mu) + 1e-12) * 100
        if pct < -5:
            verdict, n_win = "✓ 改善", n_win + 1
        elif pct > 5:
            verdict, n_loss = "✗ 退化", n_loss + 1
        else:
            verdict, n_tie = "≈ 持平", n_tie + 1
        print(f"{name}{m:<6} {h_mu:>18.4f} {m_mu:>18.4f} "
              f"{pct:>+7.1f}%  {verdict}")

    print("-" * 84)
    print(f"版本平均 IGD: H={np.mean(h_all):.4f}  M={np.mean(m_all):.4f}")
    print(f"判定统计（M 较 H）: {n_win} 胜 / {n_loss} 负 / {n_tie} 平")
    print(f"总耗时 {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
