"""
K.18 后续：退化题根因诊断。对比"手调 div_eps（立即生效）" vs
"div_eps_auto（预热期后标定）"，确认预热延迟是否导致 wfg9 m=10 /
dtlz3 m=5 退化。

运行: python experiments/k18_escape_diag.py [--workers 8] [--seeds 3]
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS_ALL = [42, 7, 2024]
FE_MAX = 100000

# (name, m, 手调 div_eps, H 基线)
CASES = [
    ("wfg9", 10, 0.015, 3.9429),
    ("dtlz3", 5, 0.01, 0.3461),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from src.algorithm import CLDMaOEA
    name, m, seed, mode, manual_deps = task
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    if mode == "manual":
        # 手调：禁用 auto，固定 div_eps，从第一代生效
        alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                       problem_type=problem_type, problem_name=name,
                       div_eps_auto=False, div_eps=manual_deps)
    else:
        # 自动标定：预热期后设置 div_eps
        alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                       problem_type=problem_type, problem_name=name,
                       div_eps_auto=True)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    return name, m, seed, mode, igd, float(alg.controller.div_eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS_ALL[:args.seeds]

    tasks = [(n, m, s, mode, md)
             for n, m, md, _ in CASES for mode in ("manual", "auto")
             for s in seeds]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                name, m, seed, mode, igd, deps = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results.setdefault((name, m, mode), []).append((igd, deps))

    print("\n" + "=" * 80)
    for name, m, manual_deps, h_base in CASES:
        print(f"\n{name} m={m}  (H 基线 {h_base:.4f}, 手调 div_eps={manual_deps})")
        for mode in ("manual", "auto"):
            vals = results.get((name, m, mode), [])
            if not vals:
                continue
            igds = [x[0] for x in vals]
            deps = [x[1] for x in vals]
            mu = np.mean(igds)
            pct = (mu - h_base) / abs(h_base) * 100
            print(f"  {mode:>6}: IGD={mu:.4f}  div_eps={np.mean(deps):.5g}  "
                  f"{pct:+.1f}%")


if __name__ == "__main__":
    main()
