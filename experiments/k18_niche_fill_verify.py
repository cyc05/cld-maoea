"""
K.17 回滚后的 niche_fill 精细启用验证（仅 r2 层）。
对无逃逸凹 PF 题（wfg4/5/6 m=10）期望 IGD+ 改善；
对逃逸题（wfg2 m=10 / wfg3 m=5 / wfg9 m=10）期望不退化。

运行: python experiments/k18_niche_fill_verify.py [--workers 8] [--seeds 3]
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

# (name, m, 是否有逃逸 override)
CASES = [
    ("wfg4", 10, False),
    ("wfg5", 10, False),
    ("wfg6", 10, False),
    ("wfg2", 10, True),
    ("wfg3", 5, True),
    ("wfg9", 10, True),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    name, m, seed, fill = task
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=problem_type, problem_name=name,
                   niche_fill=fill)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    igdp = float(IGDPlus(pf)(F))
    return name, m, seed, fill, igd, igdp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    seeds = SEEDS_ALL[:args.seeds]

    tasks = [(n, m, s, f)
             for n, m, _ in CASES for f in (False, True) for s in seeds]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                name, m, seed, fill, igd, igdp = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results.setdefault((name, m, fill), []).append((igd, igdp))

    print("\n" + "=" * 80)
    print(f"{'case':<12} {'fill':>5} {'IGD':>18} {'IGD+':>18}  ΔIGD%  ΔIGD+%")
    print("-" * 80)
    for name, m, _has_esc in CASES:
        off = results.get((name, m, False), [])
        on = results.get((name, m, True), [])
        if not off or not on:
            continue
        off_igd = np.mean([x[0] for x in off])
        off_igdp = np.mean([x[1] for x in off])
        on_igd = np.mean([x[0] for x in on])
        on_igdp = np.mean([x[1] for x in on])
        digd = (on_igd - off_igd) / (abs(off_igd) + 1e-12) * 100
        digdp = (on_igdp - off_igdp) / (abs(off_igdp) + 1e-12) * 100
        tag = "✓" if digdp < -5 else ("✗" if digdp > 5 else "≈")
        print(f"{name}{m:<6} {'off':>5} {off_igd:>18.4f} {off_igdp:>18.4f}")
        print(f"{'':<12} {'on':>5} {on_igd:>18.4f} {on_igdp:>18.4f} "
              f"{digd:>+6.1f}% {digdp:>+6.1f}%  {tag}")
    print("=" * 80)


if __name__ == "__main__":
    main()
