"""
P2 历史遗留专项验证：wfg3 回归 + dtlz7 overrides 生效确认。
运行: python experiments/p2_verify.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from experiments.m3_benchmark import make_problem, get_pf
from pymoo.indicators.igd import IGD
from src.algorithm import CLDMaOEA
from src.variants import CLDVariant

CASES = [
    # (name, m, variant_a, variant_b, label_a, label_b)
    ("wfg3", 5,  CLDVariant.HIGH_DIM, CLDVariant.LOW_DIM,  "H(champion)", "L(adaptive)"),
    ("wfg3", 10, CLDVariant.HIGH_DIM, CLDVariant.LOW_DIM,  "H(champion)", "L(adaptive)"),
    ("dtlz7", 5, CLDVariant.HIGH_DIM, CLDVariant.MULTI_MODAL, "H(lag=0.3)", "M(lag=0.4)"),
    ("dtlz7", 10, CLDVariant.HIGH_DIM, CLDVariant.MULTI_MODAL, "H(lag=0.3)", "M(lag=0.3)"),
]
SEEDS = [42, 7, 2024]
FE_MAX = 100000


def run_one(name, m, seed, variant):
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant=variant,
                   problem_type=problem_type, problem_name=name)
    t0 = time.perf_counter()
    X, F = alg.run(fe_max=FE_MAX)
    el = time.perf_counter() - t0
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    return igd, el


def main():
    print("=" * 75)
    print("P2 历史遗留验证: wfg3 回归 + dtlz7 overrides")
    print("=" * 75)
    print(f"种子: {len(SEEDS)} 个, FE={FE_MAX}")
    print()

    for name, m, va, vb, la, lb in CASES:
        a_igds, b_igds = [], []
        a_times, b_times = [], []
        for s in SEEDS:
            print(f"  {name} m={m} seed={s} ...", flush=True)
            igd_a, t_a = run_one(name, m, s, va)
            igd_b, t_b = run_one(name, m, s, vb)
            a_igds.append(igd_a)
            b_igds.append(igd_b)
            a_times.append(t_a)
            b_times.append(t_b)

        mu_a, mu_b = np.mean(a_igds), np.mean(b_igds)
        sd_a, sd_b = np.std(a_igds), np.std(b_igds)
        change = (mu_b - mu_a) / max(mu_a, 1e-12) * 100
        if mu_b < mu_a * 0.95:
            v = f"{lb}优 ({change:+.1f}%)"
        elif mu_b > mu_a * 1.05:
            v = f"{la}优 ({change:+.1f}%)"
        else:
            v = f"持平 ({change:+.1f}%)"
        print(f"  {name} m={m:<3} {la:<13} {mu_a:.4f}±{sd_a:.3f}  ({np.mean(a_times):.1f}s)")
        print(f"  {'':>10} {lb:<13} {mu_b:.4f}±{sd_b:.3f}  ({np.mean(b_times):.1f}s)")
        print(f"  {'':>10} 判定: {v}")
        print()

    print("=" * 75)
    print("预期:")
    print("  wfg3:    教训8 → H(champion+无d⊥) 应优于或持平 L(adaptive)")
    print("  dtlz7 m=5: 教训7 → M(lag=0.4) 应优于 H(lag=0.3)，差异 1.40→1.14")
    print("  dtlz7 m=10: overrides → 两者 lag 均为 0.3，应持平")
    print("=" * 75)


if __name__ == "__main__":
    main()
