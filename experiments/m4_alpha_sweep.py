"""
M4 champion-APD 的 alpha 扫描：alpha 控制方向对齐惩罚强度。
alpha=2 已验证；此处测 alpha=4, 8 在关键案例上的表现。
注: r2_mode='tche' 时 r2_alpha 仅作用于 champion 的 APD 选取。
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.m3_benchmark import make_problem, get_pf

SEEDS = [42, 7, 2024]
FE_MAX = 100000
CASES = [("dtlz3", 10), ("dtlz1", 10), ("wfg7", 10), ("wfg4", 5), ("dtlz3", 5)]
ALPHAS = [4.0, 8.0]


def run_one(name, m, seed, alpha):
    from src.algorithm import CLDMaOEA
    from pymoo.indicators.igd import IGD
    problem = make_problem(name, m)
    alg = CLDMaOEA(problem, m=m, seed=seed,
                   protect_mode="champion", r2_alpha=alpha)
    X, F = alg.run(fe_max=FE_MAX)
    return float(IGD(get_pf(name, m))(F))


def main():
    print("参照: alpha=2 → dtlz310=0.5994 dtlz110=0.3143 "
          "wfg710=2.9564 wfg45=2.0421 dtlz35=0.3159")
    for alpha in ALPHAS:
        print(f"--- alpha={alpha} ---", flush=True)
        for name, m in CASES:
            igds = [run_one(name, m, s, alpha) for s in SEEDS]
            print(f"{name}{m:<3} {np.mean(igds):.4f}±{np.std(igds):.4f}  "
                  + " ".join(f"{v:.4f}" for v in igds), flush=True)


if __name__ == "__main__":
    main()
