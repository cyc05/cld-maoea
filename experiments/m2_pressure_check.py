"""
M2 核心机制验证：m=10 DTLZ1 上闭环加压 vs 无强化（phi=0.5 固定）。
检验闭环支配强化在高目标数下的收敛加速作用。
运行: python experiments/m2_pressure_check.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD

from src.algorithm import CLDMaOEA
from src.dominance import das_dennis_weights

M = 10
FE_MAX = 50000
SEEDS = [42, 7, 2024]

VARIANTS = {
    "closed-loop": dict(fixed_phi=None),   # 完整闭环
    "no-pressure": dict(fixed_phi=0.5),    # 纯 Pareto（机制 A 关闭）
}


def get_true_pf(m, p=7):
    W = das_dennis_weights(m, p)
    return 0.5 * W / W.sum(axis=1, keepdims=True)


def main():
    problem = get_problem("dtlz1", n_var=M + 4, n_obj=M)
    igd = IGD(get_true_pf(M))

    for name, cfg in VARIANTS.items():
        igds, times = [], []
        for seed in SEEDS:
            algo = CLDMaOEA(problem, m=M, seed=seed,
                            niche_protect=True, niche_mating=True, **cfg)
            t0 = time.time()
            X, F = algo.run(fe_max=FE_MAX)
            times.append(time.time() - t0)
            igds.append(igd(F))
        print(f"{name:12s} IGD={np.mean(igds):.4f}±{np.std(igds):.4f} "
              f"(min={np.min(igds):.4f})  耗时={np.mean(times):.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
