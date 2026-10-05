"""
M2 消融实验：R2 截断策略对分布性的影响。
4 变体 × 5 种子，DTLZ1 m=5，FE=100k。

变体:
  V-tche        : Tchebycheff + Λ=W (M1 基线)
  V-apd         : APD + Λ=W
  V-tche-dense  : Tchebycheff + 增密 Λ (p=5, |Λ|=126)
  V-apd-dense   : APD + 增密 Λ

运行: python experiments/m2_ablation.py
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

M = 5
FE_MAX = 100000
SEEDS = [42, 7, 2024, 314, 99]

VARIANTS = {
    "V-tche":       dict(r2_mode="tche", niche_protect=False, niche_mating=False),
    "V-tche-np":    dict(r2_mode="tche", niche_protect=True,  niche_mating=False),
    "V-tche-np-nm": dict(r2_mode="tche", niche_protect=True,  niche_mating=True),
}


def get_true_pf(m, p=10):
    W = das_dennis_weights(m, p)
    return 0.5 * W / W.sum(axis=1, keepdims=True)


def main():
    problem = get_problem("dtlz1", n_var=M + 4, n_obj=M)
    igd = IGD(get_true_pf(M))

    results = {}
    for name, cfg in VARIANTS.items():
        igds, niches, times = [], [], []
        for seed in SEEDS:
            algo = CLDMaOEA(problem, m=M, seed=seed, **cfg)
            t0 = time.time()
            X, F = algo.run(fe_max=FE_MAX)
            times.append(time.time() - t0)
            igds.append(igd(F))
            niches.append(algo.history["niches"][-1])
        results[name] = (np.array(igds), np.array(niches), np.array(times))
        print(f"{name:14s} IGD={np.mean(igds):.4f}±{np.std(igds):.4f} "
              f"覆盖={np.mean(niches):5.1f}/35  耗时={np.mean(times):.1f}s",
              flush=True)

    os.makedirs("experiments/results", exist_ok=True)
    np.savez("experiments/results/m2_ablation.npz",
             **{f"{k}_igd": v[0] for k, v in results.items()},
             **{f"{k}_niche": v[1] for k, v in results.items()},
             **{f"{k}_time": v[2] for k, v in results.items()})
    print("已保存 experiments/results/m2_ablation.npz")


if __name__ == "__main__":
    main()
