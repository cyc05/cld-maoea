"""
M4 wfg4(m=5) 退化消融：分离控制器改动与冠军保护改动的贡献。
配置:
  M4-full   : 当前默认（stall=10, phi_relax=0.45, 冠军保护）
  relax50   : phi_relax=0.5（M3 式完全弛豫）
  nostall   : stall_patience=0（M3 式即时门控）
  m3ctrl    : phi_relax=0.5 + stall=0（≈M3 控制器）
  fixed05   : fixed_phi=0.5（纯 Pareto 参照）
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.m3_benchmark import make_problem, get_pf

SEEDS = [42, 7, 2024]
FE_MAX = 100000

CONFIGS = [
    ("M4-full", dict()),
    ("relax50", dict(phi_relax=0.5)),
    ("nostall", dict(stall_patience=0)),
    ("m3ctrl", dict(phi_relax=0.5, stall_patience=0)),
    ("fixed05", dict(fixed_phi=0.5)),
]


def run_one(cfg, seed):
    from src.algorithm import CLDMaOEA
    from pymoo.indicators.igd import IGD
    problem = make_problem("wfg4", 5)
    alg = CLDMaOEA(problem, m=5, seed=seed, **cfg)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf("wfg4", 5)
    return float(IGD(pf)(F))


def main():
    print(f"M3 基线: CLD=1.5415  NSGA3=2.0574")
    for name, cfg in CONFIGS:
        igds = [run_one(cfg, s) for s in SEEDS]
        print(f"{name:<10} {np.mean(igds):.4f}±{np.std(igds):.4f}  "
              + " ".join(f"{v:.4f}" for v in igds), flush=True)


if __name__ == "__main__":
    main()
