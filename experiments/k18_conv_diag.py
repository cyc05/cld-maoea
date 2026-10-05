"""
K.18 后续：conv_ema / div_ema 预热期分布诊断。
判断预热期分位数标定是否适用于 conv_eps（第 5 项），
对比手调 conv_eps（0.01/0.05）与预热期分布的关系。

运行: python experiments/k18_conv_diag.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.m3_benchmark import make_problem
from src.algorithm import CLDMaOEA


def profile(name, m, seed=42, fe_max=100000):
    problem = make_problem(name, m)
    pt = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=pt, problem_name=name)

    conv_series = []
    div_series = []

    def cb(X, F, self):
        conv_series.append(float(self.controller.conv_ema))
        div_series.append(float(self.controller.div_ema))

    alg.run(fe_max=fe_max, callback=cb)

    conv = np.asarray(conv_series)
    div = np.asarray(div_series)
    n = len(conv)
    warmup = int(n * 0.2)
    cw = conv[:warmup]
    dw = div[:warmup]
    print(f"\n=== {name} m={m} (总代 {n}, 预热 {warmup} 代) ===")
    print(f"conv_ema 预热期: min={cw.min():.5g} p5={np.quantile(cw,0.05):.5g} "
          f"p25={np.quantile(cw,0.25):.5g} p50={np.quantile(cw,0.5):.5g}")
    print(f"conv_ema 全期  : min={conv.min():.5g} p5={np.quantile(conv,0.05):.5g} "
          f"p25={np.quantile(conv,0.25):.5g} p50={np.quantile(conv,0.5):.5g}")
    print(f"div_ema  预热期: min={dw.min():.5g} p5={np.quantile(dw,0.05):.5g} "
          f"p25={np.quantile(dw,0.25):.5g} p50={np.quantile(dw,0.5):.5g}")


if __name__ == "__main__":
    for name, m in [("dtlz3", 5), ("dtlz3", 10), ("wfg9", 10),
                    ("dtlz1", 10), ("wfg2", 5)]:
        profile(name, m)
