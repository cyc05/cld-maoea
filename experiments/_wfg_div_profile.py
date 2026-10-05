"""WFG 全系列 div_ema 诊断，为逃逸标定提供数据。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.variants import CLDVariant
from src.algorithm import CLDMaOEA
from experiments.m3_benchmark import make_problem, get_pf
from pymoo.indicators.igd import IGD

SEEDS = [42, 7, 2024]
FE_MAX = 100000

# WFG1-9，每题 m=5/10
CASES = []
for wfg_id in range(1, 10):
    name = f"wfg{wfg_id}"
    for m in [5, 10]:
        CASES.append((name, m))

for name, m in CASES:
    print(f"\n{'='*72}")
    print(f"{name} m={m}")
    print(f"{'='*72}")

    igds = []
    for s in SEEDS:
        problem = make_problem(name, m)
        alg = CLDMaOEA(problem, m=m, seed=s, variant=CLDVariant.HIGH_DIM,
                       problem_type="wfg", problem_name=name,
                       conv_eps=0.01, div_eps=0.001)  # 极低阈值只记录不触发
        t0 = time.perf_counter()
        X, F = alg.run(fe_max=FE_MAX)
        el = time.perf_counter() - t0
        pf = get_pf(name, m)
        igd = float(IGD(pf)(F))
        igds.append(igd)

        hist = alg.controller.history
        d_emas = [h[6] for h in hist]
        stalls = [h[4] for h in hist]

        print(f"  seed={s:>4}: IGD={igd:.4f}  div_ema: "
              f"min={min(d_emas):.4f}  p10={np.percentile(d_emas, 10):.4f}  "
              f"p25={np.percentile(d_emas, 25):.4f}  "
              f"median={np.median(d_emas):.4f}  "
              f"mean={np.mean(d_emas):.4f}  max={max(d_emas):.4f}  "
              f"stall_max={max(stalls)}", flush=True)

        # 轨迹采样
        step = max(1, len(d_emas) // 8)
        print(f"       div轨迹(每{step}代): ", end="")
        for i in range(0, len(d_emas), step):
            print(f"t{i}={d_emas[i]:.3f}", end=" ")
        print()

    print(f"  → IGD: {np.mean(igds):.4f}±{np.std(igds):.4f}")

print("\n=== 建议 div_eps 汇总 ===")
for name, m in CASES:
    pt = "wfg"
    c = get_variant_config(CLDVariant.HIGH_DIM, m=m, problem_type=pt, problem_name=name)
    print(f"{name} m={m}: [待按 p25-median 标定]")
