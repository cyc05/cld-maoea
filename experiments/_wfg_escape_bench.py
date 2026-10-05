"""WFG 全系列逃逸验证：H 基线 vs M 逃逸。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.variants import CLDVariant
from src.algorithm import CLDMaOEA
from experiments.m3_benchmark import make_problem, get_pf
from pymoo.indicators.igd import IGD

SEEDS = [42, 7, 2024]
FE_MAX = 100000

CASES = []
for wfg_id in range(1, 10):
    name = f"wfg{wfg_id}"
    for m in [5, 10]:
        CASES.append((name, m))

results = {}
for name, m in CASES:
    print(f"\n{'='*72}")
    print(f"{name} m={m}")
    print(f"{'='*72}")
    pf = get_pf(name, m)
    ptype = "wfg"

    for variant, label in [(CLDVariant.HIGH_DIM, "H"), (CLDVariant.MULTI_MODAL, "M")]:
        igds, els = [], []
        esc_gens, stall_max = 0, 0
        for s in SEEDS:
            problem = make_problem(name, m)
            alg = CLDMaOEA(problem, m=m, seed=s, variant=variant,
                           problem_type=ptype, problem_name=name)
            t0 = time.perf_counter()
            X, F = alg.run(fe_max=FE_MAX)
            els.append(time.perf_counter() - t0)
            igds.append(float(IGD(pf)(F)))
            hist = alg.controller.history
            esc_gens += sum(1 for h in hist if h[7] > 0.5)
            stall_max = max(stall_max, max(h[4] for h in hist))
        mu, sd = np.mean(igds), np.std(igds)
        results[(name, m, label)] = (mu, sd, esc_gens, stall_max)

        if label == "M":
            h_mu, _, _, _ = results[(name, m, "H")]
            pct = (mu - h_mu) / (abs(h_mu) + 1e-12) * 100
            tag = f" ({'+' if pct>0 else ''}{pct:.1f}%)"
        else:
            tag = ""
        print(f"  {label}: IGD={mu:.4f}±{sd:.4f}{tag}  "
              f"time={np.mean(els):.1f}s  esc={esc_gens}  stall={stall_max}", flush=True)

print("\n" + "="*72)
print("WFG 逃逸效果汇总（M 较 H 变化）")
print("="*72)
wins, losses, ties = 0, 0, 0
for name, m in CASES:
    h_mu, h_sd, _, _ = results[(name, m, "H")]
    m_mu, m_sd, esc, _ = results[(name, m, "M")]
    pct = (m_mu - h_mu) / (abs(h_mu) + 1e-12) * 100
    if pct < -5:
        verdict = "✓ 改善"
        wins += 1
    elif pct > 5:
        verdict = "✗ 退化"
        losses += 1
    else:
        verdict = "≈ 持平"
        ties += 1
    print(f"{name} m={m:>2}: H {h_mu:.4f} → M {m_mu:.4f} ({pct:+.1f}%)  {verdict}  esc={esc}")

print(f"\n胜/负/平: {wins}/{losses}/{ties}")
