"""
P0 综合验证：
  a) 惰性贪心正确性 + 速度对比 (S 版 with/without lazy)
  b) 控制器死代码诊断 (dtlz7 m=5/10 上 stall/gamma/phi_relax 触发情况)
运行: python experiments/p0_verify.py
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

SEEDS = [42, 7, 2024]
FE_MAX = 100000


# ─── P0-a: 惰性贪心正确性 + 速度 ───
def test_lazy_greedy():
    print("=" * 70)
    print("P0-a: 惰性贪心正确性 + 速度对比")
    print("=" * 70)
    cases = [("dtlz3", 10), ("dtlz7", 5), ("wfg9", 10)]
    print(f"{'case':<12} {'non-lazy IGD':>14} {'lazy IGD':>14} {'ΔIGD':>8} "
          f"{'non-lazy t':>11} {'lazy t':>11} {'加速':>6}")
    print("-" * 70)

    for name, m in cases:
        pt = "dtlz" if name.startswith("dtlz") else "wfg"
        nl_igds, lz_igds, nl_ts, lz_ts = [], [], [], []
        for s in SEEDS:
            problem = make_problem(name, m)
            # non-lazy: S 版但关闭 lazy
            alg_nl = CLDMaOEA(problem, m=m, seed=s, variant=CLDVariant.SPEED,
                              problem_type=pt, problem_name=name,
                              use_lazy_greedy=False)
            t0 = time.perf_counter()
            X_nl, F_nl = alg_nl.run(fe_max=FE_MAX)
            nl_t = time.perf_counter() - t0
            pf = get_pf(name, m)
            nl_igd = float(IGD(pf)(F_nl))

            # lazy: S 版开启 lazy
            problem2 = make_problem(name, m)
            alg_lz = CLDMaOEA(problem2, m=m, seed=s, variant=CLDVariant.SPEED,
                              problem_type=pt, problem_name=name,
                              use_lazy_greedy=True)
            t0 = time.perf_counter()
            X_lz, F_lz = alg_lz.run(fe_max=FE_MAX)
            lz_t = time.perf_counter() - t0
            lz_igd = float(IGD(pf)(F_lz))

            nl_igds.append(nl_igd)
            lz_igds.append(lz_igd)
            nl_ts.append(nl_t)
            lz_ts.append(lz_t)

        nl_mu, lz_mu = np.mean(nl_igds), np.mean(lz_igds)
        nl_t_mu, lz_t_mu = np.mean(nl_ts), np.mean(lz_ts)
        delta = lz_mu - nl_mu
        speedup = nl_t_mu / lz_t_mu if lz_t_mu > 0 else 0
        match = "✓" if abs(delta) < 0.01 else "✗"
        print(f"{name}{m:<6} {nl_mu:>10.4f}     {lz_mu:>10.4f}     "
              f"{delta:>+7.4f}{match} {nl_t_mu:>8.1f}s   {lz_t_mu:>8.1f}s   "
              f"{speedup:>5.2f}x")
    print()


# ─── P0-b: 控制器死代码诊断 ───
def diagnose_controller():
    print("=" * 70)
    print("P0-b: 控制器死代码诊断 (conv_eps=1e-2, 弛豫路径激活验证)")
    print("=" * 70)
    cases = [
        ("dtlz7", 5,  CLDVariant.HIGH_DIM,    "H(stall=10,relax=0.45)"),
        ("dtlz7", 5,  CLDVariant.MULTI_MODAL, "M(stall=15,relax=0.40)"),
        ("dtlz7", 10, CLDVariant.HIGH_DIM,    "H(stall=10,relax=0.45)"),
        ("dtlz7", 10, CLDVariant.MULTI_MODAL, "M(stall=15,relax=0.40)"),
        ("dtlz3", 5,  CLDVariant.HIGH_DIM,    "H(stall=10,relax=0.45)"),
        ("dtlz3", 5,  CLDVariant.MULTI_MODAL, "M(stall=15,relax=0.40)"),
        ("dtlz3", 10, CLDVariant.HIGH_DIM,    "H(stall=10,relax=0.45)"),
        ("dtlz3", 10, CLDVariant.MULTI_MODAL, "M(stall=15,relax=0.40)"),
    ]
    print(f"{'case':<12} {'variant':<24} {'stall_max':>10} {'gamma<1代数':>12} "
          f"{'phi_min':>8} {'phi_relax':>10} {'触发relax':>10} {'stall>0代数':>12}")
    print("-" * 70)

    for name, m, variant, vlabel in cases:
        pt = "dtlz" if name.startswith("dtlz") else "wfg"
        problem = make_problem(name, m)
        alg = CLDMaOEA(problem, m=m, seed=42, variant=variant,
                       problem_type=pt, problem_name=name)
        X, F = alg.run(fe_max=FE_MAX)

        hist = alg.controller.history
        # history: (t, phi, rho, gamma, stall_counter, conv_ema)
        phis = [h[1] for h in hist]
        gammas = [h[3] for h in hist]
        stalls = [h[4] for h in hist]
        emas = [h[5] for h in hist]

        stall_max = max(stalls) if stalls else 0
        gamma_lt1 = sum(1 for g in gammas if g < 1.0)
        stall_gt0 = sum(1 for s in stalls if s > 0)
        phi_min_val = min(phis) if phis else 0
        phi_relax = alg.controller.phi_relax
        triggered = any(p <= phi_relax + 0.001 for p in phis) if phis else False

        print(f"{name}{m:<6} {vlabel:<24} {stall_max:>10} {gamma_lt1:>12} "
              f"{phi_min_val:>8.3f} {phi_relax:>10.3f} "
              f"{'是' if triggered else '否':>10} {stall_gt0:>12}")

        # 详细轨迹（每 20 代采样）
        print(f"  └ phi轨迹(每20代): ", end="")
        step = max(1, len(phis) // 8)
        for i in range(0, len(phis), step):
            print(f"t{i}={phis[i]:.3f}(s={stalls[i]},γ={gammas[i]:.2f})", end=" ")
        print()
    print()

    print("诊断结论:")
    print("  - stall_max > 0 表示检测到停滞（conv_ema < conv_eps）")
    print("  - gamma<1 代数 > 0 表示弛豫路径已激活")
    print("  - stall>0 代数 > 0 表示 stall_counter 曾被增加")


if __name__ == "__main__":
    test_lazy_greedy()
    diagnose_controller()
