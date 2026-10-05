"""
M4 快速验证：只跑 CLD 的 M3 失败题 + 关键回归题。
失败题: dtlz3(m=5/10), dtlz1(m=10), wfg7(m=10)  —— 多峰/高维停滞
回归题: wfg4(m=5)  —— M3 中 CLD 最优，须防退化
对比 M3 基线（均值）直接打印，不写 CSV。
运行: python experiments/m4_verify.py [--protect last|champion]
"""
import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.m3_benchmark import make_problem, get_pf

# (problem, m, M3基线IGD均值, M3最优基线IGD)
CASES = [
    ("dtlz3", 5, 1.4242, 0.2866),
    ("dtlz3", 10, 8.1401, 0.4643),
    ("dtlz1", 10, 0.3207, 0.1593),
    ("dtlz7", 5, 1.2608, 0.6649),
    ("dtlz7", 10, 2.3225, 2.2576),
    ("wfg7", 10, 5.8878, 2.4194),
    ("wfg4", 5, 1.5415, 1.5415),   # 回归题：M3 中 CLD 已最优
]
SEEDS = [42, 7, 2024]
FE_MAX = 100000


def run_one(name, m, seed, protect, lag_thresh=0.5, d_perp=True, problem_type=None,
            lag_quantile=None):
    from src.algorithm import CLDMaOEA
    from pymoo.indicators.igd import IGD
    problem = make_problem(name, m)
    alg = CLDMaOEA(problem, m=m, seed=seed, protect_mode=protect,
                   lag_thresh=lag_thresh, d_perp_tiebreak=d_perp,
                   problem_type=problem_type, problem_name=name,
                   lag_quantile=lag_quantile)
    t0 = time.perf_counter()
    X, F = alg.run(fe_max=FE_MAX)
    el = time.perf_counter() - t0
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    hist = alg.history
    return igd, el, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protect", default="adaptive",
                    choices=["last", "champion", "adaptive"])
    ap.add_argument("--lag-thresh", type=float, default=0.5)
    ap.add_argument("--no-dperp", action="store_true",
                    help="禁用 d⊥ 平局打破")
    ap.add_argument("--lag-quantile", type=float, default=None,
                    help="lag 分位数阈值（如 0.75），dtlz7 专用")
    args = ap.parse_args()
    d_perp = not args.no_dperp
    lag_q = args.lag_quantile
    print(f"protect_mode = {args.protect}, lag_thresh = {args.lag_thresh}, "
          f"d_perp = {d_perp}, lag_quantile = {lag_q}")
    print(f"{'case':<14} {'seeds':>5} {'IGD(mean±std)':>22} "
          f"{'M3基线':>10} {'M3最优':>10} {'判定':>8} {'time':>6}")
    all_rows = []
    for name, m, base, best in CASES:
        igds, els = [], []
        last_hist = None
        ptype = "dtlz" if name.startswith("dtlz") else "wfg"
        # P10 结论：dtlz7 固定 lag_thresh=0.4 优于分位数动态阈值，lag_quantile 禁用
        case_lag_q = lag_q
        if name == "dtlz7" and case_lag_q is None:
            case_lag_q = None  # 固定 0.4 由 problem_name 逻辑处理
        for s in SEEDS:
            igd, el, hist = run_one(name, m, s, args.protect, args.lag_thresh,
                                    d_perp, ptype, case_lag_q)
            igds.append(igd)
            els.append(el)
            last_hist = hist
        mu, sd = float(np.mean(igds)), float(np.std(igds))
        # 判定：均值较 M3 基线改善 >20% 记 ↑，退化 >20% 记 ↓，否则 ≈
        if mu < base * 0.8:
            verdict = "↑改善"
        elif mu > base * 1.2:
            verdict = "↓退化"
        else:
            verdict = "≈持平"
        tag = f"{name}{m}"
        print(f"{tag:<14} {len(SEEDS):>5} {mu:>10.4f}±{sd:<9.4f} "
              f"{base:>10.4f} {best:>10.4f} {verdict:>8} {np.mean(els):>5.1f}s",
              flush=True)
        all_rows.append((tag, mu, sd, igds))
        # 控制器轨迹摘要（最后一个种子）
        phi, gm = last_hist["phi"], last_hist["gamma"]
        rho = last_hist["rho"]
        print(f"  └ phi: {phi[0]:.3f}→{phi[-1]:.3f}  "
              f"gamma<1 代数: {sum(1 for g in gm if g < 1.0)}/{len(gm)}  "
              f"rho末值: {rho[-1]:.3f}  niches末值: {last_hist['niches'][-1]}",
              flush=True)

    print("\n=== 汇总 ===")
    n_improve = sum(1 for t, mu, _, _ in all_rows
                    if mu < dict(((f"{n}{m_}", b) for n, m_, b, _ in CASES))[t] * 0.8)
    print(f"改善 {n_improve}/{len(all_rows)} 例")
    for tag, mu, sd, igds in all_rows:
        print(f"  {tag:<10} {mu:.4f}±{sd:.4f}  各种子: "
              + " ".join(f"{v:.4f}" for v in igds))


if __name__ == "__main__":
    main()
