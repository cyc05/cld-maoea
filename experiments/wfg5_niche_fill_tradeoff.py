"""
wfg5 m=10 的 niche_fill trade-off 专项实验。
用更多种子（10）对比 niche_fill on/off 的 IGD 与 IGD+，
逐种子配对输出，量化 IGD 退化换 IGD+ 改善的 trade-off。

运行: python experiments/wfg5_niche_fill_tradeoff.py [--workers 8]
"""
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000
NAME, M = "wfg5", 10


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    seed, fill = task
    problem = make_problem(NAME, M)
    alg = CLDMaOEA(problem, m=M, seed=seed, variant="multi_modal",
                   problem_type="wfg", problem_name=NAME, niche_fill=fill)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(NAME, M)
    igd = float(IGD(pf)(F))
    igdp = float(IGDPlus(pf)(F))
    return seed, fill, igd, igdp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(s, f) for s in SEEDS for f in (False, True)]
    print(f"wfg5 m=10 niche_fill trade-off：{len(SEEDS)} 种子 × 2 配置，"
          f"workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    off, on = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                seed, fill, igd, igdp = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            (on if fill else off)[seed] = (igd, igdp)
            print(f"[done] seed={seed} fill={fill} IGD={igd:.4f} "
                  f"IGD+={igdp:.4f}", flush=True)

    print("\n" + "=" * 76)
    print(f"{'seed':>8} {'IGD off':>9} {'IGD on':>9} {'ΔIGD%':>7} "
          f"{'IGD+ off':>9} {'IGD+ on':>9} {'ΔIGD+%':>7}")
    print("-" * 76)

    digd_list, digdp_list = [], []
    for s in SEEDS:
        if s not in off or s not in on:
            continue
        o_igd, o_igdp = off[s]
        n_igd, n_igdp = on[s]
        digd = (n_igd - o_igd) / abs(o_igd) * 100
        digdp = (n_igdp - o_igdp) / abs(o_igdp) * 100
        digd_list.append(digd)
        digdp_list.append(digdp)
        print(f"{s:>8} {o_igd:>9.4f} {n_igd:>9.4f} {digd:>+6.1f}% "
              f"{o_igdp:>9.4f} {n_igdp:>9.4f} {digdp:>+6.1f}%")

    print("-" * 76)
    o_igd = np.array([off[s][0] for s in SEEDS if s in off])
    n_igd = np.array([on[s][0] for s in SEEDS if s in on])
    o_igdp = np.array([off[s][1] for s in SEEDS if s in off])
    n_igdp = np.array([on[s][1] for s in SEEDS if s in on])
    print(f"{'均值':>8} {o_igd.mean():>9.4f} {n_igd.mean():>9.4f} "
          f"{(n_igd.mean()-o_igd.mean())/abs(o_igd.mean())*100:>+6.1f}% "
          f"{o_igdp.mean():>9.4f} {n_igdp.mean():>9.4f} "
          f"{(n_igdp.mean()-o_igdp.mean())/abs(o_igdp.mean())*100:>+6.1f}%")
    print(f"{'标准差':>8} {o_igd.std():>9.4f} {n_igd.std():>9.4f} "
          f"{'':>7} {o_igdp.std():>9.4f} {n_igdp.std():>9.4f}")

    # 逐种子配对 t 检验（配对样本）
    from scipy import stats
    t_igd, p_igd = stats.ttest_rel(n_igd, o_igd)
    t_igdp, p_igdp = stats.ttest_rel(n_igdp, o_igdp)
    print(f"\n配对 t 检验：IGD  t={t_igd:.3f} p={p_igd:.4f}  "
          f"({'显著退化' if p_igd < 0.05 and t_igd > 0 else '不显著'})")
    print(f"配对 t 检验：IGD+ t={t_igdp:.3f} p={p_igdp:.4f}  "
          f"({'显著改善' if p_igdp < 0.05 and t_igdp < 0 else '不显著'})")


if __name__ == "__main__":
    main()
