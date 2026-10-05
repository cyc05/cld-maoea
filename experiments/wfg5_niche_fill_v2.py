"""
wfg5 m=10 niche_fill v2（方案 A：PG-EF）Stage 1 验证。
对比 off / A(0.3,5) / 参数网格，10 种子配对：
- 显著性（配对 t 检验）
- 双指标同向改善率
- 防翻转硬检查（好态种子 off IGD+<=1.5，开启后不得超 off*1.2）
- 好盆地占比（IGD+ < 1.8 视为好盆地，基线双峰 ~1.33 vs ~2.4）

运行: python experiments/wfg5_niche_fill_v2.py [--workers 8]
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000
NAME, M = "wfg5", 10
GOOD_IGDP = 1.8      # 好盆地分界（基线双峰 1.33 vs 2.4）
FLIP_LIMIT = 1.2     # 好态种子防翻转上限（off*1.2）

# (标签, niche_fill, until_frac, persist_gens)
CONFIGS = [
    ("off",       False, 1.0, 0),
    ("A_0.3_5",   True,  0.3, 5),    # 主推配置
    ("G_0.2_5",   True,  0.2, 5),
    ("G_0.5_5",   True,  0.5, 5),
    ("G_0.3_3",   True,  0.3, 3),
    ("G_0.3_10",  True,  0.3, 10),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    seed, tag, fill, until, persist = task
    problem = make_problem(NAME, M)
    alg = CLDMaOEA(problem, m=M, seed=seed, variant="multi_modal",
                   problem_type="wfg", problem_name=NAME,
                   niche_fill=fill,
                   niche_fill_until_frac=until,
                   niche_fill_persist_gens=persist)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(NAME, M)
    return tag, seed, float(IGD(pf)(F)), float(IGDPlus(pf)(F))


def analyze(tag, res, off):
    from scipy import stats
    seeds = [s for s in SEEDS if s in res and s in off]
    o_igd = np.array([off[s][0] for s in seeds])
    n_igd = np.array([res[s][0] for s in seeds])
    o_igdp = np.array([off[s][1] for s in seeds])
    n_igdp = np.array([res[s][1] for s in seeds])
    d_igd = (n_igd - o_igd) / np.abs(o_igd) * 100
    d_igdp = (n_igdp - o_igdp) / np.abs(o_igdp) * 100
    _, p_igd = stats.ttest_rel(n_igd, o_igd)
    _, p_igdp = stats.ttest_rel(n_igdp, o_igdp)
    same_dir = int(np.sum((d_igd < 0) & (d_igdp < 0)))
    flips = [(int(s), float(o), float(n)) for s, o, n in zip(seeds, o_igdp, n_igdp)
             if o <= 1.5 and n > o * FLIP_LIMIT]
    good_off = int(np.sum(o_igdp < GOOD_IGDP))
    good_on = int(np.sum(n_igdp < GOOD_IGDP))
    print(f"\n[{tag}] vs off（{len(seeds)} 种子配对）")
    print(f"  IGD  {o_igd.mean():.4f} -> {n_igd.mean():.4f}  "
          f"({d_igd.mean():+.1f}%, p={p_igd:.4f})")
    print(f"  IGD+ {o_igdp.mean():.4f} -> {n_igdp.mean():.4f}  "
          f"({d_igdp.mean():+.1f}%, p={p_igdp:.4f})")
    print(f"  双指标同向改善: {same_dir}/{len(seeds)} | "
          f"好盆地(IGD+<{GOOD_IGDP}): {good_off} -> {good_on} | "
          f"好态翻转违规: {len(flips)}{flips if flips else ''}")
    return seeds, o_igd, n_igd, o_igdp, n_igdp, d_igd, d_igdp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(s, tag, fill, until, persist)
             for s in SEEDS for (tag, fill, until, persist) in CONFIGS]
    print(f"wfg5 m=10 niche_fill v2 (PG-EF) Stage 1：{len(SEEDS)} 种子 × "
          f"{len(CONFIGS)} 配置，workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {tag: {} for tag, *_ in CONFIGS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                tag, seed, igd, igdp = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results[tag][seed] = (igd, igdp)
            print(f"[done] {tag:>8} seed={seed} IGD={igd:.4f} IGD+={igdp:.4f}",
                  flush=True)

    print("\n" + "=" * 78)
    tables = {}
    for tag, *_ in CONFIGS:
        if tag == "off":
            continue
        tables[tag] = analyze(tag, results[tag], results["off"])

    # 主推配置逐种子明细
    tag = "A_0.3_5"
    if tag in tables:
        seeds, o_igd, n_igd, o_igdp, n_igdp, d_igd, d_igdp = tables[tag]
        print(f"\n主推配置 {tag} 逐种子明细：")
        print(f"{'seed':>8} {'IGD off':>9} {'IGD on':>9} {'ΔIGD%':>7} "
              f"{'IGD+ off':>9} {'IGD+ on':>9} {'ΔIGD+%':>7}")
        for i, s in enumerate(seeds):
            print(f"{s:>8} {o_igd[i]:>9.4f} {n_igd[i]:>9.4f} {d_igd[i]:>+6.1f}% "
                  f"{o_igdp[i]:>9.4f} {n_igdp[i]:>9.4f} {d_igdp[i]:>+6.1f}%")


if __name__ == "__main__":
    main()
