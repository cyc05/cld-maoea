"""
wfg5 m=10 角点档案（方案 C）Stage C-1 验证。
配置：off / C_replace(0.1 主推) / C_union / C_replace(0.05/0.2 预算消融)，
10 种子配对。成功判据（设计文档）：
- 轨迹不变性断言（原始种群 F 哈希逐种子相等）
- IGD+ 显著改善 p<0.05 且 IGD 不显著退化
- 好盆地数 2 -> >=8（方差坍缩）
- 好态种子（off IGD+<=1.5）不得超 off*1.05

运行: python experiments/wfg5_corner_archive.py [--workers 8]
"""
import hashlib
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000
NAME, M = "wfg5", 10
GOOD_IGDP = 1.8
FLIP_LIMIT = 1.05

# (标签, corner_archive, mode, budget)
CONFIGS = [
    ("off",      False, "replace", 0.1),
    ("C_rep_0.1", True, "replace", 0.1),   # 主推
    ("C_union",  True, "union",   0.1),
    ("C_rep_0.05", True, "replace", 0.05),
    ("C_rep_0.2",  True, "replace", 0.2),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    seed, tag, arc, mode, budget = task
    problem = make_problem(NAME, M)
    alg = CLDMaOEA(problem, m=M, seed=seed, variant="multi_modal",
                   problem_type="wfg", problem_name=NAME,
                   corner_archive=arc, archive_out_mode=mode,
                   archive_budget=budget)
    X, F = alg.run(fe_max=FE_MAX)
    raw = alg.pop_F_raw if arc else F
    h = hashlib.md5(np.ascontiguousarray(raw).tobytes()).hexdigest()[:10]
    n_merged = len(F) - 0 if not arc else len(F) - len(raw) + len(raw)  # union 增量
    pf = get_pf(NAME, M)
    igd = float(IGD(pf)(F))
    igdp = float(IGDPlus(pf)(F))
    extra = len(F) - len(raw) if arc else 0
    return tag, seed, igd, igdp, h, extra


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
    traj_ok = all(res[s][3] == off[s][3] for s in seeds)
    merged = [res[s][4] for s in seeds]
    print(f"\n[{tag}] vs off（{len(seeds)} 种子配对）")
    print(f"  IGD  {o_igd.mean():.4f} -> {n_igd.mean():.4f}  "
          f"({d_igd.mean():+.1f}%, p={p_igd:.4f})")
    print(f"  IGD+ {o_igdp.mean():.4f} -> {n_igdp.mean():.4f}  "
          f"({d_igdp.mean():+.1f}%, p={p_igdp:.4f})")
    print(f"  双指标同向改善: {same_dir}/{len(seeds)} | "
          f"好盆地(IGD+<{GOOD_IGDP}): {good_off} -> {good_on} | "
          f"好态翻转违规(>{FLIP_LIMIT}x): {len(flips)}{flips if flips else ''}")
    print(f"  轨迹不变性(哈希): {'PASS' if traj_ok else '*** FAIL ***'} | "
          f"档案合并数 min/mean/max = {min(merged)}/{np.mean(merged):.1f}/{max(merged)}")
    return seeds, o_igd, n_igd, o_igdp, n_igdp, d_igd, d_igdp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(s, tag, arc, mode, budget)
             for s in SEEDS for (tag, arc, mode, budget) in CONFIGS]
    print(f"wfg5 m=10 角点档案（方案 C）Stage C-1：{len(SEEDS)} 种子 × "
          f"{len(CONFIGS)} 配置，workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {tag: {} for tag, *_ in CONFIGS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                tag, seed, igd, igdp, h, extra = fut.result()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                continue
            results[tag][seed] = (igd, igdp, None, h, extra)
            print(f"[done] {tag:>10} seed={seed} IGD={igd:.4f} IGD+={igdp:.4f} "
                  f"merged={extra}", flush=True)

    print("\n" + "=" * 78)
    tables = {}
    for tag, *_ in CONFIGS:
        if tag == "off":
            continue
        tables[tag] = analyze(tag, results[tag], results["off"])

    tag = "C_rep_0.1"
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
