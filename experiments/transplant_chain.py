"""
第三轮审稿意见 3 的回应实验：把准则链的四个已存档结构决策（corner_archive
开、policy=direction、out_mode=replace、budget=0.1）移植到轻量基线 NSGA-III/RVEA
上，验证准则链决策的价值是否独立于 CLD 算法本身。

口径：
  - 移植只改变"输出期合并"与"代际旁路观察"（passive observer），不改变基线动力学；
  - 每代回调内用与 CLD 相同的归一化口径（累计 z* + φ=0.5 首层 nadir 估计）
    喂给 CornerArchive.observe；
  - 输出期 finalize(replace, budget=0.1) 恒输出 N（预算公平）；
  - 对照 = m3_results.csv 中同种子同 FE 的 NSGA3/RVEA 原始行（上轮等预算实验已复核）；
  - 实例 = 准则链启用的 4 个：dtlz5 m=5/10、dtlz7 m=10、wfg2 m=10。

产物：results/transplant_chain.csv + results/transplant_summary.csv
运行：python experiments/transplant_chain.py [--workers 8]
"""
import os
import sys
import csv
import time
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RES_CSV = os.path.join(ROOT, "experiments", "results", "transplant_chain.csv")
SUM_CSV = os.path.join(ROOT, "experiments", "results", "transplant_summary.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
INSTANCES = [("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10), ("wfg2", 10)]
BASE_ALGOS = ["NSGA3", "RVEA"]
FE = 100000


def run_task(task):
    sys.path.insert(0, ROOT)
    name, m, algo, seed = task
    from experiments.m3_benchmark import make_problem, get_pf, n_partitions
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    from src.archives import CornerArchive
    from src.dominance import strengthened_nd_sort, estimate_nadir_extreme

    problem = make_problem(name, m)
    ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions(m))
    N = int(np.ceil(len(ref_dirs) / 4) * 4)
    if algo == "NSGA3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        method = NSGA3(ref_dirs=ref_dirs, pop_size=N)
    else:
        from pymoo.algorithms.moo.rvea import RVEA
        method = RVEA(ref_dirs=ref_dirs, pop_size=N)

    arch = CornerArchive(m, policy="direction", W=ref_dirs)
    state = {"z": None}

    def cb(a):
        F = np.asarray(a.pop.get("F"), dtype=float)
        X = np.asarray(a.pop.get("X"), dtype=float)
        if state["z"] is None:
            state["z"] = F.min(axis=0)
        else:
            state["z"] = np.minimum(state["z"], F.min(axis=0))
        try:
            F_shift = F - state["z"]
            fronts, _ = strengthened_nd_sort(F_shift, np.full(len(F), 0.5))
            span = estimate_nadir_extreme(F_shift[fronts[0]], state["z"])
        except Exception:
            # 极端退化代的雅可比奇异等退化情形：退回逐维最大值（鲁棒兜底）
            if fronts is None or len(fronts) == 0 or len(fronts[0]) == 0:
                raise
            span = np.max(F_shift[fronts[0]], axis=0)
        span = np.maximum(span, 1e-12)
        arch.observe(X, F, RF_norm=F_shift / span, front1=fronts[0])

    t0 = time.perf_counter()
    res = minimize(problem, method, termination=("n_evals", FE),
                   seed=seed, verbose=False, callback=cb)
    Xf, Ff = res.X, res.F
    z = np.min(Ff, axis=0)
    F_shift = Ff - z
    fronts, _ = strengthened_nd_sort(F_shift, np.full(len(Ff), 0.5))
    span = np.maximum(estimate_nadir_extreme(F_shift[fronts[0]], z), 1e-12)
    from src.r2select import compute_theta_w
    Xm, Fm = arch.finalize(Xf, Ff, F_norm_pop=F_shift / span, W=ref_dirs,
                           theta_w=compute_theta_w(ref_dirs),
                           mode="replace", budget=0.1)
    elapsed = time.perf_counter() - t0

    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = get_pf(name, m)
    return dict(problem=name, m=m, algo=algo, seed=seed,
                igd=float(IGD(pf)(Fm)), igd_plus=float(IGDPlus(pf)(Fm)),
                time=elapsed)


def a12(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :])
    eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(n, m, a, s) for (n, m) in INSTANCES for a in BASE_ALGOS for s in SEEDS]
    done = set()
    if os.path.exists(RES_CSV):
        with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["algo"], int(r["seed"])))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]
    print(f"移植实验 {len(tasks)} 任务（已完成 {len(done)}），workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fields = ["problem", "m", "algo", "seed", "igd", "igd_plus", "time"]
    os.makedirs(os.path.dirname(RES_CSV), exist_ok=True)
    header = not os.path.exists(RES_CSV)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, \
                open(RES_CSV, "a", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fields)
            if header:
                w.writeheader()
                fout.flush()
            futs = {pool.submit(run_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result())
                    fout.flush()
                except Exception as e:
                    print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                if i % 20 == 0 or i == len(futs):
                    print(f"[progress] {i}/{len(futs)}", flush=True)

    # ---- 对照：同种子同 FE 的原始基线（m3_results.csv）----
    from scipy.stats import wilcoxon
    tr = defaultdict(dict)
    with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tr[(r["problem"], int(r["m"]), r["algo"])][int(r["seed"])] = (
                float(r["igd"]), float(r["igd_plus"]))
    pl = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] in BASE_ALGOS:
                pl[(r["problem"], int(r["m"]), r["algo"])][int(r["seed"])] = (
                    float(r["igd"]), float(r["igd_plus"]))

    print("\n" + "=" * 70)
    print("移植效果：基线+准则链存档 vs 原始基线（负 = 移植后改善）")
    print("=" * 70)
    rows = []
    for (n, m, a), seeds in tr.items():
        plain = pl.get((n, m, a), {})
        common = sorted(set(seeds) & set(plain))
        tv = np.array([seeds[s][0] for s in common])
        pv = np.array([plain[s][0] for s in common])
        tvs = np.array([seeds[s][1] for s in common])
        pvs = np.array([plain[s][1] for s in common])
        p = wilcoxon(pv, tv).pvalue if (pv != tv).any() else 1.0
        pp = wilcoxon(pvs, tvs).pvalue if (pvs != tvs).any() else 1.0
        d_igd = (tv.mean() - pv.mean()) / pv.mean() * 100
        d_igdp = (tvs.mean() - pvs.mean()) / pvs.mean() * 100
        rows.append(dict(problem=n, m=m, algo=a, d_igd=d_igd, d_igdp=d_igdp,
                         p_igd=p, p_igdp=pp, a12_igd=a12(pv, tv),
                         a12_igdp=a12(pvs, tvs)))
        print(f"{n:>6} m={m} {a:>6}: dIGD {d_igd:+7.1f}% (p={p:.4f}) | "
              f"dIGD+ {d_igdp:+7.1f}% (p={pp:.4f})", flush=True)

    with open(SUM_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "m", "algo", "d_igd",
                                          "d_igdp", "p_igd", "p_igdp",
                                          "a12_igd", "a12_igdp"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"明细已写 {SUM_CSV}")


if __name__ == "__main__":
    main()