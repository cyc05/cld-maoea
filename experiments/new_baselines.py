"""
审稿回应：补充 2019 年后的新基线对比。

新增基线（全部来自 pymoo 0.6.2 内置实现，与主实验同种子同 FE=1e5 协议）：
  AGEMOEA  (Panichella, GECCO 2020)  —— 无需参考方向，几何生存分数
  AGEMOEA2 (Panichella, TEVC 2022)   —— AGEMOEA 的改进版
  UNSGA3   (Seada & Deb, TEVC 2020)  —— 带参考点的 U-NSGA-III

KnEA/MNEA/SRA/LMEA 等无 Python 官方实现，正文 Limitations 中说明。

产物：results/new_baselines.csv + results/new_baselines_summary.csv
运行：python experiments/new_baselines.py [--workers 8] [--smoke]
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

RES_CSV = os.path.join(ROOT, "experiments", "results", "new_baselines.csv")
SUM_CSV = os.path.join(ROOT, "experiments", "results", "new_baselines_summary.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")

PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]
SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
ALGOS = ["AGEMOEA", "AGEMOEA2", "UNSGA3"]
FIELDS = ["problem", "m", "algo", "seed", "igd", "igd_plus", "time"]


def run_task(task):
    sys.path.insert(0, ROOT)
    name, m, algo, seed = task
    fe_max = 100000
    from experiments.m3_benchmark import make_problem, get_pf, n_partitions
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions(m))
    N = int(np.ceil(len(ref_dirs) / 4) * 4)
    problem = make_problem(name, m)
    t0 = time.perf_counter()
    if algo == "AGEMOEA":
        from pymoo.algorithms.moo.age import AGEMOEA
        method = AGEMOEA(pop_size=N)
    elif algo == "AGEMOEA2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2
        method = AGEMOEA2(pop_size=N)
    else:  # UNSGA3
        from pymoo.algorithms.moo.unsga3 import UNSGA3
        method = UNSGA3(ref_dirs=ref_dirs, pop_size=N)
    res = minimize(problem, method, termination=("n_evals", fe_max),
                   seed=seed, verbose=False)
    elapsed = time.perf_counter() - t0
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = get_pf(name, m)
    return dict(problem=name, m=m, algo=algo, seed=seed,
                igd=float(IGD(pf)(res.F)), igd_plus=float(IGDPlus(pf)(res.F)),
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tasks = [(name, m, algo, seed)
             for name in PROBLEMS for m in M_VALUES
             for algo in ALGOS for seed in SEEDS]
    if args.smoke:
        tasks = [t for t in tasks if t[0] in ("dtlz1", "wfg4") and t[3] == 42]

    done = set()
    if os.path.exists(RES_CSV):
        with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["algo"], int(r["seed"])))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]
    print(f"新基线总任务 {len(tasks)} 条（已完成 {len(done)}），workers={args.workers}",
          flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.dirname(RES_CSV), exist_ok=True)
    header = not os.path.exists(RES_CSV)
    n_done = 0
    t_start = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, \
                open(RES_CSV, "a", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=FIELDS)
            if header:
                w.writeheader()
                fout.flush()
            futs = {pool.submit(run_task, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    w.writerow(fut.result())
                    fout.flush()
                except Exception as e:
                    print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                n_done += 1
                if n_done % 32 == 0 or n_done == len(futs):
                    rate = n_done / (time.time() - t_start)
                    eta = (len(futs) - n_done) / max(rate, 1e-9)
                    print(f"[progress] {n_done}/{len(futs)} ETA {eta/60:.1f} min",
                          flush=True)

    # ---- 与 CLD（m3_results.csv）种子配对对比 ----
    print("\n与 CLD 对照分析（种子配对，Wilcoxon p<0.05）", flush=True)
    from scipy.stats import wilcoxon
    cld = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                cld[(r["problem"], int(r["m"]))][int(r["seed"])] = (
                    float(r["igd"]), float(r["igd_plus"]))
    base = defaultdict(dict)
    with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            base[(r["problem"], int(r["m"]), r["algo"])][int(r["seed"])] = (
                float(r["igd"]), float(r["igd_plus"]))

    rows = {}
    for t in base:
        (name, m), algo = (t[0], t[1]), t[2]
        if (name, m) not in cld:
            continue
        seeds = sorted(set(base[t]) & set(cld[(name, m)]))
        cv = np.array([cld[(name, m)][s][0] for s in seeds])
        cvp = np.array([cld[(name, m)][s][1] for s in seeds])
        bv = np.array([base[t][s][0] for s in seeds])
        bvp = np.array([base[t][s][1] for s in seeds])
        p = wilcoxon(cv, bv).pvalue if (cv != bv).any() else 1.0
        pp = wilcoxon(cvp, bvp).pvalue if (cvp != bvp).any() else 1.0
        rows[t] = dict(
            problem=name, m=m, algo=algo,
            d_igd=(cv.mean() - bv.mean()) / bv.mean() * 100,   # CLD 相对基线
            d_igdp=(cvp.mean() - bvp.mean()) / bvp.mean() * 100,
            p_igd=p, p_igdp=pp, a12_igd=a12(cv, bv), a12_igdp=a12(cvp, bvp))

    with open(SUM_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "m", "algo", "d_igd",
                                          "d_igdp", "p_igd", "p_igdp",
                                          "a12_igd", "a12_igdp"])
        w.writeheader()
        for t in sorted(rows):
            w.writerow(rows[t])

    print("\n" + "=" * 60)
    print("新基线汇总：CLD vs 基线（d%<0 = CLD 更优；A12>0.5 = CLD 随机占优）")
    print("=" * 60)
    for algo in ALGOS:
        for metric, key, pkey in (("IGD", "d_igd", "p_igd"),
                                  ("IGD+", "d_igdp", "p_igdp")):
            wl = tl = ll = 0
            a12s = []
            for t, r in rows.items():
                if r["algo"] != algo:
                    continue
                a12s.append(r["a12_" + ("igd" if metric == "IGD" else "igdp")])
                if r[pkey] < 0.05:
                    if r[key] < 0:
                        wl += 1
                    else:
                        ll += 1
                else:
                    tl += 1
            print(f"{algo:>9} {metric:>4}: CLD 胜 {wl:2d} / 平 {tl:2d} / 负 {ll:2d}"
                  f" | A12 均值 {np.mean(a12s):.3f}")
    print(f"\n明细已写 {SUM_CSV}")


if __name__ == "__main__":
    main()