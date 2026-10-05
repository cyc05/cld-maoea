"""
第二轮审稿意见2的实证回应：等计算预算（墙钟）对比。

审稿人指出：CLD 每代做更多工作（支配矩阵/方向分化/R2截断），同 FE 下墙钟
更慢（NSGA-III 的 2.7x、RVEA 的 4x），故同 FE 对比对轻量基线不公平。
建议：给基线更多 FE 使其墙钟与 CLD 相当，再比精度。

设计（逐实例等墙钟归一化）：
  对每实例，baseline 的 FE 预算 = round(CLD_mean_time / baseline_mean_time * 1e5)
  —— 即给 NSGA-III / RVEA 足够 FE 使其单次墙钟 ≈ CLD 的 15.7s。
  仅对比这两个比 CLD 快且精度相当/更优的基线（MOEA/D、C-TAEA 已比 CLD 慢，
  等墙钟只会进一步减少其 FE，无意义）。
  然后与 CLD@1e5（= m3_results.csv 的 CLD 行）在 10 种子上比 IGD/IGD+。

  若 CLD@1e5 在等墙钟预算下仍不劣于这些基线，则说明其优势来自机制设计而
  非计算量堆砌；否则诚实揭示优势部分来自额外计算。

运行：python experiments/equal_budget.py [--workers 8]
产物：experiments/results/equal_budget.csv + stdout 对比
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "results", "equal_budget.csv")
SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
BASE_ALGOS = ["NSGA3", "RVEA"]       # 仅对比比 CLD 快的基线
BASE_FE = 100000
PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]


def mean_times():
    """每实例每算法的 mean 墙钟时间 → {(problem,m,algo): mean_time}"""
    acc = defaultdict(list)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            acc[(r["problem"], int(r["m"]), r["algo"])].append(float(r["time"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf, n_partitions
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    import time as _time
    name, m, algo, seed, fe = task
    problem = make_problem(name, m)
    ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions(m))
    N = int(np.ceil(len(ref_dirs) / 4) * 4)
    t0 = _time.perf_counter()
    if algo == "NSGA3":
        from pymoo.algorithms.moo.nsga3 import NSGA3
        method = NSGA3(ref_dirs=ref_dirs, pop_size=N)
    else:  # RVEA
        from pymoo.algorithms.moo.rvea import RVEA
        method = RVEA(ref_dirs=ref_dirs, pop_size=N)
    res = minimize(problem, method, termination=("n_evals", int(fe)),
                   seed=seed, verbose=False)
    el = _time.perf_counter() - t0
    pf = get_pf(name, m)
    return dict(problem=name, m=m, algo=algo, seed=seed, fe=int(fe),
                igd=float(IGD(pf)(res.F)), igd_plus=float(IGDPlus(pf)(res.F)),
                time=el)


def a12(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :]); eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    mt = mean_times()
    tasks = []
    for name in PROBLEMS:
        for m in M_VALUES:
            cl_t = mt.get((name, m, "CLD"))
            if cl_t is None:
                continue
            for algo in BASE_ALGOS:
                bt = mt.get((name, m, algo))
                if bt is None or bt <= 0:
                    continue
                fe = int(round(cl_t / bt * BASE_FE))
                for s in SEEDS:
                    tasks.append((name, m, algo, s, fe))

    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["algo"], int(r["seed"])))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]
    print(f"等预算对比：{len(tasks)} 次运行（NSGA3/RVEA @ 等墙钟归一化 FE），"
          f"workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fields = ["problem", "m", "algo", "seed", "fe", "igd", "igd_plus", "time"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    header = not os.path.exists(OUT_CSV)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool, \
                open(OUT_CSV, "a", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=fields)
            if header:
                w.writeheader(); fout.flush()
            futs = {pool.submit(run_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    w.writerow(fut.result()); fout.flush()
                except Exception as e:
                    print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
                if i % 40 == 0 or i == len(futs):
                    print(f"[progress] {i}/{len(futs)}", flush=True)

    # ---- 对比分析：CLD@1e5 vs 基线@等墙钟 ----
    from scipy.stats import wilcoxon
    eq = defaultdict(dict)
    with open(OUT_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            eq[(r["problem"], int(r["m"]), r["algo"])][int(r["seed"])] = \
                (float(r["igd"]), float(r["igd_plus"]), int(r["fe"]))
    cld = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                cld[(r["problem"], int(r["m"]))][int(r["seed"])] = \
                    (float(r["igd"]), float(r["igd_plus"]))

    for metric_idx, metric in ((0, "IGD"), (1, "IGD+")):
        print("\n" + "=" * 74)
        print(f"等计算预算对比 — 指标 {metric}（基线已用足与 CLD 相同的墙钟）")
        print("=" * 74)
        for algo in BASE_ALGOS:
            win = tie = lose = 0
            a12s = []
            fes = []
            for (name, m) in sorted(cld, key=lambda k: (k[1], k[0])):
                if (name, m, algo) not in eq:
                    continue
                common = sorted(set(cld[(name, m)]) & set(eq[(name, m, algo)]))
                if len(common) < 5:
                    continue
                cv = np.array([cld[(name, m)][s][metric_idx] for s in common])
                bv = np.array([eq[(name, m, algo)][s][metric_idx] for s in common])
                fes.append(eq[(name, m, algo)][common[0]][2])
                try:
                    p = wilcoxon(cv, bv).pvalue if (cv != bv).any() else 1.0
                except Exception:
                    p = 1.0
                a12s.append(a12(cv, bv))   # P(CLD < baseline@equal-budget)
                if cv.mean() < bv.mean() and p < 0.05:
                    win += 1
                elif cv.mean() > bv.mean() and p < 0.05:
                    lose += 1
                else:
                    tie += 1
            print(f"  CLD vs {algo}@等墙钟(平均FE≈{int(np.mean(fes))/1000:.0f}k): "
                  f"胜 {win} / 平 {tie} / 负 {lose} | A12(CLD优)={np.mean(a12s):.3f}")


if __name__ == "__main__":
    main()
