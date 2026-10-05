"""
审稿意见回应（下游）：配置器最优配置 vs 准则链配置的**留出评估**。

auto_config.py 在调参种子 [11,22] 上产出 randsearch / paramils 的最优配置；
本脚本在**评估种子 SEEDS10**（tuner 从未见过，与调参种子不相交）上评估这些
配置，并与准则链的按题决策配置（= K.26，即 m3_results.csv 中 "CLD" 行）配对对比。

对比口径（与主表 m3 协议一致，教训 29：配对种子、IGD/IGD+ 越小越好）：
  * 每实例每方法：tuner 最优配置 × 10 评估种子；
  * 与 CLD（准则链配置）逐实例 Wilcoxon + 汇总 wins/ties/losses + A12 效应量；
  * 若 tuner（RandomSearch 全局 / ParamILS 从准则链解出发）在同预算下不能
    稳定优于准则链，则证明准则链以零额外调参预算达到通用配置器水平。

运行：python experiments/eval_tuned.py [--workers 8]     （需先跑完 auto_config.py）
产物：experiments/results/tuner_eval.csv + stdout 汇总
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TUNER_CSV = os.path.join(ROOT, "experiments", "results", "tuner_configs.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "results", "tuner_eval.csv")

SEEDS10 = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]  # 与 m3_benchmark 一致
FE_MAX = 100000
METHODS = ["randsearch", "paramils"]


def load_best_cfgs():
    """读 tuner 最优配置 → {(problem, m, method): cfg_dict}"""
    best = {}
    with open(TUNER_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]), r["method"])
            # 每 (实例,方法) 一行最优（auto_config 已输出 best），直接解析
            best[key] = json.loads(r["cfg_json"])
    return best


def run_task(task):
    """单次评估：tuner 配置 × 单评估种子 → IGD/IGD+"""
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    name, m, method, cfg, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    pf = get_pf(name, m)
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=ptype, problem_name=name,
                   n_partitions=2 if m > 10 else None,
                   corner_archive=bool(cfg["corner_archive"]),
                   archive_out_mode=cfg["archive_out_mode"],
                   archive_policy=cfg["archive_policy"],
                   archive_budget=cfg["archive_budget"],
                   archive_union_cap=0.1,
                   rho_star=cfg["rho_star"], ki=cfg["ki"], kd=cfg["kd"])
    X, F = alg.run(fe_max=FE_MAX)
    return dict(problem=name, m=m, method=method, seed=seed,
                igd=float(IGD(pf)(F)), igd_plus=float(IGDPlus(pf)(F)))


def load_cld():
    """m3_results.csv 的 CLD（准则链配置）评估数据 → {(problem,m): {seed: igd+}}"""
    cld = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                cld[(r["problem"], int(r["m"]))][int(r["seed"])] = (
                    float(r["igd_plus"]))
    return cld


def a12(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :]); eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    best = load_best_cfgs()
    print(f"载入 tuner 最优配置：{len(best)} 条（实例×方法）", flush=True)

    # 断点续跑
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["method"], int(r["seed"])))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)

    tasks = []
    for (name, m, method), cfg in best.items():
        for s in SEEDS10:
            if (name, m, method, s) not in done:
                tasks.append((name, m, method, cfg, s))
    print(f"评估任务：{len(tasks)} 次运行（tuner 配置 × 10 评估种子），"
          f"workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fields = ["problem", "m", "method", "seed", "igd", "igd_plus"]
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

    # ---------------- 汇总对比 ----------------
    from scipy.stats import wilcoxon
    evals = defaultdict(dict)   # (problem,m,method) -> {seed: igd+}
    with open(OUT_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            evals[(r["problem"], int(r["m"]), r["method"])][int(r["seed"])] = \
                float(r["igd_plus"])
    cld = load_cld()

    print("\n" + "=" * 80)
    print("配置器最优配置 vs 准则链配置（评估种子 SEEDS10，IGD+ 越小越好）")
    print("=" * 80)
    for method in METHODS:
        win = tie = lose = 0
        a12s = []
        diffs = []
        for (name, m, meth), seedvals in sorted(evals.items()):
            if meth != method or (name, m) not in cld:
                continue
            common = sorted(set(seedvals) & set(cld[(name, m)]))
            if len(common) < 5:
                continue
            t = np.array([seedvals[s] for s in common])
            c = np.array([cld[(name, m)][s] for s in common])
            # tuner 更优 = t < c
            try:
                p = wilcoxon(t, c).pvalue if (t != c).any() else 1.0
            except Exception:
                p = 1.0
            a12s.append(a12(t, c))       # P(tuner < chain)
            diffs.append((t.mean() - c.mean()) / abs(c.mean()) * 100)
            if t.mean() < c.mean() and p < 0.05:
                win += 1                  # tuner 显著更优
            elif t.mean() > c.mean() and p < 0.05:
                lose += 1                 # 准则链显著更优
            else:
                tie += 1
        a12s = np.array(a12s); diffs = np.array(diffs)
        print(f"\n  {method:11s} vs 准则链：tuner 胜 {win} / 平 {tie} / 负 {lose}"
              f"（胜=tuner显著更优）")
        print(f"    A^12(tuner<chain) mean={a12s.mean():.3f} "
              f"median={np.median(a12s):.3f} | "
              f"ΔIGD+% mean={diffs.mean():+.2f} median={np.median(diffs):+.2f}")
    print("\n说明：tuner 胜少负多 → 准则链零调参决策不劣于同预算通用配置器。")


if __name__ == "__main__":
    main()
