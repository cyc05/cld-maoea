"""
审稿回应：M1/M2/M3 两两组合消融。

M1 = 闭环支配控制（PD 控制 phi + 收敛门控 + 逃逸模式）
M2 = 方向分化交配（φ_x 按方向滞后差异化）
M3 = R2 贪心截断（含小生境保护；off = 临界层纯拥挤距离截断）

变体（基于 K.26 multi_modal 配置，corner_archive/逃逸等 overrides 保持不变）：
  M1  = 仅闭环控制（M2 off: use_differentiation=False；M3 off: r2_mode="crowding"）
  M12 = M1+M2（M3 off）
  M13 = M1+M3（M2 off）
  M23 = M2+M3（M1 off: fixed_phi=0.5 标准支配，门控/逃逸随之关闭）
完整版 F = m3_results.csv 的 CLD 行（同种子、同 FE=1e5 预算，不重跑）。

产物：results/module_ablation.csv（1280 行）+ results/module_ablation_summary.csv
运行：python experiments/module_ablation.py [--workers 8] [--smoke]
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

RES_CSV = os.path.join(ROOT, "experiments", "results", "module_ablation.csv")
SUM_CSV = os.path.join(ROOT, "experiments", "results", "module_ablation_summary.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")

PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]
SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]

# 变体 -> 覆盖参数（在 MULTI_MODAL 配置之上）
CONF_SPEC = {
    "M1": {"use_differentiation": False, "r2_mode": "crowding"},
    "M12": {"r2_mode": "crowding"},
    "M13": {"use_differentiation": False},
    "M23": {"fixed_phi": 0.5},
}
FIELDS = ["problem", "m", "conf", "seed", "igd", "igd_plus", "time"]


def run_task(task):
    sys.path.insert(0, ROOT)
    name, m, conf, seed = task
    fe_max = 100000
    from experiments.m3_benchmark import make_problem, get_pf
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    t0 = time.perf_counter()
    alg = CLDMaOEA(problem, m=m, seed=seed, problem_type=problem_type,
                   problem_name=name, variant=CLDVariant.MULTI_MODAL,
                   **CONF_SPEC[conf])
    X, F = alg.run(fe_max=fe_max)
    elapsed = time.perf_counter() - t0
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = get_pf(name, m)
    return dict(problem=name, m=m, conf=conf, seed=seed,
                igd=float(IGD(pf)(F)), igd_plus=float(IGDPlus(pf)(F)),
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

    tasks = [(name, m, conf, seed)
             for name in PROBLEMS for m in M_VALUES
             for conf in CONF_SPEC for seed in SEEDS]
    if args.smoke:
        tasks = [t for t in tasks if t[0] in ("dtlz1", "wfg4") and t[3] == 42]

    done = set()
    if os.path.exists(RES_CSV):
        with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["conf"], int(r["seed"])))
    tasks = [t for t in tasks if (t[0], t[1], t[2], t[3]) not in done]
    print(f"模块消融总任务 {len(tasks)} 条（已完成 {len(done)}），workers={args.workers}",
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

    # ---- 对照完整版 F（m3_results.csv 的 CLD 行，按种子配对） ----
    print("\n开始与完整版对照分析（种子配对，Wilcoxon p<0.05）", flush=True)
    from scipy.stats import wilcoxon
    full = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                full[(r["problem"], int(r["m"]))][int(r["seed"])] = (
                    float(r["igd"]), float(r["igd_plus"]))
    abl = defaultdict(dict)
    with open(RES_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            abl[(r["problem"], int(r["m"]), r["conf"])][int(r["seed"])] = (
                float(r["igd"]), float(r["igd_plus"]))

    rows = {}
    for t in abl:
        (name, m), conf = (t[0], t[1]), t[2]
        if (name, m) not in full:
            continue
        seeds = sorted(set(abl[t]) & set(full[(name, m)]))
        fv = np.array([full[(name, m)][s][0] for s in seeds])
        fvp = np.array([full[(name, m)][s][1] for s in seeds])
        av = np.array([abl[t][s][0] for s in seeds])
        avp = np.array([abl[t][s][1] for s in seeds])
        p = wilcoxon(fv, av).pvalue if (fv != av).any() else 1.0
        pp = wilcoxon(fvp, avp).pvalue if (fvp != avp).any() else 1.0
        rows[t] = dict(
            problem=name, m=m, conf=conf,
            d_igd=(av.mean() - fv.mean()) / fv.mean() * 100,
            d_igdp=(avp.mean() - fvp.mean()) / fvp.mean() * 100,
            p_igd=p, p_igdp=pp, a12_igd=a12(fv, av), a12_igdp=a12(fvp, avp))

    with open(SUM_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "m", "conf", "d_igd",
                                          "d_igdp", "p_igd", "p_igdp",
                                          "a12_igd", "a12_igdp"])
        w.writeheader()
        for t in sorted(rows):
            w.writerow(rows[t])

    # 打印汇总：每变体 vs F 的 W/T/L（IGD）
    print("\n" + "=" * 60)
    print("两两消融汇总：各变体与完整版 F 对比（d% = 变体相对 F 的 IGD 均值差）")
    print("=" * 60)
    for conf in CONF_SPEC:
        wl = tl = ll = 0
        ds = []
        for t, r in rows.items():
            if r["conf"] != conf:
                continue
            ds.append(r["d_igd"])
            if r["p_igd"] < 0.05:
                if r["d_igd"] < 0:
                    wl += 1
                else:
                    ll += 1
            else:
                tl += 1
        print(f"{conf:>4}: dIGD 均值 {np.mean(ds):+7.2f}% | 胜过F {wl:2d} / 平 {tl:2d} / 劣于F {ll:2d}")
    print(f"\n明细已写 {SUM_CSV}")


if __name__ == "__main__":
    main()