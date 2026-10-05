"""
方案 A：判据链输出模式自适应验证。

背景：判据链当前仅决策"开/关档案"，输出模式固定为 replace。
当 S1 union 探针增益大但 S2 replace 验证失败时（尤其 S2③ 硬检查超标），
假设 replace 删除侧噪声是主因，union 模式可能通过 S2 验证。

目标实例（S1 阳性但 replace S2 拒绝的 held-out + 训练集已知边际）：
  held-out: cdtlz2 m=5/10, sdtlz1 m=5/10, dtlz7 m=8
  训练集  : dtlz6 m=10 (union 已知 -40.8% / replace -6.9%)

协议（与 criterion_replay / dtlz6_union_ablation 同款）：
  off     : corner_archive=False
  replace : policy="direction" + out_mode="replace" + budget=0.1
  union   : policy="direction" + out_mode="union" + union_cap=0.1
  10 种子同进程 off→replace→union 顺序运行

用法:
  python experiments/union_mode_adaptive.py --workers 8
"""
import os
import sys
import csv
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000

# S1 阳性但 replace S2 拒绝的实例
# (alias, real_name, problem_type, n_var)
INSTANCES = [
    ("cdtlz2", "convex_dtlz2", "wfg", 5 + 9, 5),
    ("cdtlz2", "convex_dtlz2", "wfg", 10 + 9, 10),
    ("sdtlz1", "sdtlz1", "wfg", 5 + 4, 5),
    ("sdtlz1", "sdtlz1", "wfg", 10 + 4, 10),
    ("dtlz7", "dtlz7", "dtlz", 8 + 19, 8),
    ("dtlz6", "dtlz6", "dtlz", 10 + 9, 10),
]

# S2 判据参数（与 criterion_replay 一致）
S2_P = 0.05
S2_SAMEDIR = 0.8
S2_HARD = 1.05


def _make_problem(real_name, m, n_var):
    from pymoo.problems import get_problem
    return get_problem(real_name, n_var=n_var, n_obj=m)


def _get_pf(real_name, m, n_var, n=2000):
    from pymoo.util.ref_dirs import get_reference_directions
    prob = _make_problem(real_name, m, n_var)
    p_dense = 10 if m <= 5 else 3
    ref = get_reference_directions("das-dennis", m, n_partitions=p_dense)
    try:
        pf = prob.pareto_front()
        if pf is not None and len(pf):
            pf = np.atleast_2d(pf)
            if len(pf) > n:
                idx = np.random.default_rng(0).choice(len(pf), size=n, replace=False)
                pf = pf[idx]
            return pf
    except Exception:
        pass
    return prob.pareto_front(ref)


def run_task(task):
    sys.path.insert(0, ROOT)
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA

    alias, real, ptype, n_var, m, seed = task
    problem = _make_problem(real, m, n_var)
    pf = _get_pf(real, m, n_var)
    kw = dict(m=m, seed=seed, variant="multi_modal",
              problem_type=ptype, problem_name=alias,
              n_partitions=2 if m > 10 else None)

    res = {}
    # off → replace → union（同进程顺序）
    for tag, arc, out_mode, extra in [
        ("off", False, None, {}),
        ("replace", True, "replace", {"archive_budget": 0.1}),
        ("union", True, "union", {"archive_union_cap": 0.1}),
    ]:
        alg = CLDMaOEA(problem, corner_archive=arc,
                       archive_policy="direction",
                       archive_out_mode=out_mode, **extra, **kw)
        _, F = alg.run(fe_max=FE_MAX)
        res[tag] = (float(IGD(pf)(F)), float(IGDPlus(pf)(F)))

    return (alias, m, seed,
            res["off"][0], res["off"][1],
            res["replace"][0], res["replace"][1],
            res["union"][0], res["union"][1])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_CSV = os.path.join(ROOT, "experiments", "results",
                           "union_mode_adaptive.csv")

    # 断点续跑
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), int(r["seed"])))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)

    tasks = [(a, r, t, nv, m, s)
             for (a, r, t, nv, m) in INSTANCES for s in SEEDS
             if (a, m, s) not in done]
    n_total = len(INSTANCES) * len(SEEDS)
    print(f"方案A：{len(INSTANCES)} 实例 × {len(SEEDS)} 种子 = "
          f"{n_total} 配对（剩余 {len(tasks)}），每配对 3 次运行，"
          f"workers={args.workers}", flush=True)

    fields = ["problem", "m", "seed",
              "igd_off", "igdp_off",
              "igd_replace", "igdp_replace",
              "igd_union", "igdp_union"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    header = not os.path.exists(OUT_CSV)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(OUT_CSV, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        if header:
            writer.writeheader()
            fout.flush()
        futs = {pool.submit(run_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                writer.writerow(dict(zip(fields, fut.result())))
                fout.flush()
            except Exception as e:
                t = futs[fut]
                print(f"[FAIL] {t[0]} m={t[4]} seed={t[5]}: "
                      f"{type(e).__name__} {e}", flush=True)
            if i % 10 == 0 or i == len(futs):
                print(f"[progress] {i}/{len(futs)}", flush=True)

    print(f"\n明细已存 {OUT_CSV}")

    # 分析
    from scipy.stats import wilcoxon
    pairs = {}
    with open(OUT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            pairs.setdefault((r["problem"], int(r["m"])), []).append(r)

    print("\n" + "=" * 110)
    print("方案 A：union vs replace S2 判据对比")
    print("=" * 110)
    print(f"| {'实例':>12} {'m':>3} | {'模式':>7} | "
          f"{'ΔIGD%':>7} {'p_igd':>7} | {'ΔIGD+%':>7} {'p_igdp':>7} | "
          f"{'同向':>5} {'最差比':>6} | S2① S2② S2③ | 判定")
    print("-" * 110)

    rescue_count = 0
    for (alias, real, ptype, n_var, m) in INSTANCES:
        rows = pairs.get((alias, m), [])
        if len(rows) < len(SEEDS):
            print(f"| {alias:>12} {m:>3} | 数据不足（{len(rows)}/{len(SEEDS)}）")
            continue
        arr = np.array([(float(r["igd_off"]), float(r["igdp_off"]),
                         float(r["igd_replace"]), float(r["igdp_replace"]),
                         float(r["igd_union"]), float(r["igdp_union"]))
                        for r in rows])
        igd_o, igdp_o = arr[:, 0], arr[:, 1]
        verdicts = {}
        for tag, col_igd, col_igdp in [("replace", 2, 3), ("union", 4, 5)]:
            n_igd, n_igdp = arr[:, col_igd], arr[:, col_igdp]
            d_igd = (n_igd - igd_o) / np.abs(igd_o) * 100
            d_igdp = (n_igdp - igdp_o) / np.abs(igdp_o) * 100
            try:
                p1 = wilcoxon(igd_o, n_igd).pvalue if (igd_o != n_igd).any() else 1.0
                p2 = wilcoxon(igdp_o, n_igdp).pvalue if (igdp_o != n_igdp).any() else 1.0
            except Exception:
                p1 = p2 = 1.0
            ok1 = (d_igd.mean() < 0 and d_igdp.mean() < 0 and p1 < S2_P and p2 < S2_P)
            rate = float((d_igdp < 0).mean())
            ok2 = rate >= S2_SAMEDIR
            worst = float((n_igd / igd_o).max())
            ok3 = worst <= S2_HARD
            ok = ok1 and ok2 and ok3
            verdicts[tag] = ok
            print(f"| {alias:>12} {m:>3} | {tag:>7} | "
                  f"{d_igd.mean():>+6.1f}% {p1:>7.4f} | "
                  f"{d_igdp.mean():>+6.1f}% {p2:>7.4f} | "
                  f"{int((d_igdp<0).sum()):>2}/10  {worst:>5.3f}x | "
                  f"{'✓' if ok1 else '✗'}  {'✓' if ok2 else '✗'}  "
                  f"{'✓' if ok3 else '✗'} | "
                  f"{'启用' if ok else '关闭'}")

        if verdicts.get("union") and not verdicts.get("replace"):
            rescue_count += 1
            print(f"| {'':>12} {'':>3} | → union 救活（replace 拒绝但 union 通过）")
        print("-" * 110)

    print(f"\n结论：{rescue_count}/{len(INSTANCES)} 实例被 union 救活")
    print("（union 通过但 replace 拒绝 → 判据链应增加输出模式自适应分支）")


if __name__ == "__main__":
    main()
