"""
Phase 2 问题5：cProfile 三机制时间分解。

对 CLD-MaOEA 单次运行做 cProfile 插桩，按三机制 + 其他归类统计运行时间占比：
  M1 闭环支配控制：dominance_matrix + strengthened_nd_sort + controller.update
  M2 方向分化交配：direction_differentiation
  M3 R2 贪心截断：r2_greedy_truncation + compute_theta_w
  其他：problem.evaluate + SBX + PM + archive + 归一化等

代表实例：dtlz3 m=5（多峰）、dtlz6 m=10（退化 PF）、wfg3 m=10（不连通），
每个实例 3 种子取均值。

运行: python experiments/cprofile_decompose.py [--workers 3]
产物: 打印时间分解表 → results/cprofile_decompose.csv
"""
import os
import sys
import csv
import cProfile
import pstats
import io
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FE_MAX = 100000
CASES = [("dtlz3", 5), ("dtlz6", 10), ("wfg3", 10)]
SEEDS = [42, 7, 2024]

# 函数名 → 机制映射（cProfile 函数名匹配子串）
MAPPING = [
    ("M1 闭环支配控制", [
        "dominance_matrix", "strengthened_nd_sort", "estimate_nadir_extreme",
        "update",  # controller.update
    ]),
    ("M2 方向分化交配", [
        "direction_differentiation",
    ]),
    ("M3 R2贪心截断", [
        "r2_greedy_truncation", "compute_theta_w",
    ]),
    ("遗传算子(SBX+PM)", [
        "sbx_crossover", "polynomial_mutation",
    ]),
    ("问题评估", [
        "evaluate",
    ]),
    ("角点档案", [
        "observe", "finalize",
    ]),
]


def match_category(func_name):
    for cat, patterns in MAPPING:
        for pat in patterns:
            if pat in func_name:
                return cat
    return None


def run_profiled(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant

    name, m, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"

    pr = cProfile.Profile()
    alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                   problem_type=ptype, problem_name=name)
    pr.enable()
    alg.run(fe_max=FE_MAX)
    pr.disable()

    # 提取统计
    stats = pstats.Stats(pr)
    # 按 cumtime 汇总到类别
    cat_times = {}
    total_time = 0.0
    for (filename, line, funcname), (cc, nc, tt, ct, callers) in stats.stats.items():
        # tt = tottime (self time), ct = cumtime
        cat = match_category(funcname)
        if cat is None:
            cat = "其他"
        cat_times[cat] = cat_times.get(cat, 0.0) + tt  # 用 self time 避免重复计
        total_time += tt

    return name, m, seed, cat_times, total_time


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    tasks = [(n, m, s) for (n, m) in CASES for s in SEEDS]
    all_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_profiled, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                name, m, seed, cat_times, total = fut.result()
                all_results.setdefault((name, m), []).append(
                    (seed, cat_times, total))
                print(f"[done] {name} m={m} seed={seed} total={total:.2f}s",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)

    # 汇总：每实例取 3 种子均值
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    all_cats = [cat for cat, _ in MAPPING] + ["其他"]
    lines = []
    lines.append("=" * 90)
    lines.append("cProfile 三机制时间分解（self time, 3 种子均值）")
    lines.append("=" * 90)
    header = f"| {'实例':>10} {'m':>3} | {'总时间(s)':>8} | " + " | ".join(
        f"{c:>14}" for c in all_cats) + " |"
    lines.append(header)
    lines.append("|" + "-" * 88 + "|")

    csv_rows = []
    for (name, m) in CASES:
        runs = all_results.get((name, m), [])
        if not runs:
            lines.append(f"| {name:>10} {m:>3} | {'FAIL':>8} | " +
                        " | ".join(f"{'—':>14}" for _ in all_cats) + " |")
            continue
        total_mean = np.mean([t for _, _, t in runs])
        cat_means = {}
        for cat in all_cats:
            vals = [ct.get(cat, 0.0) for _, ct, _ in runs]
            cat_means[cat] = np.mean(vals)

        row = f"| {name:>10} {m:>3} | {total_mean:>8.2f} | " + " | ".join(
            f"{cat_means[c]:>8.2f}s({cat_means[c]/total_mean*100:>4.1f}%)" if total_mean > 0
            else f"{'0.00':>14}" for c in all_cats) + " |"
        lines.append(row)
        csv_rows.append({
            "problem": name, "m": m, "total_s": total_mean,
            **{c: cat_means.get(c, 0.0) for c in all_cats}
        })

    # 跨实例汇总
    lines.append("|" + "-" * 88 + "|")
    all_totals = [t for runs in all_results.values() for _, _, t in runs]
    grand_total = np.mean(all_totals) if all_totals else 0
    grand_cats = {}
    for cat in all_cats:
        vals = [ct.get(cat, 0.0) for runs in all_results.values()
                for _, ct, _ in runs]
        grand_cats[cat] = np.mean(vals)
    lines.append(f"| {'平均':>10} {'':>3} | {grand_total:>8.2f} | " + " | ".join(
        f"{grand_cats[c]:>8.2f}s({grand_cats[c]/grand_total*100:>4.1f}%)" if grand_total > 0
        else f"{'0.00':>14}" for c in all_cats) + " |")
    lines.append("=" * 90)

    text = "\n".join(lines)
    print(text)

    out_csv = os.path.join(ROOT, "experiments", "results",
                           "cprofile_decompose.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "m", "total_s"] + all_cats)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSAVED {out_csv}")


if __name__ == "__main__":
    main()
