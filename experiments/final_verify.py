"""
CLD-MaOEA 最终全量验证：确认 M 版精简配置（K.10）在多峰/多样性相关题上的最终效果。

覆盖范围（K.7/K.8/K.9 涉及的所有问题）：
- dtlz3 m=5/10   (K.7 逃逸机制首次突破)
- dtlz1 m=5/10   (K.8 多样性标定)
- wfg1-9 m=5/10  (K.9 WFG 全系列推广)

对比：H 版（基线，无逃逸） vs M 版（精简后，仅保留确认有效 escape overrides）。

运行: python experiments/final_verify.py [--seeds 3] [--variants H M]
"""
import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.m3_benchmark import make_problem, get_pf
from pymoo.indicators.igd import IGD
from src.algorithm import CLDMaOEA
from src.variants import CLDVariant

VARIANT_MAP = {
    "H": CLDVariant.HIGH_DIM,
    "M": CLDVariant.MULTI_MODAL,
}

SEEDS_ALL = [42, 7, 2024]
FE_MAX = 100000

# (problem, m) 覆盖 K.7/K.8/K.9 全部题
CASES = (
    [("dtlz3", 5), ("dtlz3", 10)]      # K.7 逃逸基准
    + [("dtlz1", 5), ("dtlz1", 10)]    # K.8
    + [(f"wfg{i}", m) for i in range(1, 10) for m in (5, 10)]  # K.9
)


def run_one(name, m, seed, variant):
    problem = make_problem(name, m)
    problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant=variant,
                   problem_type=problem_type, problem_name=name)
    t0 = time.perf_counter()
    X, F = alg.run(fe_max=FE_MAX)
    el = time.perf_counter() - t0
    pf = get_pf(name, m)
    igd = float(IGD(pf)(F))
    return igd, el


def main():
    ap = argparse.ArgumentParser(description="CLD-MaOEA 最终全量验证")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["H", "M"],
                    choices=list(VARIANT_MAP.keys()))
    ap.add_argument("--workers", type=int, default=8,
                    help="并行进程数（1=串行，保持原行为）")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="只跑指定案例（格式 name:m，如 dtlz5:5 dtlz7:10）；默认全部")
    args = ap.parse_args()

    seeds = SEEDS_ALL[:args.seeds]
    variants = [VARIANT_MAP[v] for v in args.variants]

    cases = CASES
    if args.cases:
        wanted = set()
        for c in args.cases:
            n, mm = c.rsplit(":", 1)
            wanted.add((n, int(mm)))
        unknown = wanted - set(CASES)
        if unknown:
            # 集外题（不在默认回归集）：直接放行，用于扩展验证
            pass
        cases = sorted(wanted, key=lambda t: CASES.index(t) if t in CASES else 999)

    print("=" * 84)
    print("CLD-MaOEA 最终全量验证（M 版精简配置）")
    print("=" * 84)
    print(f"版本: {args.variants}   案例: {len(cases)} 题   种子: {len(seeds)}"
          f"   workers={args.workers}")
    print()

    # 并行收集全部 (case, variant, seed) 结果
    raw = {}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(n, m, s, v) for (n, m) in cases for s in seeds for v in variants]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, *t): t for t in tasks}
            for fut in as_completed(futs):
                n, m, s, v = futs[fut]
                raw.setdefault((n, m, v), []).append(fut.result())
    else:
        for (n, m) in cases:
            for v in variants:
                for s in seeds:
                    raw.setdefault((n, m, v), []).append(run_one(n, m, s, v))

    header = f"{'case':<12} " + " ".join(
        f"{v:>16}" for v in args.variants) + "   变化      判定"
    print(header)
    print("-" * 84)

    results = {v: [] for v in args.variants}
    n_win = n_loss = n_tie = 0

    for name, m in cases:
        per_var = {}
        for v, vname in zip(variants, args.variants):
            igds = [r[0] for r in raw.get((name, m, v), [])]
            els = [r[1] for r in raw.get((name, m, v), [])]
            mu, sd = float(np.mean(igds)), float(np.std(igds))
            per_var[vname] = (mu, sd, np.mean(els) if els else 0.0)
            results[vname].append(mu)

        row = f"{name}{m:<6}"
        for vname in args.variants:
            mu, sd, _ = per_var[vname]
            row += f" {mu:>8.4f}±{sd:<5.3f}"

        # 变化百分比（以第一个版本为基线）
        base_mu = per_var[args.variants[0]][0]
        if len(args.variants) > 1:
            m_mu = per_var[args.variants[1]][0]
            pct = (m_mu - base_mu) / (abs(base_mu) + 1e-12) * 100
            if pct < -5:
                verdict = "✓ 改善"
                n_win += 1
            elif pct > 5:
                verdict = "✗ 退化"
                n_loss += 1
            else:
                verdict = "≈ 持平"
                n_tie += 1
            row += f"  {pct:+7.1f}%  {verdict}"
        print(row, flush=True)

    print("-" * 84)
    print("\n版本平均 IGD:")
    for vname in args.variants:
        print(f"  {vname}: {np.mean(results[vname]):.4f}")

    if len(args.variants) > 1:
        print(f"\n判定统计（{args.variants[1]} 较 {args.variants[0]}）: "
              f"{n_win} 胜 / {n_loss} 负 / {n_tie} 平")


if __name__ == "__main__":
    main()
