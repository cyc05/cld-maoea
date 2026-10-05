"""
CLD-MaOEA 四版本对比测试脚本。
运行: python experiments/variant_compare.py [--variants H L M S] [--cases dtlz1-5 wfg1-10]
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
from src.variants import CLDVariant, get_variant_config, auto_variant, list_variants

VARIANT_MAP = {
    "H": CLDVariant.HIGH_DIM,
    "L": CLDVariant.LOW_DIM,
    "M": CLDVariant.MULTI_MODAL,
    "S": CLDVariant.SPEED,
}

CASES_PRESETS = {
    "dtlz1-5": [("dtlz1", 5), ("dtlz2", 5), ("dtlz3", 5), ("dtlz4", 5), ("dtlz5", 5)],
    "dtlz1-10": [("dtlz1", 10), ("dtlz2", 10), ("dtlz3", 10), ("dtlz4", 10), ("dtlz5", 10)],
    "dtlz7": [("dtlz7", 5), ("dtlz7", 10)],
    "wfg1-5": [("wfg1", 5), ("wfg2", 5), ("wfg3", 5), ("wfg4", 5), ("wfg5", 5)],
    "wfg1-10": [("wfg1", 10), ("wfg2", 10), ("wfg3", 10), ("wfg4", 10), ("wfg5", 10)],
    "wfg6-10": [("wfg6", 10), ("wfg7", 10), ("wfg8", 10), ("wfg9", 10)],
    "multi_modal": [("dtlz3", 5), ("dtlz3", 10), ("dtlz7", 5), ("dtlz7", 10),
                     ("wfg7", 5), ("wfg7", 10), ("wfg9", 5), ("wfg9", 10)],
    "high_dim": [("dtlz1", 10), ("dtlz2", 10), ("dtlz3", 10), ("dtlz4", 10),
                  ("dtlz5", 10), ("dtlz6", 10), ("dtlz7", 10),
                  ("wfg1", 10), ("wfg2", 10), ("wfg3", 10), ("wfg4", 10),
                  ("wfg5", 10), ("wfg6", 10), ("wfg7", 10), ("wfg8", 10), ("wfg9", 10)],
    "low_dim": [("dtlz1", 5), ("dtlz2", 5), ("dtlz3", 5), ("dtlz4", 5),
                 ("dtlz5", 5), ("dtlz6", 5), ("dtlz7", 5),
                 ("wfg1", 5), ("wfg2", 5), ("wfg3", 5), ("wfg4", 5),
                 ("wfg5", 5), ("wfg6", 5), ("wfg7", 5), ("wfg8", 5), ("wfg9", 5)],
    "all": [],  # 动态生成
}

SEEDS = [42, 7, 2024]
FE_MAX = 100000


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
    ap = argparse.ArgumentParser(description="CLD-MaOEA 版本对比测试")
    ap.add_argument("--variants", nargs="+", default=["H", "L", "M", "S"],
                    choices=list(VARIANT_MAP.keys()),
                    help="测试版本：H(高维) L(低维) M(多峰) S(速度)")
    ap.add_argument("--cases", nargs="+", default=["multi_modal"],
                    choices=list(CASES_PRESETS.keys()),
                    help="测试案例集")
    ap.add_argument("--seeds", type=int, default=3,
                    help="种子数（默认 3）")
    ap.add_argument("--auto", action="store_true",
                    help="使用 auto_variant 自动推荐而非手动指定")
    args = ap.parse_args()

    # 生成案例列表
    cases = []
    for preset in args.cases:
        if preset == "all":
            cases.extend([(f"dtlz{i}", m) for i in range(1, 8) for m in (5, 10)])
            cases.extend([(f"wfg{i}", m) for i in range(1, 10) for m in (5, 10)])
        else:
            cases.extend(CASES_PRESETS[preset])
    # 去重
    cases = list(dict.fromkeys(cases))

    variants = [VARIANT_MAP[v] for v in args.variants]
    seeds = SEEDS[:args.seeds]

    print("=" * 80)
    print("CLD-MaOEA 版本对比测试")
    print("=" * 80)
    print(f"版本: {[v for v in args.variants]}")
    print(f"案例: {args.cases} ({len(cases)} 个)")
    print(f"种子: {len(seeds)} 个")
    print()

    # 版本描述
    for v in variants:
        # 用 m=5 和 problem_type="dtlz" 作为示例展示配置
        cfg = get_variant_config(v, m=5, problem_type="dtlz", problem_name="dtlz1")
        print(f"[{v}] {list_variants()[v]}")
        print(f"    配置: protect={cfg['protect_mode']}, lag_thresh={cfg['lag_thresh']}, "
              f"d_perp={cfg['d_perp_tiebreak']}, stall={cfg['stall_patience']}, "
              f"phi_relax={cfg['phi_relax']}")
    print()

    # 表头
    header = f"{'case':<12}"
    for v in args.variants:
        header += f" {v+' IGD':>18} {v+' time':>8}"
    header += "  最优版本"
    print(header)
    print("-" * 80)

    results = {v: [] for v in args.variants}
    for name, m in cases:
        row = f"{name}{m:<6}"
        best_igd = float("inf")
        best_v = None
        for v, vname in zip(variants, args.variants):
            igds, els = [], []
            for s in seeds:
                if args.auto:
                    v_actual = auto_variant(name, m, make_problem(name, m).n_var)
                else:
                    v_actual = v
                igd, el = run_one(name, m, s, v_actual)
                igds.append(igd)
                els.append(el)
            mu, sd = np.mean(igds), np.std(igds)
            t_mu = np.mean(els)
            row += f" {mu:.4f}±{sd:.3f}  {t_mu:>6.1f}s"
            results[vname].append(mu)
            if mu < best_igd:
                best_igd = mu
                best_v = vname
        row += f"  {best_v}"
        print(row)

    print("-" * 80)
    # 汇总
    print("\n版本平均 IGD:")
    for v in args.variants:
        avg = np.mean(results[v])
        print(f"  {v}: {avg:.4f}")
    print(f"\n最优版本: {min(args.variants, key=lambda v: np.mean(results[v]))}")


if __name__ == "__main__":
    main()
