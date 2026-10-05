"""
论文缺口 #3：泛化性基准（零调参直跑：无任何实例命中 multi_modal.json 的 override）。

背景：pymoo 0.6.2 不含 MaF 套件（升级 pymoo 会破坏 K.26 主表的外部基线可比性，拒绝）。
替代方案（均为 pymoo 官方实现，与 MaF1-7 同源的 DTLZ 变体族 + 维度扩展）：
  A. 几何变体 8 实例（m∈{5,10}）——覆盖 MaF 套件的核心几何特征：
     - dtlz1i  (InvertedDTLZ1)：线性倒置 PF（= MaF1 核心特征）
     - cdtlz2  (ConvexDTLZ2)：凸 PF（= MaF12 核心特征）
     - cdtlz4  (ConvexDTLZ4)：凸 + 偏置（= MaF5 核心特征）
     - sdtlz1  (ScaledDTLZ1)：badly-scaled 线性（= MaF4 核心特征）
  B. 维度扩展 10 实例（m∈{8,15}）——检验 CLD 对未调参维度的外推：
     - dtlz2/dtlz4/dtlz7（平滑凹/偏置/不连通）+ wfg5/wfg6（凹、无 override 命中）

零调参验证：上述实例名/m 组合均不命中 configs/multi_modal.json 的任何 m 级 override
（dtlz1i/cdtlz2/cdtlz4/sdtlz1 不在 override 表；dtlz2/dtlz4 无 override；
dtlz7 仅 m=10 有；wfg5/wfg6 无）——等价于"M 版基础配置直跑"，论文泛化性卖点。

d⊥ 映射（K.13：线性/凸关、凹/退化/不连通开）：dtlz1i/sdtlz1/cdtlz2/cdtlz4 → wfg（关）；
dtlz2/dtlz4/dtlz7 → dtlz（开）；wfg5/wfg6 → wfg（关）。

运行:
  python experiments/gen_benchmark.py --check          # 探测支持矩阵
  python experiments/gen_benchmark.py --seeds 3        # Stage A：3 种子快速筛查
  python experiments/gen_benchmark.py                  # Stage B：10 种子论文口径
  python experiments/m3_analyze.py --csv experiments/results/gen_results.csv
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_CSV = os.path.join(ROOT, "experiments", "results", "gen_results.csv")
FIELDS = ["problem", "m", "algo", "seed", "igd", "igd_plus", "time"]

# (脚本内名, pymoo 注册名, problem_type, n_var 公式)
# n_var 沿用 DTLZ 基类公式：DTLZ1 系 m+4，DTLZ2/4 系 m+9
CASES_DEF = [
    ("dtlz1i", "dtlz1^-1", "wfg", lambda m: m + 4),
    ("cdtlz2", "convex_dtlz2", "wfg", lambda m: m + 9),
    ("cdtlz4", "convex_dtlz4", "wfg", lambda m: m + 9),
    ("sdtlz1", "sdtlz1", "wfg", lambda m: m + 4),
    ("dtlz2", "dtlz2", "dtlz", lambda m: m + 9),
    ("dtlz4", "dtlz4", "dtlz", lambda m: m + 9),
    ("dtlz7", "dtlz7", "dtlz", lambda m: m + 19),
    ("wfg5", "wfg5", "wfg", lambda m: 2 * (m - 1) + 20),
    ("wfg6", "wfg6", "wfg", lambda m: 2 * (m - 1) + 20),
]
GEOM_M = [5, 10]      # A 组：几何变体
DIM_M = [8, 15]       # B 组：维度扩展

SEEDS_10 = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
ALGOS = ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA"]


def all_cases():
    """返回 [(脚本名, pymoo 名, ptype, m)]"""
    out = []
    for (alias, real, ptype, nv) in CASES_DEF:
        for m in (GEOM_M if alias in ("dtlz1i", "cdtlz2", "cdtlz4", "sdtlz1")
                  else DIM_M):
            out.append((alias, real, ptype, m))
    return out


def n_partitions(m):
    return 3 if m <= 5 else (2 if m <= 10 else 1)


def algo_ref_dirs(m):
    """算法种群用参考方向。m>10 时 das-dennis p=1 仅 16 方向（N=16 种群饿死，
    Stage A 证实为协议混杂而非算法差距），统一改用 p=2（m=15 → 120 方向，
    CLD 内部 n_partitions 与外部基线同源同量，对比公平）。"""
    from pymoo.util.ref_dirs import get_reference_directions
    return get_reference_directions("das-dennis", m, n_partitions=n_partitions(m) if m <= 10 else 2)


def make_problem(real_name, m):
    from pymoo.problems import get_problem
    nv = next(f for (a, r, t, f) in CASES_DEF if r == real_name)(m)
    return get_problem(real_name, n_var=nv, n_obj=m)


def get_pf(real_name, m, n=2000):
    """PF 采样：解析特判 → 无参 pareto_front() → das-dennis ref → 失败。

    解析特判依据：
    - dtlz1^-1：pymoo 0.6.2 的 _calc_pareto_front 引用不存在的 self.problem（bug）；
      由定义 F' = 0.5(1+g) - F、g=0、DTLZ1 PF=0.5*ref 推得 PF = 0.5*(1-ref)
    - wfg5/wfg6：基类迭代式 PF 生成在 m=15 爆内存；pymoo 源码注释给出解析式
      PF = generic_sphere(ref) * S = (ref/||ref||) * [2,4,...,2m]
    """
    from pymoo.util.ref_dirs import get_reference_directions
    prob = make_problem(real_name, m)

    def _ref_dirs():
        p_dense = 10 if m <= 5 else 3
        return get_reference_directions("das-dennis", m, n_partitions=p_dense)

    if real_name == "dtlz1^-1":
        return 0.5 * (1.0 - _ref_dirs())
    if real_name in ("wfg5", "wfg6"):
        ref = _ref_dirs()
        S = np.arange(2, 2 * m + 1, 2, dtype=float)
        return ref / np.linalg.norm(ref, axis=1, keepdims=True) * S

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
    return prob.pareto_front(_ref_dirs())


def run_task(task):
    """单次运行（子进程入口），协议与 m3_benchmark 一致"""
    sys.path.insert(0, ROOT)
    alias, real, ptype, m, algo, seed = task
    fe_max = 100000
    ref_dirs = algo_ref_dirs(m)
    n_ref = len(ref_dirs)
    N = int(np.ceil(n_ref / 4) * 4)
    problem = make_problem(real, m)

    t0 = time.perf_counter()
    if algo == "CLD":
        from src.algorithm import CLDMaOEA
        from src.variants import CLDVariant
        # 零调参直跑：variant 基础配置，problem_name=alias 不命中任何 override
        # m>10 显式 n_partitions=2 与外部基线方向集同源（120 方向），否则内部默认
        alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                       problem_type=ptype, problem_name=alias,
                       n_partitions=2 if m > 10 else None)
        X, F = alg.run(fe_max=fe_max)
    else:
        from pymoo.optimize import minimize
        if algo == "NSGA3":
            from pymoo.algorithms.moo.nsga3 import NSGA3
            method = NSGA3(ref_dirs=ref_dirs, pop_size=N)
        elif algo == "MOEAD":
            from pymoo.algorithms.moo.moead import MOEAD
            method = MOEAD(ref_dirs=ref_dirs, n_neighbors=15)
        elif algo == "RVEA":
            from pymoo.algorithms.moo.rvea import RVEA
            method = RVEA(ref_dirs=ref_dirs, pop_size=N)
        elif algo == "CTAEA":
            from pymoo.algorithms.moo.ctaea import CTAEA
            method = CTAEA(ref_dirs=ref_dirs)
        else:
            raise ValueError(algo)
        res = minimize(problem, method, termination=("n_evals", fe_max),
                       seed=seed, verbose=False)
        F = res.F
    elapsed = time.perf_counter() - t0

    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = get_pf(real, m)
    return dict(problem=alias, m=m, algo=algo, seed=seed,
                igd=float(IGD(pf)(F)), igd_plus=float(IGDPlus(pf)(F)),
                time=elapsed)


def check():
    """探测支持矩阵：构造 + PF 采样（统一走 get_pf，含解析特判）"""
    print(f"{'case':8s} {'pymoo 名':<14s} {'m':>3s} {'n_var':>5s} {'PF 形状':<10s} 状态")
    print("-" * 62)
    ok = []
    for (alias, real, ptype, m) in all_cases():
        try:
            prob = make_problem(real, m)
            nvar = prob.n_var
        except Exception as e:
            print(f"{alias:8s} {real:<14s} {m:>3d}     -  {'-':<10s} 构造失败 {type(e).__name__}")
            continue
        try:
            pf = np.atleast_2d(get_pf(real, m))
            shape = f"{pf.shape[0]}x{pf.shape[1]}"
            status = "OK"
            ok.append((alias, real, ptype, m))
        except Exception as e:
            shape, status = "-", f"PF 失败 {type(e).__name__}"
        print(f"{alias:8s} {real:<14s} {m:>3d} {nvar:>5d} {shape:<10s} {status}")
    print(f"\n可跑实例：{len(ok)}/{len(all_cases())}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只探测支持矩阵")
    ap.add_argument("--seeds", type=int, default=10, choices=[3, 10])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=RESULT_CSV)
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    args = ap.parse_args()

    if args.check:
        check()
        return

    seeds = SEEDS_10[:args.seeds]
    result_csv = args.csv
    supported = {(a, m) for (a, r, t, m) in check()}

    tasks = [(a, r, t, m, algo, seed)
             for (a, r, t, m) in all_cases() if (a, m) in supported
             for algo in args.algos for seed in seeds]

    done = set()
    if os.path.exists(result_csv):
        with open(result_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done.add((row["problem"], int(row["m"]), row["algo"], int(row["seed"])))
        print(f"已完成 {len(done)} 条，续跑剩余")
    tasks = [t for t in tasks if (t[0], t[3], t[4], t[5]) not in done]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    header = not os.path.exists(result_csv)
    n_done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(result_csv, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if header:
            writer.writeheader()
            fout.flush()
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)
                continue
            writer.writerow(row)
            fout.flush()
            n_done += 1
            if n_done % 20 == 0 or n_done == len(tasks):
                rate = n_done / (time.time() - t_start)
                eta = (len(tasks) - n_done) / max(rate, 1e-9)
                print(f"进度 {n_done}/{len(tasks)}  ETA {eta/60:.1f} min", flush=True)
    print(f"完成，结果写入 {result_csv}")


if __name__ == "__main__":
    main()
