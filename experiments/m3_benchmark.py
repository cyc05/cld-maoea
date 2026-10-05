"""
M3 全基准测试：DTLZ1-7 + WFG1-9, m ∈ {5, 10}, 10 种子。
算法: CLD-MaOEA(ours) / NSGA-III / MOEA-D / RVEA / C-TAEA
指标: IGD, IGD+, 墙钟时间; 结果增量写入 experiments/results/m3_results.csv (支持断点续跑)

运行: python experiments/m3_benchmark.py [--smoke]
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULT_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
FIELDS = ["problem", "m", "algo", "seed", "igd", "igd_plus", "time"]

PROBLEMS = (
    [("dtlz%d" % i) for i in range(1, 8)] +
    [("wfg%d" % i) for i in range(1, 10)]
)
M_VALUES = [5, 10]
SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
ALGOS = ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA", "CLDBandit"]


def n_partitions(m):
    return 3 if m <= 5 else (2 if m <= 10 else 1)


def make_problem(name, m):
    from pymoo.problems import get_problem
    if name == "dtlz1":
        n_var = m + 4
    elif name.startswith("dtlz"):
        n_var = m + 9 if name != "dtlz7" else m + 19
    else:  # wfg
        k = 2 * (m - 1)
        n_var = k + 20
    return get_problem(name, n_var=n_var, n_obj=m)


def get_pf(name, m, n=2000):
    """获取参考 PF 采样（IGD/IGD+ 用）"""
    from pymoo.util.ref_dirs import get_reference_directions
    prob = make_problem(name, m)
    if name in ("dtlz5", "dtlz6", "dtlz7"):
        pf = prob.pareto_front()
        idx = np.random.default_rng(0).choice(len(pf), size=min(n, len(pf)), replace=False)
        return pf[idx]
    p_dense = 10 if m <= 5 else 3
    ref = get_reference_directions("das-dennis", m, n_partitions=p_dense)
    return prob.pareto_front(ref)


def run_task(task):
    """单次运行（子进程入口）"""
    sys.path.insert(0, ROOT)
    name, m, algo, seed = task
    fe_max = 100000
    p = n_partitions(m)
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", m, n_partitions=p)
    n_ref = len(ref_dirs)
    N = int(np.ceil(n_ref / 4) * 4)
    problem = make_problem(name, m)

    t0 = time.perf_counter()
    if algo in ("CLD", "CLDBandit"):
        from src.algorithm import CLDMaOEA
        from src.variants import CLDVariant
        problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
        # K.26 基线：M 版完整配置（configs/multi_modal.json，含逃逸/niche_fill/corner_archive overrides）
        # CLDBandit：审稿对比基线，phi 改由 FRRMAB 在线选臂（取代 PD 闭环控制），其余同 CLD
        alg = CLDMaOEA(problem, m=m, seed=seed, problem_type=problem_type,
                       problem_name=name, variant=CLDVariant.MULTI_MODAL,
                       phi_bandit=(algo == "CLDBandit"))
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

    # 指标（每个任务独立取 PF，避免跨进程序列化大对象）
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    pf = get_pf(name, m)
    igd = IGD(pf)(F)
    igdp = IGDPlus(pf)(F)
    return dict(problem=name, m=m, algo=algo, seed=seed,
                igd=float(igd), igd_plus=float(igdp), time=elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="只跑 2 配置冒烟测试")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=RESULT_CSV, help="结果 CSV 路径")
    ap.add_argument("--algos", nargs="+", default=ALGOS,
                    help="只跑指定算法（如 --algos CLD）")
    args = ap.parse_args()
    result_csv = args.csv

    tasks = [(name, m, algo, seed)
             for name in PROBLEMS for m in M_VALUES
             for algo in args.algos for seed in SEEDS]
    if args.smoke:
        tasks = [t for t in tasks if t[0] in ("dtlz1", "wfg4") and t[3] == 42]

    # 断点续跑：跳过已完成
    done = set()
    if os.path.exists(result_csv):
        with open(result_csv, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["problem"], int(row["m"]), row["algo"], int(row["seed"])))
        print(f"已完成 {len(done)} 条，续跑剩余")
    tasks = [t for t in tasks if t not in done]
    print(f"总任务 {len(tasks)} 条，workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(os.path.dirname(result_csv), exist_ok=True)
    header = not os.path.exists(result_csv)
    n_done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(result_csv, "a", newline="") as fout:
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
