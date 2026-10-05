"""
收敛曲线轨迹补跑：3 个代表实例 × 5 算法 × 3 种子（论文收敛行为图数据）。
- 实例: dtlz6 m=10（退化 PF，CLD 制胜区）/ dtlz7 m=10（不连通 PF，corner_archive 启用）/
        wfg3 m=10（退化+不连通，CLD 领先 NSGA3 5.9×）
- 轨迹: IGD/IGD+ @ FE 检查点网格（2500..97500，种群口径），
        终点 FE=100k 记录返回输出（CLD 为档案合并后的输出，基线为最终种群）
- 协议: 与 m3_benchmark.py 同款（ref_dirs/N/终止条件/种子）
- 输出: experiments/results/history/convergence_history.csv（支持断点续跑）

运行: python experiments/convergence_history.py [--workers 8]
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "experiments", "results", "history")
OUT_CSV = os.path.join(OUT_DIR, "convergence_history.csv")
FIELDS = ["problem", "m", "algo", "seed", "fe", "igd", "igd_plus", "time"]

CASES = [("dtlz6", 10), ("dtlz7", 10), ("wfg3", 10)]
SEEDS = [42, 7, 2024]
ALGOS = ["CLD", "NSGA3", "MOEAD", "RVEA", "CTAEA"]
FE_MAX = 100000
FE_STEP = 2500


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
    """与 m3_benchmark.get_pf 完全一致（含 dtlz5/6/7 采样种子 0）"""
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
    """单次运行（子进程入口）：返回该 run 的全部检查点行"""
    sys.path.insert(0, ROOT)
    name, m, algo, seed = task
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus

    problem = make_problem(name, m)
    pf = get_pf(name, m)
    igd_f, igdp_f = IGD(pf), IGDPlus(pf)
    rows = []

    def snap(fe_grid, F, t=""):
        rows.append(dict(problem=name, m=m, algo=algo, seed=seed, fe=fe_grid,
                         igd=float(igd_f(F)), igd_plus=float(igdp_f(F)), time=t))

    t0 = time.perf_counter()
    if algo == "CLD":
        from src.algorithm import CLDMaOEA
        from src.variants import CLDVariant
        problem_type = "dtlz" if name.startswith("dtlz") else "wfg"
        # K.26 基线配置（configs/multi_modal.json 自动命中，含 corner_archive overrides）
        alg = CLDMaOEA(problem, m=m, seed=seed, problem_type=problem_type,
                       problem_name=name, variant=CLDVariant.MULTI_MODAL)
        nxt = FE_STEP

        def cb(X, F, a):
            nonlocal nxt
            fe = a.history["fe"][-1]
            while nxt <= FE_MAX - FE_STEP and fe >= nxt:
                snap(nxt, F)
                nxt += FE_STEP

        X, F = alg.run(fe_max=FE_MAX, callback=cb)
    else:
        from pymoo.optimize import minimize
        from pymoo.core.callback import Callback
        from pymoo.util.ref_dirs import get_reference_directions
        ref_dirs = get_reference_directions("das-dennis", m, n_partitions=n_partitions(m))
        N = int(np.ceil(len(ref_dirs) / 4) * 4)
        if algo == "NSGA3":
            from pymoo.algorithms.moo.nsga3 import NSGA3
            method = NSGA3(ref_dirs=ref_dirs, pop_size=N)
        elif algo == "MOEAD":
            from pymoo.algorithms.moo.moead import MOEAD
            method = MOEAD(ref_dirs=ref_dirs, n_neighbors=15)
        elif algo == "RVEA":
            from pymoo.algorithms.moo.rvea import RVEA
            method = RVEA(ref_dirs=ref_dirs, pop_size=N)
        else:
            from pymoo.algorithms.moo.ctaea import CTAEA
            method = CTAEA(ref_dirs=ref_dirs)

        class Tracker(Callback):
            def __init__(self):
                super().__init__()
                self.nxt = FE_STEP

            def notify(self, algorithm):
                n = algorithm.evaluator.n_eval
                if n < self.nxt:
                    return
                try:
                    F = algorithm.pop.get("F")
                except Exception:
                    try:
                        F = np.array([s.F for s in algorithm.pop])
                    except Exception:
                        return
                while self.nxt <= FE_MAX - FE_STEP and n >= self.nxt:
                    snap(self.nxt, F)
                    self.nxt += FE_STEP

        res = minimize(problem, method, termination=("n_evals", FE_MAX),
                       seed=seed, verbose=False, callback=Tracker())
        F = res.F

    snap(FE_MAX, F, t=round(time.perf_counter() - t0, 2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tasks = [(name, m, algo, seed) for name, m in CASES
             for algo in ALGOS for seed in SEEDS]

    # 断点续跑：以终点行（fe=FE_MAX）判定 run 完成
    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["fe"]) == FE_MAX:
                    done.add((row["problem"], int(row["m"]), row["algo"], int(row["seed"])))
        print(f"已完成 {len(done)} 个 run，续跑剩余")
    tasks = [t for t in tasks if t not in done]
    print(f"总任务 {len(tasks)}，workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(OUT_DIR, exist_ok=True)
    header = not os.path.exists(OUT_CSV)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(OUT_CSV, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        if header:
            writer.writeheader()
            fout.flush()
        futs = {pool.submit(run_task, t): t for t in tasks}
        n = 0
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)
                continue
            for r in rows:
                writer.writerow(r)
            fout.flush()
            n += 1
            print(f"[{n}/{len(tasks)}] {t} 完成，{rows[-1]['time']}s", flush=True)
    print(f"完成，写入 {OUT_CSV}，耗时 {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
