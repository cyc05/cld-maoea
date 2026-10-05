"""
φ/ρ（支配强度 / 等级-1 解占比）闭环演化轨迹 + 逃逸模式可视化。
代表实例 × 3 种子：dtlz3 m=5（多峰，逃逸模式激活）、dtlz6 m=10（退化 PF）、
wfg3 m=10（退化+不连通）。输出 per-fe 轨迹 CSV + fig_phirho.pdf。

运行: python experiments/phi_rho_trace.py [--workers 8]
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "experiments", "results", "history")
OUT_CSV = os.path.join(OUT_DIR, "phi_rho_trace.csv")
FIG = os.path.join(ROOT, "paper", "figures", "fig_phirho.pdf")
FIG_PNG = os.path.join(ROOT, "paper", "figures", "fig_phirho.png")

CASES = [("dtlz3", 5), ("dtlz6", 10), ("wfg3", 10), ("dtlz7", 10), ("wfg2", 10)]
SEEDS = [42, 7, 2024]
FE_MAX = 100000


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant
    name, m, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                   problem_type=ptype, problem_name=name)
    alg.run(fe_max=FE_MAX)
    fe = np.asarray(alg.history["fe"], dtype=float)
    phi = np.asarray(alg.history["phi"], dtype=float)
    rho = np.asarray(alg.history["rho"], dtype=float)
    gamma = np.asarray(alg.history["gamma"], dtype=float)
    ch = alg.controller.get_history()
    escape = ch[:, 7].astype(float) if ch.size else np.zeros_like(fe)
    return name, m, seed, fe, phi, rho, gamma, escape


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(n, m, s) for (n, m) in CASES for s in SEEDS]
    with open(OUT_CSV, "w", newline="") as f:  # 覆盖写，轨迹量小
        w = csv.writer(f)
        w.writerow(["problem", "m", "seed", "fe", "phi", "rho", "gamma", "escape"])

    rows_by_case = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        with open(OUT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    name, m, seed, fe, phi, rho, gamma, esc = fut.result()
                except Exception as e:
                    print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)
                    continue
                rows_by_case.setdefault((name, m), []).append(
                    (seed, fe, phi, rho, gamma, esc))
                for i in range(len(fe)):
                    w.writerow([name, m, seed, fe[i], phi[i], rho[i], gamma[i], esc[i]])
                print(f"[done] {t}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(20.5, 3.4), sharey=False)
    for ax, (name, m) in zip(axes, CASES):
        if (name, m) not in rows_by_case:
            continue
        for seed, fe, phi, rho, gamma, esc in rows_by_case[(name, m)]:
            ax.plot(fe, phi, lw=1.1, label=f"phi (seed {seed})")
            # 逃逸期填充
            on = esc > 0.5
            if on.any():
                idx = np.where(on)[0]
                for a, b in _runs(idx):
                    ax.axvspan(fe[a], fe[min(b, len(fe) - 1)],
                               color="orange", alpha=0.18, lw=0)
        ax2 = ax.twinx()
        ax2.set_ylabel(r"$\rho$ (rank-1 share)", fontsize=8)
        for seed, fe, phi, rho, gamma, esc in rows_by_case[(name, m)]:
            ax2.plot(fe, rho, lw=0.7, ls="--", color="gray", alpha=0.8)
        ax.set_title(f"{name} $m={m}$", fontsize=10)
        ax.set_xlabel("evaluations", fontsize=8)
        ax.set_ylabel(r"$\phi$", fontsize=8)
        ax.set_ylim(0.12, 0.52)
        ax2.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(FIG_PNG, bbox_inches="tight", dpi=200)
    print("SAVED", FIG)
    print("SAVED", FIG_PNG)
    print("CSV", OUT_CSV)


def _runs(idx):
    """连续区间 -> [(start,end_exclusive)]"""
    runs = []
    s = p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            runs.append((s, p + 1)); s = p = i
    runs.append((s, p + 1))
    return runs


if __name__ == "__main__":
    main()