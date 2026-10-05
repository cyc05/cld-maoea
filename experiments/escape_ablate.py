"""
逃逸模式消融：在 multi_modal.json 启用逃逸的 8 个实例上对比
  escape ON（部署配置） vs escape OFF（div_eps=0 永不触发逃逸 + eta_m=20）
的 IGD。3 种子。证明逃逸模式对多峰题的净增益。

运行: python experiments/escape_ablate.py [--workers 8]
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024]
FE_MAX = 100000

# multi_modal.json overrides 中启用 escape_eta_m=5 的 (instance, m)
INSTANCES = [
    ("dtlz3", 5), ("dtlz3", 10), ("dtlz1", 10),
    ("wfg2", 5), ("wfg2", 10), ("wfg3", 5),
    ("wfg7", 10), ("wfg9", 10),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from src.algorithm import CLDMaOEA
    name, m, seed, mode = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    kw = dict(m=m, seed=seed, variant="multi_modal",
              problem_type=ptype, problem_name=name)
    alg = CLDMaOEA(problem, **kw)
    if mode == "off":
        # 关闭逃逸模式：多样性停滞时回退到 v1.4 弛豫（gamma<1），并禁用
        # 逃逸动作（无大步长变异、无 Δφ 倍增、无 phi_min 下压、无重启）。
        alg.controller.escape_on_div_stall = False
        alg.escape_eta_m = None
        alg.escape_delta_phi_scale = 1.0
        alg.escape_phi_min = None
        alg.escape_restart_frac = 0
    _, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(name, m)
    return name, m, seed, mode, float(IGD(pf)(F))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(n, m, s, mode)
             for (n, m) in INSTANCES for s in SEEDS for mode in ("on", "off")]
    print(f"逃逸消融：{len(INSTANCES)} 实例 × {len(SEEDS)} 种子 × on/off"
          f" = {len(tasks)} runs，workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                name, m, seed, mode, igd = fut.result()
                results.setdefault((name, m), {}).setdefault(mode, []).append(igd)
                print(f"[done] {name} m={m} seed={seed} mode={mode} IGD={igd:.4f}", flush=True)
            except Exception as e:
                print(f"[FAIL] {t}: {type(e).__name__} {e}", flush=True)

    print("\n" + "=" * 66)
    print(f"{'实例':>12} | {'escape ON':>10} | {'escape OFF':>11} | {'ΔIGD%':>7}")
    print("-" * 66)
    for (name, m) in INSTANCES:
        on = results.get((name, m), {}).get("on", [])
        off = results.get((name, m), {}).get("off", [])
        if len(on) < len(SEEDS) or len(off) < len(SEEDS):
            print(f"{name:>6} m={m:<3} | 数据不足"); continue
        mu_on, mu_off = np.mean(on), np.mean(off)
        d = (mu_on - mu_off) / abs(mu_off) * 100
        print(f"{name:>6} m={m:<3} | {mu_on:>10.4f} | {mu_off:>11.4f} | {d:>+6.1f}%")
    print("-" * 66)
    print("负值 = escape 改善（IGD 更低）；正值 = escape 有害")


if __name__ == "__main__":
    main()