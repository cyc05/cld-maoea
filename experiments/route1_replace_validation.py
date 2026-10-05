"""
Route 1：7 个 ★ 选择性缺口候选的 replace 模式验证（10 种子，同进程配对）。
协议（教训 27）：off 与 on 在同一进程顺序运行，逐种子断言 array_equal
（轨迹不变性）；replace 恒输出 N（预算公平）。
通过判据：IGD+ 显著改善（p<0.05）且 IGD 不显著退化 + 同向率 ≥8/10
+ 无种子 IGD+ 恶化超 5%。
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000

# Route 0 的 7 个 ★ 候选
INSTANCES = [
    ("wfg3", 5), ("wfg2", 10), ("dtlz6", 10),
    ("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10), ("dtlz7", 5),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    name, m, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    kw = dict(m=m, seed=seed, variant="multi_modal",
              problem_type=ptype, problem_name=name)
    pf = get_pf(name, m)

    # 同进程：先 off 后 on（教训 27），断言轨迹逐位一致
    _, F_off = CLDMaOEA(problem, **kw).run(fe_max=FE_MAX)
    alg = CLDMaOEA(problem, corner_archive=True, archive_policy="direction",
                   archive_out_mode="replace", archive_budget=0.1, **kw)
    X_on, F_on = alg.run(fe_max=FE_MAX)
    traj_ok = bool(np.array_equal(F_off, alg.pop_F_raw))
    out_n = len(F_on)

    igd_off, igdp_off = float(IGD(pf)(F_off)), float(IGDPlus(pf)(F_off))
    igd_on, igdp_on = float(IGD(pf)(F_on)), float(IGDPlus(pf)(F_on))
    return name, m, seed, traj_ok, out_n, \
        igd_off, igd_on, igdp_off, igdp_on


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = [(n, m, s) for (n, m) in INSTANCES for s in SEEDS]
    print(f"Route 1 replace 验证：{len(INSTANCES)} 实例 × {len(SEEDS)} 种子"
          f"（同进程配对），workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                name, m, seed, traj_ok, out_n, io, in_, po, pn = fut.result()
                results.setdefault((name, m), []).append(
                    (seed, traj_ok, out_n, io, in_, po, pn))
                print(f"[done] {name} m={m} seed={seed} traj={'OK' if traj_ok else 'FAIL'} "
                      f"IGD {io:.4f}->{in_:.4f} IGD+ {po:.4f}->{pn:.4f}", flush=True)
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)

    from scipy import stats
    print("\n" + "=" * 96)
    print(f"{'实例':>10} | {'IGD off':>8} {'IGD on':>8} {'ΔIGD%':>7} {'p':>7} | "
          f"{'IGD+ off':>8} {'IGD+ on':>8} {'ΔIGD+%':>7} {'p':>7} | {'同向':>4} {'最差比':>6} {'轨迹':>4} | 判定")
    print("-" * 96)
    n_pass = 0
    for (name, m) in INSTANCES:
        rows = results.get((name, m), [])
        if len(rows) < len(SEEDS):
            print(f"{name:>6} m={m:<3} | 数据不足（{len(rows)}/{len(SEEDS)}）")
            continue
        rows.sort()
        o_igd = np.array([r[3] for r in rows])
        n_igd = np.array([r[4] for r in rows])
        o_igdp = np.array([r[5] for r in rows])
        n_igdp = np.array([r[6] for r in rows])
        d_igd = (n_igd - o_igd) / np.abs(o_igd) * 100
        d_igdp = (n_igdp - o_igdp) / np.abs(o_igdp) * 100
        _, p_igd = stats.ttest_rel(n_igd, o_igd)
        _, p_igdp = stats.ttest_rel(n_igdp, o_igdp)
        same = int(np.sum((d_igd < 0) & (d_igdp < 0)))
        worst = float(np.max(n_igdp / o_igdp))
        traj = all(r[1] for r in rows)
        ok = (d_igdp.mean() <= -5 and p_igdp < 0.05
              and not (d_igd.mean() > 0 and p_igd < 0.05)
              and same >= 8 and worst <= 1.05 and traj)
        n_pass += ok
        print(f"{name:>6} m={m:<3} | {o_igd.mean():>8.4f} {n_igd.mean():>8.4f} "
              f"{d_igd.mean():>+6.1f}% {p_igd:>7.4f} | {o_igdp.mean():>8.4f} "
              f"{n_igdp.mean():>8.4f} {d_igdp.mean():>+6.1f}% {p_igdp:>7.4f} | "
              f"{same:>3}/10 {worst:>5.3f}x {'OK' if traj else 'FAIL'} | "
              f"{'✓ 通过' if ok else '✗ 不通过'}")
    print("-" * 96)
    print(f"通过 {n_pass}/{len(INSTANCES)} 个实例")


if __name__ == "__main__":
    main()
