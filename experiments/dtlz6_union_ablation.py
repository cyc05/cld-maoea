"""
dtlz6 m=10 corner_archive 输出模式消融（union vs replace）。

背景（教训 28）：Route 1 中 dtlz6 m=10 被 S2② 拒绝（同向率 6/10），
但 S1 画像增益大（P2 union 探针 ΔIGD+ -44.9%）且 S2① 双指标显著（p=0.031）。
教训 28 假设：replace 风险在删除侧（缺口边际时删除噪声暴露），
union 增益在补入侧。本实验验证：union 模式（纯增量、与 S1 P2 探针同款）
是否通过判据链 S2①②③——区分"缺口真实但 replace 删除噪声致拒"
与"缺口本身边际、拒绝正确"。

配置（与既有协议同款）：
  off     : corner_archive=False
  union   : policy="direction" + out_mode="union" + union_cap=0.1（= S1 P2 探针）
  replace : policy="direction" + out_mode="replace" + budget=0.1（= K.26 落地）

协议（教训 27）：三种配置同进程顺序运行（同一种子），off 轨迹一致性断言。
注意 union 恒输出 N+|A|（预算不公平，画像口径），replace 恒输出 N（预算公平）。

运行:
  python experiments/dtlz6_union_ablation.py --workers 8
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
FE_MAX = 100000
INSTANCE = ("dtlz6", 10)


def run_task(seed):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA

    name, m = INSTANCE
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    kw = dict(m=m, seed=seed, variant="multi_modal",
              problem_type=ptype, problem_name=name)
    pf = get_pf(name, m)

    # 同进程顺序：off → union → replace（教训 27）
    _, F_off = CLDMaOEA(problem, **kw).run(fe_max=FE_MAX)

    alg_u = CLDMaOEA(problem, corner_archive=True, archive_policy="direction",
                     archive_out_mode="union", archive_union_cap=0.1, **kw)
    _, F_u = alg_u.run(fe_max=FE_MAX)
    traj_u = bool(np.array_equal(F_off, alg_u.pop_F_raw))

    alg_r = CLDMaOEA(problem, corner_archive=True, archive_policy="direction",
                     archive_out_mode="replace", archive_budget=0.1, **kw)
    _, F_r = alg_r.run(fe_max=FE_MAX)
    traj_r = bool(np.array_equal(F_off, alg_r.pop_F_raw))

    out = {}
    for tag, F in (("off", F_off), ("union", F_u), ("replace", F_r)):
        out[tag] = (len(F), float(IGD(pf)(F)), float(IGDPlus(pf)(F)))
    return seed, traj_u, traj_r, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print(f"dtlz6 m=10 输出模式消融：{len(SEEDS)} 种子 × "
          f"{{off, union, replace}} 同进程配对，workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, s): s for s in SEEDS}
        for fut in as_completed(futs):
            try:
                seed, tu, tr, out = fut.result()
                rows.append((seed, tu, tr, out))
                print(f"[done] seed={seed} traj_u={'OK' if tu else 'FAIL'} "
                      f"traj_r={'OK' if tr else 'FAIL'} | "
                      f"off N={out['off'][0]} IGD+={out['off'][2]:.4f} | "
                      f"union N={out['union'][0]} IGD+={out['union'][2]:.4f} | "
                      f"replace N={out['replace'][0]} IGD+={out['replace'][2]:.4f}",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] seed={futs[fut]}: {type(e).__name__} {e}", flush=True)

    rows.sort()
    if len(rows) < len(SEEDS):
        print(f"\n数据不足（{len(rows)}/{len(SEEDS)}），不输出判定")
        return

    # 存档（判读复用，避免重跑）
    import csv as _csv
    out_csv = os.path.join(ROOT, "experiments", "results", "dtlz6_union_ablation.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["seed", "traj_union", "traj_replace",
                    "igd_off", "igd_union", "igd_replace",
                    "igdp_off", "igdp_union", "igdp_replace",
                    "n_off", "n_union", "n_replace"])
        for seed, tu, tr, out in rows:
            w.writerow([seed, tu, tr,
                        out["off"][1], out["union"][1], out["replace"][1],
                        out["off"][2], out["union"][2], out["replace"][2],
                        out["off"][0], out["union"][0], out["replace"][0]])
    print(f"\n明细已存 {out_csv}")

    from scipy import stats
    o_igd = np.array([r[3]["off"][1] for r in rows])
    o_igdp = np.array([r[3]["off"][2] for r in rows])

    print("\n" + "=" * 100)
    print("dtlz6 m=10：S2 判据对比（union vs replace）")
    print("=" * 100)
    print(f"{'模式':>8} | {'N':>4} | {'IGD off':>8} {'IGD on':>8} {'ΔIGD%':>7} {'p':>7} | "
          f"{'IGD+ off':>8} {'IGD+ on':>8} {'ΔIGD+%':>7} {'p':>7} | {'同向':>4} {'最差比':>6} | S2 判定")
    print("-" * 100)

    verdicts = {}
    for tag in ("union", "replace"):
        n_igd = np.array([r[3][tag][1] for r in rows])
        n_igdp = np.array([r[3][tag][2] for r in rows])
        out_n = int(np.mean([r[3][tag][0] for r in rows]))
        d_igd = (n_igd - o_igd) / np.abs(o_igd) * 100
        d_igdp = (n_igdp - o_igdp) / np.abs(o_igdp) * 100
        _, p_igd = stats.ttest_rel(n_igd, o_igd)
        _, p_igdp = stats.ttest_rel(n_igdp, o_igdp)
        same = int(np.sum((d_igd < 0) & (d_igdp < 0)))
        worst = float(np.max(n_igdp / o_igdp))
        # S2①②③（与 criterion_replay / route1 同款）
        c1 = (d_igdp.mean() <= -5 and p_igdp < 0.05
              and not (d_igd.mean() > 0 and p_igd < 0.05))
        c2 = same >= 8
        c3 = worst <= 1.05
        ok = c1 and c2 and c3
        verdicts[tag] = ok
        print(f"{tag:>8} | {out_n:>4} | {o_igd.mean():>8.4f} {n_igd.mean():>8.4f} "
              f"{d_igd.mean():>+6.1f}% {p_igd:>7.4f} | {o_igdp.mean():>8.4f} "
              f"{n_igdp.mean():>8.4f} {d_igdp.mean():>+6.1f}% {p_igdp:>7.4f} | "
              f"{same:>3}/10 {worst:>5.3f}x | "
              f"{'✓ 通过' if ok else '✗ 拒绝'}"
              f"（①{'√' if c1 else '×'}②{'√' if c2 else '×'}③{'√' if c3 else '×'}）")
    traj_ok = all(r[1] and r[2] for r in rows)
    print("-" * 100)
    print(f"轨迹不变性（off vs on 内部种群）：{'10/10 OK' if traj_ok else 'FAIL'}")
    if verdicts.get("union") and not verdicts.get("replace"):
        print("结论：union 通过而 replace 拒绝 → 缺口真实，replace 删除侧噪声致拒"
              "（教训 28 成立；dtlz6 m=10 属'union 增益在补入侧'的边际缺口）")
    elif not verdicts.get("union"):
        print("结论：union 也拒绝 → 缺口本身边际/画像增益跨种子不稳，S2② 拒绝正确")
    else:
        print("结论：双模式通过 → S2② 6/10 为 replace 轮次噪声，可重新评估")


if __name__ == "__main__":
    main()
