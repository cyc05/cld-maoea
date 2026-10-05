"""
Phase 2 问题1：18 泛化实例判据链 held-out 验证。

对 gen_benchmark.py 的 18 个零调参泛化实例跑完整 S1→S2 判据链，
验证判据链在调参期从未接触过的实例上能否自动产出 on/off 决策。

S1 画像：off vs P2(direction union) 3 种子 → ΔIGD+% ≤ -5% & 3/3 同向 → 阳性候选
S2 验证：10 种子同进程 off→on(replace) 三条件判定
  ① 双指标 Wilcoxon p<0.05  ② IGD+ 同向率 ≥8/10  ③ 最差种子 IGD ≤1.05×

运行: python experiments/gen_heldout.py [--workers 8]
产物: 打印决策表 → results/gen_heldout_decision.md
"""
import os
import sys
import csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FE_MAX = 100000
SEEDS_S1 = [42, 7, 2024]
SEEDS_S2 = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]
S1_DELTA_MAX = -5.0
S1_SAMEDIR = 3
S2_P = 0.05
S2_SAMEDIR = 0.8
S2_HARD = 1.05

# 18 泛化实例（与 gen_benchmark.py CASES_DEF 同源）
CASES = [
    ("dtlz1i", "dtlz1^-1", "wfg", 5),
    ("dtlz1i", "dtlz1^-1", "wfg", 10),
    ("cdtlz2", "convex_dtlz2", "wfg", 5),
    ("cdtlz2", "convex_dtlz2", "wfg", 10),
    ("cdtlz4", "convex_dtlz4", "wfg", 5),
    ("cdtlz4", "convex_dtlz4", "wfg", 10),
    ("sdtlz1", "sdtlz1", "wfg", 5),
    ("sdtlz1", "sdtlz1", "wfg", 10),
    ("dtlz2", "dtlz2", "dtlz", 8),
    ("dtlz2", "dtlz2", "dtlz", 15),
    ("dtlz4", "dtlz4", "dtlz", 8),
    ("dtlz4", "dtlz4", "dtlz", 15),
    ("dtlz7", "dtlz7", "dtlz", 8),
    ("dtlz7", "dtlz7", "dtlz", 15),
    ("wfg5", "wfg5", "wfg", 8),
    ("wfg5", "wfg5", "wfg", 15),
    ("wfg6", "wfg6", "wfg", 8),
    ("wfg6", "wfg6", "wfg", 15),
]


def _n_var(real_name, m):
    if real_name == "dtlz1^-1" or real_name == "sdtlz1":
        return m + 4
    elif real_name == "dtlz7":
        return m + 19
    elif real_name in ("dtlz2", "dtlz4", "convex_dtlz2", "convex_dtlz4"):
        return m + 9
    else:  # wfg5, wfg6
        return 2 * (m - 1) + 20


def _make_problem(real_name, m):
    from pymoo.problems import get_problem
    return get_problem(real_name, n_var=_n_var(real_name, m), n_obj=m)


def _get_pf(real_name, m, n=2000):
    from pymoo.util.ref_dirs import get_reference_directions
    prob = _make_problem(real_name, m)

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


def _n_partitions(m):
    if m <= 5:
        return 3
    elif m <= 10:
        return 2
    else:
        return 2  # m>10 与 gen_benchmark 协议同款


# ---- S1 画像 ----
def run_s1_task(task):
    sys.path.insert(0, ROOT)
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant

    alias, real, ptype, m, seed, tag = task
    problem = _make_problem(real, m)
    npart = _n_partitions(m)
    if tag == "off":
        alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                       problem_type=ptype, problem_name=alias,
                       n_partitions=npart, corner_archive=False)
    else:  # P2
        alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                       problem_type=ptype, problem_name=alias,
                       n_partitions=npart, corner_archive=True,
                       archive_policy="direction", archive_out_mode="union",
                       archive_union_cap=0.1)
    _, F = alg.run(fe_max=FE_MAX)
    pf = _get_pf(real, m)
    return alias, m, tag, seed, float(IGD(pf)(F)), float(IGDPlus(pf)(F))


# ---- S2 验证 ----
def run_s2_task(task):
    sys.path.insert(0, ROOT)
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    from src.variants import CLDVariant

    alias, real, ptype, m, seed = task
    problem = _make_problem(real, m)
    npart = _n_partitions(m)
    pf = _get_pf(real, m)
    res = {}
    for tag, arc in (("off", False), ("on", True)):
        alg = CLDMaOEA(problem, m=m, seed=seed, variant=CLDVariant.MULTI_MODAL,
                       problem_type=ptype, problem_name=alias,
                       n_partitions=npart, corner_archive=arc,
                       archive_policy="direction",
                       archive_out_mode="replace", archive_budget=0.1)
        _, F = alg.run(fe_max=FE_MAX)
        res[tag] = (float(IGD(pf)(F)), float(IGDPlus(pf)(F)))
    return (alias, m, seed, res["off"][0], res["off"][1],
            res["on"][0], res["on"][1])


def main():
    import argparse
    from scipy.stats import wilcoxon
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # ---- Stage 1：画像 ----
    s1_tasks = [(a, r, p, m, s, t)
                for (a, r, p, m) in CASES
                for s in SEEDS_S1 for t in ("off", "P2")]
    print(f"[S1] {len(CASES)} 实例 × {len(SEEDS_S1)} 种子 × 2 配置 = {len(s1_tasks)} 次",
          flush=True)

    s1_data = {}  # (alias, m, tag) -> {seed: (igd, igdp)}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_s1_task, t): t for t in s1_tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                a, m, tag, seed, igd, igdp = fut.result()
                s1_data.setdefault((a, m, tag), {})[seed] = (igd, igdp)
            except Exception as e:
                print(f"[S1 FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
            if i % 20 == 0:
                print(f"[S1 progress] {i}/{len(s1_tasks)}", flush=True)

    s1_verdicts = {}
    for (alias, real, ptype, m) in CASES:
        off = s1_data.get((alias, m, "off"), {})
        p2 = s1_data.get((alias, m, "P2"), {})
        seeds = [s for s in SEEDS_S1 if s in off and s in p2]
        if len(seeds) < len(SEEDS_S1):
            s1_verdicts[(alias, m)] = ("数据不足", float("nan"), 0)
            continue
        o_igdp = np.array([off[s][1] for s in seeds])
        n_igdp = np.array([p2[s][1] for s in seeds])
        d_igdp = (n_igdp - o_igdp) / np.abs(o_igdp) * 100
        mu = float(d_igdp.mean())
        n_neg = int((d_igdp < 0).sum())
        if mu <= S1_DELTA_MAX and n_neg >= S1_SAMEDIR:
            verdict = "阳性"
        elif mu <= -3.0 and n_neg >= 2:
            verdict = "边缘"
        else:
            verdict = "阴性"
        s1_verdicts[(alias, m)] = (verdict, mu, n_neg)

    candidates = sorted(k for k, v in s1_verdicts.items() if v[0] == "阳性")
    print(f"[S1] 阳性候选 {len(candidates)}：{[f'{a}m{m}' for a, m in candidates]}",
          flush=True)

    # ---- Stage 2：验证 ----
    s2_results = {}
    if candidates:
        s2_tasks = [(a, r, p, m, s)
                    for (a, r, p, m) in CASES if (a, m) in candidates
                    for s in SEEDS_S2]
        print(f"[S2] {len(candidates)} 候选 × {len(SEEDS_S2)} 种子 = {len(s2_tasks)} 配对",
              flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_s2_task, t): t for t in s2_tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    a, m, seed, io, po, inn, pn = fut.result()
                    s2_results.setdefault((a, m), []).append(
                        (seed, io, po, inn, pn))
                except Exception as e:
                    print(f"[S2 FAIL] {futs[fut]}: {type(e).__name__} {e}",
                          flush=True)
                if i % 10 == 0:
                    print(f"[S2 progress] {i}/{len(s2_tasks)}", flush=True)

    # ---- 输出决策表 ----
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines = []
    lines.append("=" * 100)
    lines.append("判据链 held-out 验证：18 泛化实例 on/off 决策表")
    lines.append("=" * 100)
    lines.append(f"| {'实例':>10} {'m':>3} | {'S1 判定':>6} {'S1 ΔIGD+%':>10} "
                 f"| {'S2① 显著':>20} | {'S2② 同向':>10} | {'S2③ 硬检':>12} "
                 f"| {'自动决策':>6} |")
    lines.append("|" + "-" * 98 + "|")
    n_on = 0
    for (alias, real, ptype, m) in CASES:
        v1, mu, neg = s1_verdicts[(alias, m)]
        if v1 != "阳性":
            lines.append(f"| {alias:>10} {m:>3} | {v1:>6} {mu:>+9.1f}% "
                         f"| {'—':>20} | {'—':>10} | {'—':>12} | {'关闭':>6} |")
            continue
        rows = s2_results.get((alias, m), [])
        if len(rows) < len(SEEDS_S2):
            lines.append(f"| {alias:>10} {m:>3} | {v1:>6} {mu:>+9.1f}% "
                         f"| {'数据不足':>20} | {'—':>10} | {'—':>12} | {'—':>6} |")
            continue
        arr = np.array([r[1:] for r in sorted(rows)])
        igd_o, igdp_o, igd_n, igdp_n = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        d_igd = (igd_n - igd_o) / np.abs(igd_o) * 100
        d_igdp = (igdp_n - igdp_o) / np.abs(igdp_o) * 100
        try:
            p1 = wilcoxon(igd_o, igd_n).pvalue if (igd_o != igd_n).any() else 1.0
            p2 = wilcoxon(igdp_o, igdp_n).pvalue if (igdp_o != igdp_n).any() else 1.0
        except Exception:
            p1 = p2 = 1.0
        ok1 = (d_igd.mean() < 0 and d_igdp.mean() < 0 and p1 < S2_P and p2 < S2_P)
        rate = float((d_igdp < 0).mean())
        ok2 = rate >= S2_SAMEDIR
        worst = float((igd_n / igd_o).max())
        ok3 = worst <= S2_HARD
        auto = "启用" if (ok1 and ok2 and ok3) else "关闭"
        if auto == "启用":
            n_on += 1
        c21 = f"{'✓' if ok1 else '✗'} Δ{d_igd.mean():+.1f}% p={p1:.3f}"
        c22 = f"{'✓' if ok2 else '✗'} {int((d_igdp<0).sum())}/{len(d_igdp)}"
        c23 = f"{'✓' if ok3 else '✗'} {worst:.2f}×"
        lines.append(f"| {alias:>10} {m:>3} | {v1:>6} {mu:>+9.1f}% "
                     f"| {c21:>20} | {c22:>10} | {c23:>12} | **{auto}** |")

    lines.append("=" * 100)
    lines.append(f"自动启用 {n_on}/{len(CASES)} 实例；关闭 {len(CASES)-n_on}/{len(CASES)}")
    text = "\n".join(lines)
    print(text)

    out_md = os.path.join(ROOT, "experiments", "results",
                          "gen_heldout_decision.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSAVED {out_md}")


if __name__ == "__main__":
    main()
