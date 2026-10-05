"""
缺口画像仪 = 判据链 Stage 1（S1，教训 26 缺口性质判据的工具化），
被 criterion_replay.py stage1 复用（数据缺口性质分类的唯一来源）。

默认口径：DTLZ1-7 + WFG1-9 共 16 问题 × m∈{5,10}（32 实例）× 3 种子，
配置 off / P1(extreme 逐目标极值) / P2(direction 逐参考方向 best-ever 首层解)，
union 探针（纯增量输出，archive_union_cap=0.1 公平上限）。

判据（S1，与 criterion_replay.py 的 S1_* 常量保持一致，论文引用处标注）：
  P2 ΔIGD+% 均值 ≤ -5% 且 3/3 种子同向 → ★ 选择性缺口候选（S2 验证候选）
  P2 ΔIGD+% 均值 ≤ -3% 且 ≥2/3 同向   → ○ 边缘候选（需 10 种子复验）
  其余                                → 生成性/无缺口（记录式修复不适用）
注意：union 探针 ΔIGD 出现正值即配对失效信号（IGD 超集单调不删；
教训 27 混沌/双稳题必须同进程配对，本脚本 off/P1/P2 为独立进程运行，
故只作筛查口径，最终判定以 criterion_replay stage2 同进程配对为准）。

协议要点：FE_MAX=100000；m>10 时 CLDMaOEA 显式 n_partitions=2（120 方向，
与 gen_benchmark.py 协议修订同款；内部默认 p=1 仅 16 方向致 N=16 种群饿死）。

运行（论文复现命令）:
  python experiments/gap_profiler.py                      # 全 32 实例（~6 min，8 workers）
  python experiments/gap_profiler.py --problems dtlz7 --ms 8,15   # 扩展维度弱区归因

产物：experiments/results/gap_profile.csv（列 problem/m/config/seed/igd/igd_plus，
断点续跑），stdout 输出逐实例画像表与分类统计。
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# S1 画像种子（与 S2/评估 SEEDS10=[42,7,2024,314,99,1234,555,8888,271,640]、
# 留出批 [1001..1010] 完全不相交 → 修复审稿意见：S1 不得是 S2 种子子集，
# 保证画像→验证→评估三阶段种子解耦，链为事前决策而非事后解释）
SEEDS = [2001, 2002, 2003]
FE_MAX = 100000
PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]

# S1 判据参数（与 criterion_replay.py S1_DELTA_MAX/S1_SAMEDIR 一致）
S1_DELTA_MAX = -5.0   # ★ 候选阈值：P2 ΔIGD+% 均值 ≤ -5%
S1_SAMEDIR = 3        # ★ 候选同向种子数（3/3）
S1_EDGE_DELTA = -3.0  # ○ 边缘候选阈值
S1_EDGE_SAMEDIR = 2   # ○ 边缘同向种子数

# (标签, corner_archive, policy)
CONFIGS = [
    ("off", False, None),
    ("P1",  True,  "extreme"),
    ("P2",  True,  "direction"),
]


def run_task(task):
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    name, m, seed, tag, arc, policy = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    # m>10 显式 p=2（与 gen_benchmark 协议修订同款：内部默认 p=1 仅 16 方向，N=16 种群饿死）
    alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                   problem_type=ptype, problem_name=name,
                   n_partitions=2 if m > 10 else None,
                   corner_archive=arc, archive_policy=(policy or "extreme"),
                   archive_out_mode="union", archive_union_cap=0.1)
    X, F = alg.run(fe_max=FE_MAX)
    pf = get_pf(name, m)
    return name, m, tag, seed, float(IGD(pf)(F)), float(IGDPlus(pf)(F))


def main():
    import argparse
    import csv as csv_mod
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=os.path.join(ROOT, "experiments", "results",
                                                  "gap_profile.csv"),
                    help="原始数据 CSV 存档（断点续跑；'' 关闭）")
    ap.add_argument("--problems", type=str, default=None,
                    help="逗号分隔问题列表（默认全 16 问题），如 dtlz7")
    ap.add_argument("--ms", type=str, default=None,
                    help="逗号分隔 m 列表（默认 5,10），如 8,15")
    args = ap.parse_args()

    problems = (args.problems.split(",") if args.problems
                else ["dtlz%d" % i for i in range(1, 8)] +
                     ["wfg%d" % i for i in range(1, 10)])
    m_values = [int(x) for x in args.ms.split(",")] if args.ms else [5, 10]

    # 断点续跑：读已完成记录（同时载入内存供分析）
    done = set()
    archived = {}   # (name, m, tag) -> {seed: (igd, igdp)}
    if args.csv and os.path.exists(args.csv):
        with open(args.csv, newline="") as f:
            for r in csv_mod.DictReader(f):
                key = (r["problem"], int(r["m"]), r["config"], int(r["seed"]))
                done.add(key)
                archived.setdefault(key[:3], {})[key[3]] = (
                    float(r["igd"]), float(r["igd_plus"]))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)

    tasks = [(name, m, s, tag, arc, pol)
             for name in problems for m in m_values for s in SEEDS
             for (tag, arc, pol) in CONFIGS
             if (name, m, tag, s) not in done]
    print(f"Route 0 缺口画像仪：{len(problems) * len(m_values)} 实例 × "
          f"{len(SEEDS)} 种子 × {len(CONFIGS)} 配置 = 本次 {len(tasks)} 次，"
          f"workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = dict(archived)   # 存档数据 + 本次新跑
    fout = None
    writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        header = not os.path.exists(args.csv)
        fout = open(args.csv, "a", newline="")
        writer = csv_mod.DictWriter(
            fout, fieldnames=["problem", "m", "config", "seed", "igd", "igd_plus"])
        if header:
            writer.writeheader()
            fout.flush()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            try:
                name, m, tag, seed, igd, igdp = fut.result()
                results.setdefault((name, m, tag), {})[seed] = (igd, igdp)
                if writer:
                    writer.writerow(dict(problem=name, m=m, config=tag,
                                         seed=seed, igd=igd, igd_plus=igdp))
                    fout.flush()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
            done += 1
            if done % 36 == 0:
                print(f"[progress] {done}/{len(tasks)}", flush=True)
    if fout:
        fout.close()

    # 分析：每实例 P1/P2 vs off 配对
    print("\n" + "=" * 92)
    print(f"{'问题':>7} {'m':>3} | {'P1 ΔIGD%':>9} {'P1 ΔIGD+%':>10} {'-x/3':>5} | "
          f"{'P2 ΔIGD%':>9} {'P2 ΔIGD+%':>10} {'-x/3':>5} | 判定")
    print("-" * 92)
    rows = []
    for name in problems:
        for m in m_values:
            off = results.get((name, m, "off"), {})
            if len(off) < len(SEEDS):
                continue
            line = [f"{name:>7} {m:>3}"]
            d_stats = {}
            for tag in ("P1", "P2"):
                res = results.get((name, m, tag), {})
                seeds = [s for s in SEEDS if s in res and s in off]
                if len(seeds) < len(SEEDS):
                    line += ["  --  "] * 3
                    d_stats[tag] = None
                    continue
                o_igdp = np.array([off[s][1] for s in seeds])
                n_igdp = np.array([res[s][1] for s in seeds])
                o_igd = np.array([off[s][0] for s in seeds])
                n_igd = np.array([res[s][0] for s in seeds])
                d_igd = (n_igd - o_igd) / np.abs(o_igd) * 100
                d_igdp = (n_igdp - o_igdp) / np.abs(o_igdp) * 100
                n_neg = int(np.sum(d_igdp < 0))
                line += [f"{d_igd.mean():>+8.1f}%", f"{d_igdp.mean():>+9.1f}%",
                         f"{n_neg}/3"]
                d_stats[tag] = (d_igdp.mean(), n_neg)
            # 判定（S1 判据，常量见文件头）
            verdict = "生成性/无缺口"
            if d_stats.get("P2") is not None:
                mu, neg = d_stats["P2"]
                if mu <= S1_DELTA_MAX and neg >= S1_SAMEDIR:
                    verdict = "★ 选择性缺口候选（S2 验证）"
                elif mu <= S1_EDGE_DELTA and neg >= S1_EDGE_SAMEDIR:
                    verdict = "○ 边缘候选（需 10 种子复验）"
            print(" | ".join(line[:1]) + " | " + " | ".join(line[1:4]) +
                  " | " + " | ".join(line[4:7]) + " | " + verdict)
            rows.append((name, m, verdict))

    n_star = sum(1 for r in rows if r[2].startswith("★"))
    n_mid = sum(1 for r in rows if r[2].startswith("○"))
    print("-" * 92)
    print(f"合计：★ 选择性缺口候选 {n_star} | ○ 边缘候选 {n_mid} | "
          f"生成性/无缺口 {len(rows) - n_star - n_mid}（共 {len(rows)} 实例）")


if __name__ == "__main__":
    main()
