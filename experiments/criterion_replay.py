"""
论文缺口 #2：按题启用判据链的自动复现验证（"决策可复现性"实验）。

目的：证明 K.26 的 4 例 corner_archive 启用决策 100% 由算法化判据流程产出，
而非人工逐题调参。论文表述依据：本脚本从零数据出发复现出与 K.26 一致的启用集。

判据链（各阶段引用项目教训编号）：
  Stage 1 画像判据（教训 26 缺口性质判据，数据 = gap_profiler union 探针）：
      P2 ΔIGD+ 均值 ≤ -5% 且 3/3 种子同向 → 选择性缺口候选；否则关闭（生成性/无缺口）
  Stage 2 验证判据（教训 20 三条件 + K.25 Route 1 协议，数据 = 10 种子同进程 off/on 配对）：
      ① IGD 与 IGD+ 双指标均值下降且 Wilcoxon p<0.05（无真实增益即拒）
      ② IGD+ 逐种子同向率 ≥ 8/10（同向率不足即拒，如 dtlz6 m=10 的 6/10）
      ③ 好种子硬检查：无种子 IGD 恶化 >5%（>1.05×，如 wfg3 m=5 单种子 +41%）
  Stage 3 回归判据：全量回归 0 负（由 final_verify 承担，本脚本引用结论）

S2 配置（K.26 落地款 corner_archive）：policy="direction" + out_mode="replace"
+ budget=0.1；FE_MAX=100000；m>10 时 n_partitions=2（与 gen_benchmark 协议同款）。

用法（论文复现命令）：
  python experiments/criterion_replay.py stage1     # 检查/生成画像数据（调 gap_profiler）
  python experiments/criterion_replay.py stage2     # 阳性候选 10 种子同进程配对采集
  python experiments/criterion_replay.py decide     # 判据链决策 + vs K.26 一致性
  python experiments/criterion_replay.py all        # 一键全链（S1→S2→decide）
  # 扩展验证（泛化性弱区归因，跳过 S1 直接指定候选，不做 K.26 一致性对照）：
  python experiments/criterion_replay.py stage2 --candidates dtlz7:8,dtlz7:15

产物（experiments/results/）：
  gap_profile.csv        S1 画像原始数据（gap_profiler 生成，断点续跑）
  route1_pairs.csv       S2 十种子同进程配对（主口径，decide 一致性统计输入）
  route1_pairs_ext.csv   --candidates 指定候选的配对（扩展口径，与主口径隔离）
  decide 表输出到 stdout（可重定向存档，如 `> criterion_replay_decision.md`）
"""
import os
import sys
import csv
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GAP_CSV = os.path.join(ROOT, "experiments", "results", "gap_profile.csv")
PAIR_CSV = os.path.join(ROOT, "experiments", "results", "route1_pairs.csv")

# S1 画像种子（独立于 S2/评估，与 gap_profiler.py SEEDS 保持一致；
# 修复审稿意见：S1 不得是 S2 种子子集，保证画像→验证种子解耦）
SEEDS3 = [2001, 2002, 2003]
SEEDS10 = [42, 7, 2024, 314, 99, 1234, 555, 8888, 271, 640]  # 与 m3_benchmark 一致
FE_MAX = 100000

# 判据参数（与项目教训一致，论文引用处标注）
S1_DELTA_MAX = -5.0     # Stage1：P2 ΔIGD+% 均值阈值（≤ -5%）
S1_SAMEDIR = 3          # Stage1：同向种子数要求（3/3）
S2_P = 0.05             # Stage2①：Wilcoxon 显著性
S2_SAMEDIR = 0.8        # Stage2②：IGD+ 逐种子同向率下限（≥8/10）
S2_HARD = 1.05          # Stage2③：单种子 IGD 恶化硬上限（好种子不翻坏）

# K.26 实际启用集（corner_archive m 级 override，来源 configs/multi_modal.json）
K26_ENABLED = {("wfg2", 10), ("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10)}


# ---------------------------------------------------------------- Stage 1
def load_gap():
    """读画像 CSV → {(name, m, config): {seed: (igd, igdp)}}"""
    data = {}
    with open(GAP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["problem"], int(r["m"]), r["config"], int(r["seed"]))
            data.setdefault(key[:3], {})[key[3]] = (float(r["igd"]),
                                                    float(r["igd_plus"]))
    return data


def stage1_verdicts(data):
    """Stage 1 判据：返回 {(name,m): (verdict, mu_digdp, n_neg)}"""
    out = {}
    for (name, m, tag), seeds in data.items():
        if tag != "P2":
            continue
        off = data.get((name, m, "off"), {})
        common = [s for s in SEEDS3 if s in seeds and s in off]
        if len(common) < len(SEEDS3):
            out[(name, m)] = ("数据不足", float("nan"), 0)
            continue
        d_igdp = np.array([(seeds[s][1] - off[s][1]) / abs(off[s][1]) * 100
                           for s in common])
        mu = float(d_igdp.mean())
        n_neg = int((d_igdp < 0).sum())
        if mu <= S1_DELTA_MAX and n_neg >= S1_SAMEDIR:
            verdict = "阳性"
        elif mu <= -3.0 and n_neg >= 2:
            verdict = "边缘"
        else:
            verdict = "阴性"
        out[(name, m)] = (verdict, mu, n_neg)
    return out


def run_stage1(args):
    if not os.path.exists(GAP_CSV):
        print(f"[stage1] 画像数据不存在，调用 gap_profiler 生成（~6 min）...")
        os.system(f"{sys.executable} {os.path.join(ROOT, 'experiments', 'gap_profiler.py')} "
                  f"--workers {args.workers}")
    data = load_gap()
    n_rows = sum(len(v) for v in data.values())
    n = sum(1 for k in data if k[2] == "P2")
    print(f"[stage1] 画像数据：{n_rows} 条记录，覆盖 {n} 实例（P2 配置）")
    verdicts = stage1_verdicts(data)
    pos = sorted(k for k, v in verdicts.items() if v[0] == "阳性")
    mid = sorted(k for k, v in verdicts.items() if v[0] == "边缘")
    print(f"[stage1] 阳性候选 {len(pos)}：{[f'{n}m{m}' for n, m in pos]}")
    print(f"[stage1] 边缘 {len(mid)}：{[f'{n}m{m}' for n, m in mid]}（按 K.24 处理：不进入 Stage 2）")
    return pos


# ---------------------------------------------------------------- Stage 2
def run_pair_task(task):
    """单实例单种子：同进程内先 off 后 on（教训 27：off/on 配对必须同进程）"""
    sys.path.insert(0, ROOT)
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA

    name, m, seed = task
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    pf = get_pf(name, m)

    res = {}
    for tag, arc in (("off", False), ("on", True)):
        # m>10 显式 p=2（与 gen_benchmark/gap_profiler 协议同款，防 N=16 种群饿死）
        alg = CLDMaOEA(problem, m=m, seed=seed, variant="multi_modal",
                       problem_type=ptype, problem_name=name,
                       n_partitions=2 if m > 10 else None,
                       corner_archive=arc, archive_policy="direction",
                       archive_out_mode="replace", archive_budget=0.1)
        X, F = alg.run(fe_max=FE_MAX)
        res[tag] = (float(IGD(pf)(F)), float(IGDPlus(pf)(F)))
    return (name, m, seed,
            res["off"][0], res["off"][1], res["on"][0], res["on"][1])


def run_stage2(args):
    pair_csv = PAIR_CSV
    if getattr(args, "candidates", None):
        # 指定候选模式（泛化性验证等扩展用途）：跳过 S1 筛选，存档独立文件
        # 不污染 K.26 一致性统计（decide 固定读 PAIR_CSV）
        candidates = []
        for item in args.candidates.split(","):
            name, m = item.strip().split(":")
            candidates.append((name, int(m)))
        pair_csv = PAIR_CSV.replace(".csv", "_ext.csv")
        print(f"[stage2] 指定候选：{candidates}（扩展验证，存档 {pair_csv}）")
    else:
        candidates = run_stage1(args)
    if not candidates:
        print("[stage2] 无阳性候选，跳过")
        return

    done = set()
    if os.path.exists(pair_csv):
        with open(pair_csv, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), int(r["seed"])))
        print(f"[stage2] 存档已有 {len(done)} 条配对，续跑剩余")

    tasks = [(n, m, s) for (n, m) in candidates for s in SEEDS10
             if (n, m, s) not in done]
    print(f"[stage2] {len(candidates)} 候选 × {len(SEEDS10)} 种子 = 本次 {len(tasks)} 配对"
          f"（每配对同进程 off→on 两次运行），workers={args.workers}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fields = ["problem", "m", "seed", "igd_off", "igdp_off", "igd_on", "igdp_on"]
    os.makedirs(os.path.dirname(pair_csv), exist_ok=True)
    header = not os.path.exists(pair_csv)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(pair_csv, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        if header:
            writer.writeheader()
            fout.flush()
        futs = {pool.submit(run_pair_task, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                writer.writerow(dict(zip(fields, fut.result())))
                fout.flush()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
            if i % 10 == 0 or i == len(futs):
                print(f"[stage2 progress] {i}/{len(futs)}", flush=True)
    print(f"[stage2] 完成：{pair_csv}")

    # 扩展候选：就地输出 S2 三条件判定（decide 不覆盖 ext 文件）
    if pair_csv != PAIR_CSV:
        _decide_ext(pair_csv, candidates)


def _decide_ext(pair_csv, candidates):
    """指定候选模式的 S2 判定表（与 decide 同判据，不做 K.26 一致性对照）"""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import numpy as np
    from scipy.stats import wilcoxon

    pairs = {}
    with open(pair_csv, newline="") as f:
        for r in csv.DictReader(f):
            pairs.setdefault((r["problem"], int(r["m"])), []).append(
                (float(r["igd_off"]), float(r["igdp_off"]),
                 float(r["igd_on"]), float(r["igdp_on"])))

    print("\n" + "=" * 88)
    print("S2 三条件判定（扩展候选）")
    print("=" * 88)
    print(f"| 实例 | S2① 双指标显著 | S2② 同向率 | S2③ 硬检查 | 自动决策 |")
    print("|---|---|---|---|---|")
    for (name, m) in candidates:
        rows = pairs.get((name, m), [])
        if len(rows) < len(SEEDS10):
            print(f"| {name} m={m} | 数据不足（{len(rows)}/{len(SEEDS10)}） | | | |")
            continue
        arr = np.array(rows)
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
        print(f"| {name} m={m} | {'✓' if ok1 else '✗'}（ΔIGD {d_igd.mean():+.1f}% "
              f"p={p1:.3f} / ΔIGD+ {d_igdp.mean():+.1f}% p={p2:.3f}） | "
              f"{'✓' if ok2 else '✗'}（{int((d_igdp < 0).sum())}/{len(d_igdp)}） | "
              f"{'✓' if ok3 else '✗'}（最差 {worst:.2f}×） | **{auto}** |")


# ---------------------------------------------------------------- decide
def run_decide(_args):
    from scipy.stats import wilcoxon

    data = load_gap()
    verdicts = stage1_verdicts(data)

    pairs = {}
    if os.path.exists(PAIR_CSV):
        with open(PAIR_CSV, newline="") as f:
            for r in csv.DictReader(f):
                pairs.setdefault((r["problem"], int(r["m"])), []).append(
                    (float(r["igd_off"]), float(r["igdp_off"]),
                     float(r["igd_on"]), float(r["igdp_on"])))

    print("=" * 96)
    print("判据链自动决策 vs K.26 实际启用（corner_archive）")
    print("=" * 96)
    print(f"| 实例 | S1 判定 | S1 ΔIGD+% | S2① 双指标显著 | S2② 同向率 | "
          f"S2③ 硬检查 | 自动决策 | K.26 实际 | 一致 |")
    print("|---|---|---|---|---|---|---|---|---|")

    n_match = n_total = 0
    for (name, m), (v1, mu, n_neg) in sorted(verdicts.items(),
                                             key=lambda kv: (kv[0][1], kv[0][0])):
        actual = "启用" if (name, m) in K26_ENABLED else "关闭"
        if v1 != "阳性":
            auto, c21, c22, c23 = "关闭", "—", "—", "—"
        elif (name, m) not in pairs:
            auto, c21, c22, c23 = "缺S2数据", "?", "?", "?"
        else:
            arr = np.array(pairs[(name, m)])
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
            c21 = f"{'✓' if ok1 else '✗'}（ΔIGD {d_igd.mean():+.1f}% p={p1:.3f} / ΔIGD+ {d_igdp.mean():+.1f}% p={p2:.3f}）"
            c22 = f"{'✓' if ok2 else '✗'}（{int((d_igdp < 0).sum())}/{len(d_igdp)}）"
            c23 = f"{'✓' if ok3 else '✗'}（最差 {worst:.2f}×）"
        match = "✓" if auto == actual else "✗"
        if auto != "缺S2数据":
            n_total += 1
            n_match += (auto == actual)
        print(f"| {name} m={m} | {v1} | {mu:+.1f}% | {c21} | {c22} | {c23} | "
              f"**{auto}** | {actual} | {match} |")

    print(f"\n一致性：{n_match}/{n_total}"
          f"（{n_match / max(n_total, 1) * 100:.0f}%）")
    if n_match == n_total:
        print("结论：判据链从数据出发自动复现 K.26 全部启用决策——"
              "按题启用是判据驱动的，非人工逐题调参。")


def main():
    # Windows GBK 控制台防崩（decide 表含 ✓/✗ 字符）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("stage", choices=["stage1", "stage2", "decide", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--candidates", type=str, default=None,
                    help="指定候选（如 dtlz7:8,dtlz7:15），跳过 S1 直接 S2，"
                         "存档 route1_pairs_ext.csv（不用于 K.26 一致性统计）")
    args = ap.parse_args()

    if args.stage in ("stage2", "all"):
        run_stage2(args)
    elif args.stage == "stage1":
        run_stage1(args)
    if args.stage in ("decide", "all"):
        run_decide(args)


if __name__ == "__main__":
    main()
