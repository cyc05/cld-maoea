"""
审稿意见回应：与通用算法配置器（algorithm configuration）的对比。

审稿指控：未与 SMAC / irace / ParamILS 等标准配置器对比，若准则链不如这些
通用工具，则专用设计价值存疑。

回应设计（环境约束：Python 3.14 下 SMAC/ConfigSpace 无轮子、irace 需 R，
故在**同一 runner、同一种子协议**下自实现两类经典配置器，规避教训 27 的
跨进程漂移并保证可复现）：
  * RandomSearch（全局探索基线）：配置空间均匀随机采样；
  * ParamILS 风格迭代局部搜索（first-improvement，预算受限）：
    **初始点 = 准则链的按题决策配置**——最尖锐的对比：通用配置器
    从准则链的解出发，在更大搜索自由度下能否找到更优配置。

公平性：配置器与准则链给**相同运行预算**（每实例每方法 30 次 CLD 运行，
≈ 准则链 S1 9 + S2 20 = 29 次）；调参目标 = 调参种子 [11,22] 上的 mean IGD+
（与评估种子 SEEDS10、S1 种子 [2001-2003]、留出批 [1001-1010] 全不相交，
配置器同样不得"见过"评估种子）。

配置空间 = 准则链所决策的变量 + PD 控制器参数（兼顾审稿对 (ki,kd) 的敏感性关切）：
  corner_archive ∈ {0,1}  （核心决策：是否启用 passive archive）
  archive_out_mode ∈ {replace, union}      （条件：archive=1）
  archive_policy  ∈ {extreme, direction}   （条件：archive=1）
  archive_budget  ∈ {0.05,0.1,0.15,0.2}    （条件：archive=1）
  rho_star ∈ {0.05,0.10,0.15,0.20,0.30}    （PD 目标）
  ki ∈ {0.1,0.3,0.5,0.7,1.0}               （PD 比例）
  kd ∈ {0.0,0.1,0.2,0.4}                   （PD 微分）

运行：python experiments/auto_config.py [--workers 8] [--instances dtlz5:5,...]
产物：experiments/results/tuner_configs.csv
      （problem, m, method, cfg_json, tune_igdp, n_evals；断点续跑）
后续：tuner 最优配置经 eval_tuned.py 在 SEEDS10 上评估，与准则链对比。
"""
import os
import sys
import csv
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT_CSV = os.path.join(ROOT, "experiments", "results", "tuner_configs.csv")

PROBLEMS = (["dtlz%d" % i for i in range(1, 8)] +
            ["wfg%d" % i for i in range(1, 10)])
M_VALUES = [5, 10]
TUNE_SEEDS = [11, 22]          # 调参种子（独立于 S1/S2/评估/留出）
FE_MAX = 100000
BUDGET_CFG = 15                # 每方法评估的配置数（×2 种子 = 30 次运行 ≈ 准则链 29 次）

# 配置空间
SPACE = {
    "corner_archive": [0, 1],
    "archive_out_mode": ["replace", "union"],
    "archive_policy": ["extreme", "direction"],
    "archive_budget": [0.05, 0.1, 0.15, 0.2],
    "rho_star": [0.05, 0.10, 0.15, 0.20, 0.30],
    "ki": [0.1, 0.3, 0.5, 0.7, 1.0],
    "kd": [0.0, 0.1, 0.2, 0.4],
}
PARAMS = list(SPACE.keys())

# 出厂默认（准则链"关闭 archive"时的配置，作为 RandomSearch 参照点）
FACTORY = dict(corner_archive=0, archive_out_mode="replace",
               archive_policy="direction", archive_budget=0.1,
               rho_star=0.15, ki=0.5, kd=0.1)

# K.26 准则链启用集（与 criterion_replay.K26_ENABLED 一致）：ParamILS 初始点
K26_ENABLED = {("wfg2", 10), ("dtlz5", 5), ("dtlz5", 10), ("dtlz7", 10)}


def chain_config(name, m):
    """准则链的按题决策配置（ParamILS 初始点 / 对比基准）"""
    cfg = dict(FACTORY)
    if (name, m) in K26_ENABLED:
        cfg.update(corner_archive=1, archive_policy="direction",
                   archive_out_mode="replace", archive_budget=0.1)
    return cfg


def _eval_cfg(name, m, cfg, seeds):
    """配置 → 调参种子上的 mean IGD+（子进程内单次调用，串行各种子）"""
    from experiments.m3_benchmark import make_problem, get_pf
    from pymoo.indicators.igd_plus import IGDPlus
    from src.algorithm import CLDMaOEA
    problem = make_problem(name, m)
    ptype = "dtlz" if name.startswith("dtlz") else "wfg"
    pf = get_pf(name, m)
    vals = []
    for s in seeds:
        alg = CLDMaOEA(problem, m=m, seed=s, variant="multi_modal",
                       problem_type=ptype, problem_name=name,
                       n_partitions=2 if m > 10 else None,
                       corner_archive=bool(cfg["corner_archive"]),
                       archive_out_mode=cfg["archive_out_mode"],
                       archive_policy=cfg["archive_policy"],
                       archive_budget=cfg["archive_budget"],
                       archive_union_cap=0.1,
                       rho_star=cfg["rho_star"], ki=cfg["ki"], kd=cfg["kd"])
        X, F = alg.run(fe_max=FE_MAX)
        vals.append(float(IGDPlus(pf)(F)))
    return float(np.mean(vals))


def _sample(rng):
    return {p: SPACE[p][int(rng.integers(len(SPACE[p])))] for p in PARAMS}


def _perturb(rng, cfg, r=2):
    """ParamILS 随机扰动：随机换 r 个参数的取值（one/two-exchange）"""
    new = dict(cfg)
    for p in rng.choice(PARAMS, size=min(r, len(PARAMS)), replace=False):
        choices = [v for v in SPACE[p] if v != new[p]]
        if choices:
            new[p] = choices[int(rng.integers(len(choices)))]
    return new


def random_search(name, m, seed):
    rng = np.random.default_rng(seed)
    best_cfg, best_v, n = None, np.inf, 0
    # 纳入出厂默认与准则链配置作为参照（ RandSearch 至少不差于它们）
    for cfg in [FACTORY, chain_config(name, m)]:
        v = _eval_cfg(name, m, cfg, TUNE_SEEDS); n += 1
        if v < best_v:
            best_cfg, best_v = dict(cfg), v
    for _ in range(BUDGET_CFG - 2):
        cfg = _sample(rng)
        v = _eval_cfg(name, m, cfg, TUNE_SEEDS); n += 1
        if v < best_v:
            best_cfg, best_v = cfg, v
    return best_cfg, best_v, n


def paramils(name, m, seed):
    """ParamILS 风格 ILS：从准则链配置出发，first-improvement + 随机重启"""
    rng = np.random.default_rng(seed)
    cur = chain_config(name, m)
    cur_v = _eval_cfg(name, m, cur, TUNE_SEEDS); n = 1
    best_cfg, best_v = dict(cur), cur_v
    for _ in range(BUDGET_CFG - 1):
        if rng.random() < 0.1:           # 重启概率（ParamILS 标准 0.01，预算紧放大到 0.1）
            cand = _sample(rng)
        else:
            cand = _perturb(rng, cur, r=2)
        v = _eval_cfg(name, m, cand, TUNE_SEEDS); n += 1
        if v < cur_v:                    # first-improvement 移动
            cur, cur_v = cand, v
        if v < best_v:
            best_cfg, best_v = dict(cand), v
    return best_cfg, best_v, n


def tune_instance(task):
    """单实例任务：跑 rs + paramils 两个 tuner（实例间并行，实例内序贯）"""
    sys.path.insert(0, ROOT)
    name, m = task
    out = []
    for method, fn in (("randsearch", random_search), ("paramils", paramils)):
        cfg, v, n = fn(name, m, seed=hash((name, m, method)) % (2 ** 31))
        out.append(dict(problem=name, m=m, method=method,
                        cfg_json=json.dumps(cfg), tune_igdp=v, n_evals=n))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--instances", type=str, default=None,
                    help="逗号分隔 name:m（默认全 32 实例）")
    args = ap.parse_args()

    if args.instances:
        instances = []
        for it in args.instances.split(","):
            nm, mm = it.strip().split(":")
            instances.append((nm, int(mm)))
    else:
        instances = [(p, m) for p in PROBLEMS for m in M_VALUES]

    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["problem"], int(r["m"]), r["method"]))
        print(f"存档已有 {len(done)} 条，续跑剩余", flush=True)

    tasks = [t for t in instances
             if not all((t[0], t[1], meth) in done
                        for meth in ("randsearch", "paramils"))]
    print(f"自动配置器对比：{len(tasks)} 实例 × 2 方法（randsearch/paramils），"
          f"每方法 {BUDGET_CFG} 配置 × {len(TUNE_SEEDS)} 调参种子，"
          f"workers={args.workers}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    fields = ["problem", "m", "method", "cfg_json", "tune_igdp", "n_evals"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    header = not os.path.exists(OUT_CSV)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(OUT_CSV, "a", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        if header:
            w.writeheader(); fout.flush()
        futs = {pool.submit(tune_instance, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                for row in fut.result():
                    w.writerow(row)
                fout.flush()
            except Exception as e:
                print(f"[FAIL] {futs[fut]}: {type(e).__name__} {e}", flush=True)
            print(f"[progress] {i}/{len(tasks)} 实例", flush=True)
    print(f"完成：{OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
