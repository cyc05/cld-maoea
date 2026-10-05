"""
审稿意见回应（意见5核心）：准则链 vs 通用配置器的深度分析。

回答三个问题：
  (Q1) 性能：配置器在留出评估种子上是否显著优于准则链？（wins/ties/losses + A12）
  (Q2) 决策一致性：配置器是否在与准则链**相同的实例**上独立"重新发现"
       archive 启用？——若是，则强证准则链的结构性决策正确，且准则链
       以更低成本、可审计方式达到。
  (Q3) 优势来源：配置器胜出的实例，其改进来自 archive 结构决策还是
       控制器连续参数（rho_star/ki/kd）微调？——区分"结构 vs 参数"。

运行：python experiments/analyze_tuner.py
产物：stdout 分析 + results/tuner_compare.csv
"""
import os
import sys
import csv
import json
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from experiments.auto_config import chain_config, K26_ENABLED

TUNER_CFG = os.path.join(ROOT, "experiments", "results", "tuner_configs.csv")
TUNER_EVAL = os.path.join(ROOT, "experiments", "results", "tuner_eval.csv")
M3_CSV = os.path.join(ROOT, "experiments", "results", "m3_results.csv")
OUT_CSV = os.path.join(ROOT, "experiments", "results", "tuner_compare.csv")
METHODS = ["randsearch", "paramils"]


def a12(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.sum(a[:, None] < b[None, :]); eq = np.sum(a[:, None] == b[None, :])
    return (lt + 0.5 * eq) / (a.size * b.size)


def load():
    cfgs = {}
    with open(TUNER_CFG, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cfgs[(r["problem"], int(r["m"]), r["method"])] = json.loads(r["cfg_json"])
    teval = defaultdict(dict)
    with open(TUNER_EVAL, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            teval[(r["problem"], int(r["m"]), r["method"])][int(r["seed"])] = \
                float(r["igd_plus"])
    cld = defaultdict(dict)
    with open(M3_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["algo"] == "CLD":
                cld[(r["problem"], int(r["m"]))][int(r["seed"])] = float(r["igd_plus"])
    return cfgs, teval, cld


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from scipy.stats import wilcoxon
    cfgs, teval, cld = load()
    instances = sorted({(p, m) for (p, m, _) in cfgs}, key=lambda k: (k[1], k[0]))

    # ---------- Q2: archive 决策一致性 ----------
    print("=" * 78)
    print("Q2 配置器是否重新发现准则链的 archive 启用决策？")
    print("（准则链启用集 K26 =", sorted(f"{n}m{m}" for n, m in K26_ENABLED), "）")
    print("=" * 78)
    for method in METHODS:
        tuner_on = set()
        for (p, m) in instances:
            if cfgs[(p, m, method)]["corner_archive"] == 1:
                tuner_on.add((p, m))
        agree_enable = len(tuner_on & K26_ENABLED)
        agree_disable = len([k for k in instances
                             if k not in tuner_on and k not in K26_ENABLED])
        print(f"\n  {method}: tuner 启用 archive 的实例 = "
              f"{sorted(f'{n}m{m}' for n, m in tuner_on)}")
        print(f"    与准则链一致：启用重合 {agree_enable}/{len(K26_ENABLED)}，"
              f"禁用重合 {agree_disable}/{len(instances)-len(K26_ENABLED)}，"
              f"总一致 {(agree_enable+agree_disable)}/{len(instances)}")

    # ---------- Q1 + Q3: 逐实例性能对比与胜出来源 ----------
    rows = []
    for method in METHODS:
        print("\n" + "=" * 78)
        print(f"Q1/Q3  {method} vs 准则链（留出评估种子，IGD+）")
        print("=" * 78)
        print(f"{'实例':<13}{'链arc':<6}{'tuner arc':<10}{'Δ%':<8}{'p':<8}"
              f"{'A12':<7}{'判定':<10}{'胜出时改动的参数'}")
        print("-" * 78)
        win = tie = lose = 0
        a12s = []
        for (name, m) in instances:
            key = (name, m, method)
            if key not in teval or (name, m) not in cld:
                continue
            common = sorted(set(teval[key]) & set(cld[(name, m)]))
            if len(common) < 5:
                continue
            t = np.array([teval[key][s] for s in common])
            c = np.array([cld[(name, m)][s] for s in common])
            try:
                p = wilcoxon(t, c).pvalue if (t != c).any() else 1.0
            except Exception:
                p = 1.0
            d = (t.mean() - c.mean()) / (abs(c.mean()) + 1e-12) * 100
            a = a12(t, c)
            a12s.append(a)
            tc = cfgs[key]
            cc = chain_config(name, m)
            changed = [k for k in tc if tc[k] != cc[k]]
            if t.mean() < c.mean() and p < 0.05:
                verdict, win = "tuner优", win + 1
            elif t.mean() > c.mean() and p < 0.05:
                verdict, lose = "链优", lose + 1
            else:
                verdict, tie = "平", tie + 1
            chg = ",".join(changed) if verdict == "tuner优" else ""
            print(f"{name} m={m:<4}{int(cc['corner_archive']):<6}"
                  f"{int(tc['corner_archive']):<10}{d:<+7.1f}%{p:<8.3f}"
                  f"{a:<7.3f}{verdict:<10}{chg}")
            rows.append(dict(problem=name, m=m, method=method,
                             chain_archive=int(cc["corner_archive"]),
                             tuner_archive=int(tc["corner_archive"]),
                             delta_pct=round(d, 3), p=round(p, 4),
                             a12=round(a, 3), verdict=verdict,
                             changed_params=";".join(changed)))
        print("-" * 78)
        print(f"汇总：tuner 胜 {win} / 平 {tie} / 负 {lose} | "
              f"A12 mean={np.mean(a12s):.3f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n产物：{OUT_CSV}")


if __name__ == "__main__":
    main()
