"""
审稿意见回应：在线自适应算子/参数选择（adaptive operator selection）对比基线。

审稿指控：未与基于在线学习 / bandit 的 adaptive operator selection 方法
（如 FRRMAB）对比——这些方法同样解决"选择哪个机制"，且是在线决策而非离线配置。

本模块实现标准 FRRMAB（Fitness-Rate-Rank Multi-Armed Bandit，Fialho et al. 2010）
的滑窗信用赋值 + UCB 选择，用作 CLD-MaOEA 的**在线支配强度选择器**：
在运行中于若干离散 phi 档位（臂）间在线选择，与主算法的 PD 闭环控制器
（连续调 phi）在**同一被控量**上直接对比——从而检验"在线 bandit 选择"是否
优于/不亚于本文的"PD 闭环控制 + 离线准则链"范式。

设计要点（保证与 PD 控制器的对比公平）：
  * 臂 = 离散支配强度档位 phi ∈ {0.15,0.25,0.35,0.45,0.5}，覆盖控制器
    phi_min..0.5 的可控区间；方向分化、逃逸等其余机制保持不变，唯一差异
    是 phi_base 的来源（bandit 在线选 vs PD 闭环连续调）；
  * reward = 主循环现成的综合改进量 improvement（理想点坐标改进 +
    小生境标量化值横向收敛改进），与收敛门控同源，零额外开销；
  * credit assignment = 滑动窗口内臂奖励的 rank-decay 加权和（FRR 标准）；
  * 选择 = UCB1（credit + 探索红利），保证未探索臂被优先试探。
"""
import numpy as np


class FRRMAB:
    """滑窗 FRR 信用赋值 + UCB 选择的多臂老虎机（连续奖励）。"""

    def __init__(self, arms, window=50, decay=0.7, c=1.0, seed=None):
        """
        arms: 臂取值列表（如 phi 档位 [0.15,0.25,0.35,0.45,0.5]）
        window: 滑动窗口大小（保留最近 window 次 (arm, reward)）
        decay: FRR 信用衰减（窗口内该臂奖励降序第 j 名权重 decay^j）
        c: UCB 探索红利系数
        """
        self.arms = list(arms)
        self.K = len(self.arms)
        self.window = int(window)
        self.decay = float(decay)
        self.c = float(c)
        self.rng = np.random.default_rng(seed)
        self.buffer = []           # 滑动窗口 [(arm_idx, reward)]
        self.counts = np.zeros(self.K, dtype=np.int64)   # 各臂累计被拉次数
        self.t = 0                 # 总拉动次数

    def _frr_credit(self):
        """各臂的 FRR 信用：窗口内该臂奖励降序、按名次 decay 加权并归一。"""
        credit = np.zeros(self.K)
        # 预计算 decay 幂（窗口内最多 window 个名次）
        ranks_w = self.decay ** np.arange(max(1, len(self.buffer)))
        for i in range(self.K):
            rs = [r for (a, r) in self.buffer if a == i]
            if not rs:
                continue
            rs.sort(reverse=True)
            w = ranks_w[: len(rs)]
            credit[i] = float(np.dot(w, rs) / (w.sum() + 1e-12))
        return credit

    def select(self):
        """UCB 选择臂索引；未拉过的臂优先（初始化阶段遍历各臂）。"""
        for i in range(self.K):
            if self.counts[i] == 0:
                return i
        credit = self._frr_credit()
        bonus = self.c * np.sqrt(
            np.log(max(self.t, 1) + 1.0) / self.counts.astype(float))
        return int(np.argmax(credit + bonus))

    def store(self, arm_idx, reward):
        """记录一次拉动及其奖励，维护滑动窗口。"""
        self.buffer.append((int(arm_idx), float(reward)))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        self.counts[int(arm_idx)] += 1
        self.t += 1

    def phi_of(self, arm_idx):
        return float(self.arms[int(arm_idx)])
