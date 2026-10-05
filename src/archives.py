import numpy as np

from src.r2select import apd_g, compute_theta_w


class CornerArchive:
    """
    角点档案（方案 C 架构，K.23）：与截断完全解耦的边缘覆盖观察器。

    内容策略（Route 0 缺口画像仪，experiments/CornerArchive_复用路线.md）：
    - policy="extreme"（P1）：逐目标极值解 best-ever（min f_i），K.23 原版；
    - policy="direction"（P2）：逐参考方向 best-ever 首层解（min Σf），
      捕获"产生过但被截断丢弃"的方向代表——选择性缺口的直接证据。

    三个构造性保证（K.23 验证）：observe 纯读不写回（轨迹不变性）；
    不占种群名额；finalize 输出期一次性合并。跨代比较均用与归一化无关
    的原始目标值（K.23 教训：归一化上下文逐代演化，跨代 APD 不可比）。
    """

    def __init__(self, m, policy="extreme", W=None):
        self.m = m
        self.policy = policy
        self.W = W
        if policy == "direction":
            if W is None:
                raise ValueError("policy='direction' 需要参考方向 W")
            self.n_slots = len(W)
        else:
            self.n_slots = m
        self.best = np.full(self.n_slots, np.inf)  # P1: min f_i; P2: min Σf
        self.arch_F = None                        # (n_slots, m)
        self.arch_X = None                        # (n_slots, n_var)

    def _ensure(self, n_var):
        if self.arch_F is None:
            self.arch_F = np.full((self.n_slots, self.m), np.nan)
            self.arch_X = np.full((self.n_slots, n_var), np.nan)

    def observe(self, RX, RF, RF_norm=None, front1=None):
        """每代旁路观察（纯读）。P2 需 RF_norm 与首层池索引。"""
        if self.policy == "extreme":
            min_idx = np.argmin(RF, axis=0)
            min_val = RF[min_idx, np.arange(self.m)]
            idxs = np.where(min_val < self.best)[0]
            if len(idxs) == 0:
                return
            self._ensure(RX.shape[1])
            self.best[idxs] = min_val[idxs]
            self.arch_F[idxs] = RF[min_idx[idxs]]
            self.arch_X[idxs] = RX[min_idx[idxs]]
        else:  # direction (P2)：逐方向记录首层代表（原始 Σf best-ever）
            if RF_norm is None or front1 is None or len(front1) == 0:
                return
            Fn = RF_norm[front1]
            dot = Fn @ self.W.T
            d_perp_sq = np.maximum(
                np.sum(Fn ** 2, axis=1)[:, None] - dot ** 2, 0.0)
            assoc = np.argmin(d_perp_sq, axis=1)
            sumf = RF[front1].sum(axis=1)
            for j in np.unique(assoc):
                members = np.where(assoc == j)[0]
                b = members[np.argmin(sumf[members])]
                if sumf[b] < self.best[j]:
                    self._ensure(RX.shape[1])
                    self.best[j] = sumf[b]
                    self.arch_F[j] = RF[front1[b]]
                    self.arch_X[j] = RX[front1[b]]

    @staticmethod
    def _axis_row_map(W, m):
        """P1：目标轴 i -> W 中轴单位方向行号（丢失方向判定用）"""
        rows = np.zeros(m, dtype=int)
        for i in range(m):
            hits = np.where(np.isclose(W[:, i], 1.0))[0]
            rows[i] = hits[0] if len(hits) else int(np.argmax(W[:, i]))
        return rows

    def finalize(self, X_pop, F_pop, F_norm_pop, W, theta_w=None, alpha=2.0,
                 mode="replace", budget=0.1, union_cap=None):
        """
        输出期一次性合并（演化已结束，不影响动力学）。
        mode="replace"：删冗余补档案，恒输出 N（预算公平）；
        mode="union"  ：纯增量并集（画像探针）；union_cap（种群比例，如 0.1）
        限制补入数量保证 P1/P2 公平，补入优先"末代种群已丢失的方向"。
        """
        if self.arch_F is None:
            return X_pop, F_pop
        valid = np.isfinite(self.best)
        A_F = self.arch_F[valid].copy()
        A_X = self.arch_X[valid].copy()
        if len(A_F) == 0:
            return X_pop, F_pop

        # 过滤：被种群支配的档案解 + 与种群重复的解
        le = (F_pop[:, None, :] <= A_F[None, :, :]).all(axis=2)
        lt = (F_pop[:, None, :] < A_F[None, :, :]).any(axis=2)
        dominated = (le & lt).any(axis=0)
        dup = np.array([np.any(np.all(F_pop == A_F[i], axis=1))
                        for i in range(len(A_F))])
        keep = ~(dominated | dup)
        A_F, A_X = A_F[keep], A_X[keep]
        if len(A_F) == 0:
            return X_pop, F_pop

        if mode == "union":
            if union_cap is not None:
                cap = int(np.ceil(union_cap * len(X_pop)))
                if len(A_F) > cap:
                    slot_idx = np.where(valid)[0][keep]
                    best_kept = self.best[slot_idx]
                    dirs = (self._axis_row_map(W, self.m)[slot_idx]
                            if self.policy == "extreme" else slot_idx)
                    dot = F_norm_pop @ W.T
                    dpp = np.maximum(
                        np.sum(F_norm_pop ** 2, 1)[:, None] - dot ** 2, 0.0)
                    covered = set(np.argmin(dpp, axis=1).tolist())
                    lost = np.array([0 if d in covered else 1 for d in dirs])
                    # 丢失方向优先，再按 best 升序（更收敛的先补）
                    order = np.lexsort((best_kept, -lost))
                    A_F, A_X = A_F[order[:cap]], A_X[order[:cap]]
            return np.vstack([X_pop, A_X]), np.vstack([F_pop, A_F])

        # replace：删冗余补档案，恒输出 N
        k = min(len(A_F), int(np.ceil(budget * len(X_pop))))
        drop = self._redundant_drop(F_norm_pop, W, theta_w, alpha, k)
        if len(drop) == 0:
            return X_pop, F_pop
        k_eff = min(len(drop), len(A_F))
        drop = drop[:k_eff]
        keep_mask = np.ones(len(X_pop), dtype=bool)
        keep_mask[drop] = False
        return (np.vstack([X_pop[keep_mask], A_X[:k_eff]]),
                np.vstack([F_pop[keep_mask], A_F[:k_eff]]))

    @staticmethod
    def _redundant_drop(F_norm_pop, W, theta_w, alpha, k):
        """
        冗余判定：解的方向上已有更优成员（非 champion）且自身方向 APD
        最差 → 优先删除。champion（每方向最优）永不删，保证不打开新缺口。
        """
        if F_norm_pop is None or len(F_norm_pop) < 2:
            return np.array([], dtype=int)
        dot = F_norm_pop @ W.T
        d_perp_sq = np.maximum(
            np.sum(F_norm_pop ** 2, axis=1)[:, None] - dot ** 2, 0.0)
        assoc = np.argmin(d_perp_sq, axis=1)
        G_dir = apd_g(F_norm_pop, W,
                      theta_w if theta_w is not None else compute_theta_w(W),
                      alpha)
        g_own = G_dir[np.arange(len(F_norm_pop)), assoc]
        is_champ = np.zeros(len(F_norm_pop), dtype=bool)
        for j in range(W.shape[0]):
            members = np.where(assoc == j)[0]
            if len(members) > 0:
                is_champ[members[np.argmin(g_own[members])]] = True
        cand = np.where(~is_champ)[0]
        if len(cand) == 0:
            return cand
        return cand[np.argsort(-g_own[cand])][:k]
