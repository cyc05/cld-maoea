import numpy as np

def tchebycheff_g(F_norm, W, delta=1e-6, rho=0.01):
    """
    计算增广 Tchebycheff 标量化值。
    F_norm: (n, m)
    W: (n_w, m)
    返回: G (n, n_w), G[i,j] = g^te(x_i | w_j)
    """
    ratio = F_norm[:, None, :] / (W[None, :, :] + delta)
    g = np.max(ratio, axis=2) + rho * np.sum(ratio, axis=2)
    return g


def compute_theta_w(W):
    """
    参考方向的特征角（θ-DEA 定义）：
    每个方向与其最近邻方向夹角的均值。作为 APD 惩罚的基准角。
    """
    # W 已单位化，cos 相似度矩阵
    sim = np.clip(W @ W.T, -1.0, 1.0)
    np.fill_diagonal(sim, -np.inf)
    nearest = np.max(sim, axis=1)
    theta_nn = np.arccos(nearest)
    return float(np.mean(theta_nn))


def apd_g(F_norm, W, theta_w, alpha=2.0):
    """
    角度惩罚距离 APD（θ-DEA 标量化）。
    APD = d1 * (1 + (theta/theta_w)^alpha)
    d1 = 沿方向投影长度，theta = 解向量与方向的夹角
    惩罚偏离方向的解，利于逐方向均匀分布。
    F_norm: (n, m); W: (n_w, m) 已单位化
    返回: G (n, n_w)
    """
    f_norm = np.linalg.norm(F_norm, axis=1)                 # (n,)
    f_safe = np.maximum(f_norm, 1e-12)
    d1 = F_norm @ W.T                                       # (n, n_w)
    cos_theta = np.clip(d1 / f_safe[:, None], -1.0, 1.0)
    theta = np.arccos(cos_theta)                            # (n, n_w)
    penalty = (theta / max(theta_w, 1e-12)) ** alpha
    return d1 * (1.0 + penalty)


def _crowding_distance(F):
    """NSGA-II 式拥挤距离（M3-off 消融的截断准则）。边界解记 +inf。"""
    n = len(F)
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for j in range(F.shape[1]):
        order = np.argsort(F[:, j])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span > 1e-12:
            cd[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return cd


def scalarization_g(F_norm, W, mode="tche", theta_w=None, alpha=2.0):
    """标量化函数统一入口"""
    if mode == "apd":
        if theta_w is None:
            theta_w = compute_theta_w(W)
        return apd_g(F_norm, W, theta_w, alpha)
    return tchebycheff_g(F_norm, W)


def _r2_contrib(G):
    """
    向量化计算每个解的 R2 贡献。
    G: (n, n_w) 标量化值矩阵
    返回: contrib (n,), C(i) = R2(A\\{i}) - R2(A)
    原理: 对每个权重列取最小值与次小值；仅当解 i 是某列唯一最小者时，
          移除 i 会使该列取值从最小升至次小。
    """
    n_w = G.shape[1]
    # 每列最小的两个值
    part = np.partition(G, 1, axis=0)
    best, second = part[0], part[1]          # (n_w,)
    best_idx = np.argmin(G, axis=0)          # (n_w,)
    # 将 (second - best) 累加到对应最小值索引上
    contrib = np.zeros(G.shape[0])
    np.add.at(contrib, best_idx, second - best)
    return contrib / n_w


def _update_cache_after_removal(G, alive, removed_idx,
                                 col_min_idx, col_min_val,
                                 col_second_val, col_second_idx):
    """惰性贪心专用：移除一个解后增量更新 per-column min/second-min 缓存。"""
    # 1. removed 是某些列的 min → 提升 second 为新 min，找新 second
    affected_min = np.where(col_min_idx == removed_idx)[0]
    for j in affected_min:
        col_min_idx[j] = col_second_idx[j]
        col_min_val[j] = col_second_val[j]
        mask = alive.copy()
        mask[col_min_idx[j]] = False
        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            vals = G[candidates, j]
            pos = np.argmin(vals)
            col_second_idx[j] = candidates[pos]
            col_second_val[j] = vals[pos]
        else:
            col_second_idx[j] = col_min_idx[j]
            col_second_val[j] = col_min_val[j]

    # 2. removed 是某些列的 second（非 min）→ 找新 second
    affected_second = np.where(col_second_idx == removed_idx)[0]
    affected_second = np.setdiff1d(affected_second, affected_min)
    for j in affected_second:
        mask = alive.copy()
        mask[col_min_idx[j]] = False
        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            vals = G[candidates, j]
            pos = np.argmin(vals)
            col_second_idx[j] = candidates[pos]
            col_second_val[j] = vals[pos]
        else:
            col_second_idx[j] = col_min_idx[j]
            col_second_val[j] = col_min_val[j]


def r2_greedy_truncation(F_norm, selected_idx, critical_idx, k, W,
                         mode="tche", theta_w=None, alpha=2.0,
                         assoc_dirs=None, protect_mode="champion",
                         lag=None, lag_thresh=0.5,
                         lag_quantile=None,
                         d_perp_tiebreak=True,
                         use_lazy_greedy=False,
                         niche_fill=False,
                         empty_streak=None,
                         persist_gens=0):
    """
    R2 贪心截断（算法 3，M2 扩展）。
    F_norm: (n, m) 全部候选解的归一化目标
    selected_idx: 已确定选入的解索引 (list)，受保护不参与剔除
    critical_idx: 临界层解索引 (list)
    k: 需要从临界层再选多少个
    W: R2 权重集 (n_w, m)，可与参考方向不同（支持增密 Λ）
    mode: 'tche' | 'apd' | 'crowding'（crowding = M3-off 消融：临界层纯拥挤
          距离截断，无 R2 贡献与保护）
    assoc_dirs: 小生境保护用的参考方向 (n_d, m)；不为 None 时启用覆盖保底，
                除非存活候选全部是受保护成员（被迫情形）
    protect_mode: 'champion' —— 冠军保护（v1.5）：每个方向 APD 最优者豁免；
                  'last'     —— 末位保护（v1.3）：每方向最后存活者豁免；
                  'adaptive' —— 自适应保护（v1.6，默认）：按方向滞后量 lag 切换，
                               lag > lag_thresh 的滞后方向用 champion（加压追赶），
                               其余用 last（保留 R2 自由选择）
    lag: (n_d,) 方向滞后量，仅 adaptive 模式使用
    lag_thresh: lag 切换阈值，默认 0.5
    返回: 选中的临界层解在原 F_norm 中的索引 (list)
    """
    if k <= 0:
        return []
    if k >= len(critical_idx):
        return list(critical_idx)

    # M3-off 消融：临界层纯拥挤距离截断（无 R2 贡献、无小生境保护）
    if mode == "crowding":
        idx = list(critical_idx)
        while len(idx) > k:
            cd = _crowding_distance(F_norm[idx])
            idx.pop(int(np.argmin(cd)))
        return idx

    A = list(selected_idx) + list(critical_idx)
    FA = F_norm[A]
    G = scalarization_g(FA, W, mode=mode, theta_w=theta_w, alpha=alpha)

    # 小生境关联（垂直距离最近方向）
    if assoc_dirs is not None:
        dot = FA @ assoc_dirs.T
        d_perp_sq = np.sum(FA ** 2, axis=1)[:, None] - dot ** 2
        d_perp_sq = np.maximum(d_perp_sq, 0.0)
        assoc = np.argmin(d_perp_sq, axis=1)          # (n_A,)
        n_d = assoc_dirs.shape[0]
        if d_perp_tiebreak:
            d_perp_own = np.sqrt(d_perp_sq[np.arange(len(A)), assoc])  # P1-a: d⊥ 用于平局打破
        else:
            d_perp_own = None
        if protect_mode == "adaptive":
            # 自适应保护（v2.1）：lag > eff_thresh 的滞后方向用 champion，
            # 其余方向用 last；未提供 lag 时退化为全 champion
            # P11: 动态分位数阈值——根据 lag 分布形状自动选择分位数，
            # 右偏分布（长尾滞后）用低分位数保护更多方向，左偏用高分位数
            if lag is None:
                champion_dirs = np.ones(n_d, dtype=bool)
            else:
                lag_arr = np.asarray(lag)
                if lag_quantile is not None:
                    # 显式指定分位数
                    eff_thresh = np.quantile(lag_arr, lag_quantile)
                else:
                    # 动态分位数：根据 lag 分布偏度自动选择
                    # 右偏（mean > median，长尾滞后方向多）→ 低分位数保护更多
                    # 左偏（mean < median）→ 高分位数减少保护
                    # 对称 → 中位数
                    mean_lag = np.mean(lag_arr)
                    median_lag = np.median(lag_arr)
                    if mean_lag > median_lag * 1.1:
                        # 右偏：25% 分位数，保护更多滞后方向
                        eff_thresh = np.quantile(lag_arr, 0.25)
                    elif mean_lag < median_lag * 0.9:
                        # 左偏：75% 分位数，减少保护
                        eff_thresh = np.quantile(lag_arr, 0.75)
                    else:
                        # 对称：50% 分位数（中位数）
                        eff_thresh = median_lag
                    # 兜底：与 lag_thresh 取较大者，防止动态阈值过低
                    eff_thresh = max(eff_thresh, lag_thresh)
                champion_dirs = lag_arr > eff_thresh
            if np.any(champion_dirs):
                theta_dir = compute_theta_w(assoc_dirs)
                G_dir = apd_g(FA, assoc_dirs, theta_dir, alpha)
                g_own = G_dir[np.arange(len(A)), assoc]
            champion = np.zeros(len(A), dtype=bool)
            for j in range(n_d):
                members = np.where(assoc == j)[0]
                if len(members) > 0 and champion_dirs[j]:
                    champion[members[np.argmin(g_own[members])]] = True
        elif protect_mode == "champion":
            # 冠军保护（v1.5）：每个有成员的方向上，APD 值最优者受保护。
            # 用 APD（角度惩罚距离）而非 tche 选代表：兼顾收敛性与方向对齐，
            # 避免纯 tche 冠军偏离方向、破坏覆盖均匀性（WFG4 退化根因）
            theta_dir = compute_theta_w(assoc_dirs)
            G_dir = apd_g(FA, assoc_dirs, theta_dir, alpha)
            g_own = G_dir[np.arange(len(A)), assoc]
            champion = np.zeros(len(A), dtype=bool)
            for j in range(n_d):
                members = np.where(assoc == j)[0]
                if len(members) > 0:
                    champion[members[np.argmin(g_own[members])]] = True
        else:
            champion = None
    else:
        assoc = None
        champion = None
        d_perp_own = None

    # niche_fill：空方向强制填充（K.16/K.17）。凹 PF 上角点方向无解关联，
    # champion 保护不覆盖空方向，导致 IGD+ 边缘覆盖不足。对每个空方向，
    # 选垂直距离最近的解强制保护（不改 assoc，避免干扰逃逸的滞后方向识别）。
    # 方案 A（PG-EF，v2）：persist_gens>0 时启用持续缺口守卫——方向连续空
    # persist_gens 代（跨代累计于 empty_streak，由调用方持有）才填充，
    # 收敛振荡期的瞬时空洞免疫，消除双稳态题上的随机状态翻转。
    if niche_fill and assoc_dirs is not None and champion is not None:
        has_member = np.zeros(n_d, dtype=bool)
        has_member[assoc] = True
        empty_mask = ~has_member
        if empty_streak is not None and persist_gens > 0:
            empty_streak[empty_mask] += 1
            empty_streak[~empty_mask] = 0
            fill_mask = empty_mask & (empty_streak >= persist_gens)
        else:
            fill_mask = empty_mask
        for j in np.where(fill_mask)[0]:
            champion[int(np.argmin(d_perp_sq[:, j]))] = True

    n_protected = len(selected_idx)
    target_size = n_protected + k
    alive = np.ones(len(A), dtype=bool)

    # 惰性贪心：初始化 per-column min/second-min 缓存，避免每轮 np.partition
    if use_lazy_greedy and len(A) > 1:
        n_w = G.shape[1]
        order = np.argsort(G, axis=0)
        col_min_idx = order[0].copy()
        col_min_val = G[order[0], np.arange(n_w)].copy()
        col_second_val = G[order[1], np.arange(n_w)].copy()
        col_second_idx = order[1].copy()
    else:
        use_lazy_greedy = False  # 只有 1 个解时无意义

    n_remove = len(A) - target_size
    for _ in range(n_remove):
        alive_idx = np.where(alive)[0]
        if use_lazy_greedy:
            # 从缓存计算贡献 O(n_w)，避免 O(n_alive·n_w) 的 partition
            contrib_full = np.zeros(len(A))
            np.add.at(contrib_full, col_min_idx,
                      col_second_val - col_min_val)
            contrib_full /= n_w
            contrib_full[~alive] = np.inf
            contrib = contrib_full[alive_idx]
        else:
            Ga = G[alive_idx]
            contrib = _r2_contrib(Ga)
        # 受保护的前 n_protected 个解不可剔除
        contrib[: n_protected] = np.inf

        if assoc is not None:
            if protect_mode == "adaptive":
                # 自适应：冠军方向按 champion 保护，其余方向按 last 保护
                champ_alive = champion[alive_idx]
                assoc_alive = assoc[alive_idx]
                counts = np.bincount(assoc_alive, minlength=n_d)
                last_alive = counts[assoc_alive] == 1
                last_dirs = ~champion_dirs
                last_alive = last_alive & last_dirs[assoc_alive]
                protected = champ_alive | last_alive
                removable = (~protected) & (contrib < np.inf)
                if np.any(removable):
                    contrib[protected] = np.inf
                else:
                    contrib[protected] = np.inf
            elif champion is not None:
                # 冠军保护：固定豁免冠军，被迫时剔除 R2 贡献最小的冠军
                champ_alive = champion[alive_idx]
                removable = (~champ_alive) & (contrib < np.inf)
                if np.any(removable):
                    contrib[champ_alive] = np.inf
                else:
                    contrib[~champ_alive] = np.inf
            else:
                # 末位保护：每个方向最后存活者豁免（动态），
                # 被迫时（全部为末位成员）剔除 R2 贡献最小者
                assoc_alive = assoc[alive_idx]
                counts = np.bincount(assoc_alive, minlength=n_d)
                last_mask = counts[assoc_alive] == 1
                removable = (~last_mask) & (contrib < np.inf)
                if np.any(removable):
                    contrib[last_mask] = np.inf
                else:
                    contrib[last_mask] = np.inf

        # P1-a: d⊥ 平局打破——贡献差 < 1e-6 时按 d⊥ 排序（NSGA-III 式垂直距离）
        # contrib 与 alive_idx 等长（仅含存活解），removable 为 contrib 局部索引
        removable = np.isfinite(contrib)
        removable_idx = alive_idx[removable]
        if len(removable_idx) > 1 and d_perp_own is not None:
            min_contrib = np.min(contrib[removable])
            tie_mask = (contrib[removable] - min_contrib) < 1e-6
            if np.any(tie_mask):
                tie_local = np.where(removable)[0][tie_mask]
                tie_global = alive_idx[tie_local]
                worst = tie_global[np.argmax(d_perp_own[tie_global])]
            else:
                worst = removable_idx[np.argmin(contrib[removable])]
        else:
            worst = alive_idx[np.argmin(contrib)]
        alive[worst] = False
        if use_lazy_greedy:
            _update_cache_after_removal(
                G, alive, worst, col_min_idx, col_min_val,
                col_second_val, col_second_idx)

    result = np.where(alive)[0]
    # 返回属于临界层的解（原 F_norm 索引）
    return [A[i] for i in result[n_protected:]]
