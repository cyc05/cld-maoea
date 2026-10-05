import numpy as np

def das_dennis_weights(m, p):
    """
    生成 Das-Dennis 参考方向（单纯形格点），单位化。
    m: 目标数
    p: 分割数
    返回: (n_weights, m)
    """
    if p == 0:
        w = np.ones((1, m)) / m
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    from itertools import combinations
    def gen_points(m, p):
        # 生成所有非负整数解 sum = p
        if m == 1:
            yield (p,)
        else:
            for i in range(p + 1):
                for rest in gen_points(m - 1, p - i):
                    yield (i,) + rest
    w = np.array(list(gen_points(m, p)), dtype=float) / p
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def normalize_objectives(F, z_star, z_nad):
    """目标归一化到 [0,1] 近似"""
    span = z_nad - z_star
    span[span < 1e-12] = 1.0
    return (F - z_star) / span


def cdas_transform(F_norm, phi):
    """
    CDAS 目标变换 (Controlling Dominance Area of Solutions).
    F_norm: (n, m) 归一化目标
    phi: (n,) 每个解的强度参数, 或标量
    返回变换后的目标矩阵 (n, m)
    """
    n, m = F_norm.shape
    r = np.linalg.norm(F_norm, axis=1)  # (n,)
    r_safe = np.maximum(r, 1e-12)
    # 各分量对应的 omega 角
    omega = np.arccos(np.clip(F_norm / r_safe[:, None], -1.0, 1.0))
    phi_arr = np.broadcast_to(phi, (n,))[:, None]  # (n,1)
    # 变换公式
    numerator = np.sin(omega + phi_arr * np.pi)
    denominator = np.sin(phi_arr * np.pi)
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return r_safe[:, None] * numerator / denominator


def estimate_nadir_extreme(F_norm, z_star):
    """
    NSGA-III 式 nadir 估计：对第一非支配层，逐轴最小化 ASF 找极点，
    过 m 个极点拟合超平面，取与坐标轴截距作为 z_nad - z_star。
    失败（奇异/截距非法）时回退到逐维最大值。
    F_norm: (n, m) 已归一化到以 z_star 为原点的目标值（第一层即可）
    返回: span (m,), 即 z_nad - z_star
    """
    n, m = F_norm.shape
    eps = 1e-6
    # ASF: max_i (f_i / w_i)，w 为轴单位向量（0 分量替换为 eps）
    extremes = []
    for axis in range(m):
        w = np.full(m, eps)
        w[axis] = 1.0
        asf = np.max(F_norm / w[None, :], axis=1)
        extremes.append(F_norm[np.argmin(asf)])
    E = np.array(extremes)  # (m, m)
    fallback = np.max(F_norm, axis=0)
    fallback = np.where(fallback < eps, 1.0, fallback)
    try:
        # 解 E @ alpha = 1，截距 = 1/alpha
        alpha = np.linalg.solve(E, np.ones(m))
        if np.any(np.abs(alpha) < eps) or np.any(~np.isfinite(alpha)):
            return fallback
        intercepts = 1.0 / alpha
        if np.any(intercepts < -eps) or np.any(~np.isfinite(intercepts)):
            return fallback
        intercepts = np.maximum(intercepts, 0.0)
        # 截距不能小于当前最大值（否则压缩），也不能为 0
        span = np.maximum(intercepts, fallback * 0.5)
        span = np.where(span < eps, fallback, span)
        return span
    except np.linalg.LinAlgError:
        return fallback



def dominance_matrix(F_norm, phi_per_x):
    """
    方向差异化强化支配矩阵（设计文档 §2.4 精确实现）。
    两两比较时双方统一使用 phi_xy = min(phi_x, phi_y) 做 CDAS 变换，
    保证单次比较内尺度一致。
    F_norm: (n, m) 归一化目标
    phi_per_x: (n,) 每个解的强度参数
    返回: dom (n, n) bool, dom[i,j]=True 表示 i 支配 j
    内存: O(n^2 * m)，n 为 2N；N~240, m~15 时约 30MB 量级
    """
    n, m = F_norm.shape
    r = np.linalg.norm(F_norm, axis=1)                      # (n,)
    r_safe = np.maximum(r, 1e-12)
    omega = np.arccos(np.clip(F_norm / r_safe[:, None], -1.0, 1.0))  # (n, m)

    PHI = np.minimum(phi_per_x[:, None], phi_per_x[None, :])  # (n, n)
    denom = np.sin(PHI * np.pi)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)     # (n, n)

    angle = PHI[:, :, None] * np.pi                            # (n, n, 1)
    Ti = r_safe[:, None, None] * np.sin(omega[:, None, :] + angle) / denom[:, :, None]
    Tj = r_safe[None, :, None] * np.sin(omega[None, :, :] + angle) / denom[:, :, None]

    le = np.all(Ti <= Tj, axis=2)
    lt = np.any(Ti < Tj, axis=2)
    return le & lt


def strengthened_nd_sort(F_norm, phi_per_x):
    """
    强化支配非支配排序。
    返回: fronts (list of lists), rank (n,)
    复杂度 O(m * n^2)
    """
    n = F_norm.shape[0]
    dom = dominance_matrix(F_norm, phi_per_x)
    n_dom = dom.sum(axis=0)          # 支配 j 的解数量
    rank = np.full(n, -1, dtype=int)
    fronts = []
    current = np.where(n_dom == 0)[0]
    k = 0
    remaining = n
    while len(current) > 0:
        fronts.append(current.tolist())
        rank[current] = k
        remaining -= len(current)
        if remaining == 0:
            break
        # 从支配计数中移除当前层
        n_dom = n_dom - dom[current].sum(axis=0)
        n_dom[current] = -1          # 防止重复入层
        current = np.where(n_dom == 0)[0]
        k += 1
    return fronts, rank


def direction_differentiation(F_norm, W, phi_base, delta_phi, phi_min):
    """
    方向差异化强度计算（算法 2）。
    F_norm: (n, m)
    W: (n_w, m)
    phi_base: 标量
    返回: phi_per_x (n,), assoc (n,), g_hat (n_w,), lag (n_w,)
    """
    n, m = F_norm.shape
    # 关联到最近参考方向（垂直距离）
    # d_perp^2 = |f|^2 - (f·w)^2 / |w|^2, W 已单位化
    dot = F_norm @ W.T  # (n, n_w)
    f_norm_sq = np.sum(F_norm ** 2, axis=1)  # (n,)
    d_perp_sq = f_norm_sq[:, None] - dot ** 2
    assoc = np.argmin(d_perp_sq, axis=1)  # (n,)

    # 各方向当前最优 Tchebycheff 值
    n_w = W.shape[0]
    g_hat = np.full(n_w, np.inf)
    # 为避免每个方向都遍历全部解，先算所有解的 g^te
    delta = 1e-6
    rho_tch = 0.01
    # 使用所有方向计算会太慢，只计算关联方向的 g
    # 简化：计算每个解在其关联方向上的 g^te
    w_assoc = W[assoc]  # (n, m)
    g_vals = np.max(F_norm / (w_assoc + delta), axis=1) + rho_tch * np.sum(
        F_norm / (w_assoc + delta), axis=1
    )  # (n,)
    for j in range(n_w):
        mask = assoc == j
        if np.any(mask):
            g_hat[j] = np.min(g_vals[mask])
    # 无关联解的方向取最大值（表示滞后）
    no_assoc = np.isinf(g_hat)
    if np.any(~no_assoc):
        g_hat[no_assoc] = np.max(g_hat[~no_assoc])
    else:
        g_hat[no_assoc] = 1.0

    # 相对滞后量
    g_min, g_max = np.min(g_hat), np.max(g_hat)
    if g_max - g_min < 1e-12:
        lag = np.zeros(n_w)
    else:
        lag = (g_hat - g_min) / (g_max - g_min)

    # 个体化 phi
    phi_per_x = phi_base - delta_phi * lag[assoc]
    phi_per_x = np.clip(phi_per_x, phi_min, 0.5)
    return phi_per_x, assoc, g_hat, lag
