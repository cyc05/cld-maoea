import numpy as np
from src.dominance import (
    das_dennis_weights,
    normalize_objectives,
    strengthened_nd_sort,
    dominance_matrix,
    direction_differentiation,
    estimate_nadir_extreme,
)
from src.controller import DominanceController
from src.r2select import r2_greedy_truncation, compute_theta_w
from src.variants import CLDVariant, get_variant_config
from src.archives import CornerArchive


class CLDMaOEA:
    """
    CLD-MaOEA: 闭环支配强化超多目标进化算法（M2 版本）。
    """

    def __init__(
        self,
        problem,
        m,
        n_partitions=None,
        rho_star=0.15,
        ki=0.5,
        kd=0.1,
        delta_phi=0.05,
        phi_min=0.15,
        eta_c=20,
        eta_m=20,
        seed=None,
        use_differentiation=True,
        r2_mode="tche",
        r2_dense_p=None,
        r2_alpha=2.0,
        niche_protect=True,
        niche_mating=True,
        protect_mode=None,
        lag_thresh=None,
        problem_type=None,
        problem_name=None,
        d_perp_tiebreak=None,
        lag_quantile=None,
        fixed_phi=None,
        stall_patience=None,
        phi_relax=None,
        variant=None,
        use_lazy_greedy=None,
        niche_fill=None,
        niche_fill_until_frac=None,
        niche_fill_persist_gens=None,
        corner_archive=None,
        archive_out_mode="replace",
        archive_budget=0.1,
        archive_policy="extreme",
        archive_union_cap=None,
        conv_eps=None,
        div_eps=None,
        div_eps_auto=None,
        div_eps_warmup=0.2,
        div_eps_quantile=None,
        escape_delta_phi_scale=None,
        escape_phi_min=None,
        escape_eta_m=None,
        escape_restart_frac=None,
        phi_bandit=None,
        verbose=False,
    ):
        """
        problem: pymoo Problem 实例
        m: 目标数
        variant: CLDVariant 预定义版本（high_dim/low_dim/multi_modal/speed），
                 显式参数优先于版本配置
        n_partitions: Das-Dennis 分割数（参考方向 W），默认按目标数自适应
        r2_mode: R2 截断标量化 'tche' | 'apd'
        r2_dense_p: R2 权重集 Λ 的 Das-Dennis 分割数；None 表示 Λ=W
        r2_alpha: APD 惩罚指数
        """
        # 版本配置：先应用预定义版本，再叠加显式参数
        if variant is not None:
            cfg = get_variant_config(variant, m=m, problem_type=problem_type,
                                     problem_name=problem_name)
            if protect_mode is None:
                protect_mode = cfg.get("protect_mode")
            if lag_thresh is None:
                lag_thresh = cfg.get("lag_thresh")
            if lag_quantile is None:
                lag_quantile = cfg.get("lag_quantile")
            if d_perp_tiebreak is None:
                d_perp_tiebreak = cfg.get("d_perp_tiebreak")
            if stall_patience is None:
                stall_patience = cfg.get("stall_patience")
            if phi_relax is None:
                phi_relax = cfg.get("phi_relax")
            if use_lazy_greedy is None:
                use_lazy_greedy = cfg.get("use_lazy_greedy", False)
            if niche_fill is None:
                niche_fill = cfg.get("niche_fill", False)
            if niche_fill_until_frac is None:
                niche_fill_until_frac = cfg.get("niche_fill_until_frac", 1.0)
            if niche_fill_persist_gens is None:
                niche_fill_persist_gens = cfg.get("niche_fill_persist_gens", 0)
            if corner_archive is None:
                corner_archive = cfg.get("corner_archive", False)
            if archive_out_mode == "replace":
                archive_out_mode = cfg.get("archive_out_mode", "replace")
            if archive_budget == 0.1:
                archive_budget = cfg.get("archive_budget", 0.1)
            if archive_policy == "extreme":
                archive_policy = cfg.get("archive_policy", "extreme")
            if archive_union_cap is None:
                archive_union_cap = cfg.get("archive_union_cap", None)
            if conv_eps is None:
                conv_eps = cfg.get("conv_eps")
            if div_eps is None:
                div_eps = cfg.get("div_eps")
            if div_eps_auto is None:
                div_eps_auto = cfg.get("div_eps_auto", False)
            div_eps_warmup = cfg.get("div_eps_warmup", div_eps_warmup)
            if div_eps_quantile is None:
                div_eps_quantile = cfg.get("div_eps_quantile", 0.25)
            if escape_delta_phi_scale is None:
                escape_delta_phi_scale = cfg.get("escape_delta_phi_scale")
            if escape_phi_min is None:
                escape_phi_min = cfg.get("escape_phi_min")
            if escape_eta_m is None:
                escape_eta_m = cfg.get("escape_eta_m")
            if escape_restart_frac is None:
                escape_restart_frac = cfg.get("escape_restart_frac")
        # 显式参数默认值
        if protect_mode is None:
            protect_mode = "adaptive"
        if stall_patience is None:
            stall_patience = 10
        if phi_relax is None:
            phi_relax = 0.45
        if use_lazy_greedy is None:
            use_lazy_greedy = False
        self.use_lazy_greedy = use_lazy_greedy
        if niche_fill is None:
            niche_fill = False
        self.niche_fill = niche_fill
        # 方案 A（PG-EF）：niche_fill 分期与持续缺口守卫。
        # until_frac=1.0 + persist_gens=0 时行为与原版逐位一致（wfg4 现配置零影响）
        if niche_fill_until_frac is None:
            niche_fill_until_frac = 1.0
        if niche_fill_persist_gens is None:
            niche_fill_persist_gens = 0
        self.niche_fill_until_frac = niche_fill_until_frac
        self.niche_fill_persist_gens = niche_fill_persist_gens
        # 方案 C：角点档案（默认关闭 → 现有全部配置逐位不变）
        if corner_archive is None:
            corner_archive = False
        self.corner_archive = corner_archive
        self.archive_out_mode = archive_out_mode
        self.archive_budget = archive_budget
        self.archive_policy = archive_policy
        self.archive_union_cap = archive_union_cap
        self.problem = problem
        self.m = m
        self.rng = np.random.default_rng(seed)
        # FRRMAB 在线支配强度选择（审稿对比基线）：truthy 时由 bandit 在离散
        # phi 档位间在线选择 phi_base，取代 PD 闭环控制器（其余机制不变）。
        # 传入 True 用默认档位 {0.15,...,0.5}；或传入臂列表自定义。
        self.phi_bandit = None
        if phi_bandit:
            from src.bandit import FRRMAB
            arms = ([0.15, 0.25, 0.35, 0.45, 0.5] if phi_bandit is True
                    else list(phi_bandit))
            self.phi_bandit = FRRMAB(arms, seed=(seed or 0) + 7919)
        self._bandit_arm = 0

        # 参考方向与种群规模
        if n_partitions is None:
            # 简单规则：m<=5 用 p=3, m<=10 用 p=2, 否则 p=1
            if m <= 5:
                n_partitions = 3
            elif m <= 10:
                n_partitions = 2
            else:
                n_partitions = 1
        self.W = das_dennis_weights(m, n_partitions)
        self.N = int(np.ceil(len(self.W) / 4) * 4)  # 不小于 |W| 的最小 4 倍数
        # 确保 N >= |W|
        if self.N < len(self.W):
            self.N = len(self.W)

        # R2 权重集（可独立于参考方向增密）
        self.r2_mode = r2_mode
        self.r2_alpha = r2_alpha
        self.niche_protect = niche_protect
        self.niche_mating = niche_mating
        self.protect_mode = protect_mode
        # d⊥ 平局打破按问题类型启用：DTLZ 系列（凸/线性 PF）启用以聚焦方向对齐，
        # WFG 系列（复杂 PF）禁用——WFG 的 PF 形状导致"偏离方向"的解是覆盖边缘所需
        if d_perp_tiebreak is None:
            if problem_type is not None:
                d_perp_tiebreak = problem_type.lower() == "dtlz"
            else:
                # 无类型信息时按目标数 fallback：m>=10 启用
                d_perp_tiebreak = m >= 10
        self.d_perp_tiebreak = d_perp_tiebreak
        # lag 阈值按目标数自适应（P11：dtlz7 专项移除，统一用动态分位数逻辑）
        # 动态分位数在 r2select.py 中根据 lag 分布偏度自动选择，lag_thresh 仅作兜底
        if lag_thresh is None:
            if m >= 10:
                lag_thresh = 0.2
            else:
                lag_thresh = 0.3
        self.lag_thresh = lag_thresh
        self.lag_quantile = lag_quantile
        self.fixed_phi = fixed_phi
        if r2_dense_p is not None:
            self.Lambda = das_dennis_weights(m, r2_dense_p)
        else:
            self.Lambda = self.W
        self.theta_w = compute_theta_w(self.Lambda)

        # 逃逸模式：多样性停滞时的响应（反转响应，替代弛豫）
        # - escape_delta_phi_scale: Δφ 倍增（支配强化路线，K.6 证实无增益）
        # - escape_eta_m: 变异分布指数（20→5 大步长跳跃，多样性注入路线）
        # - escape_restart_frac: 每代随机重启个体比例（多样性注入路线）
        self.escape_delta_phi_scale = (escape_delta_phi_scale
                                       if escape_delta_phi_scale is not None else 1.0)
        self.escape_phi_min = escape_phi_min  # None 表示逃逸时维持 phi_min
        self.escape_eta_m = escape_eta_m      # None 表示逃逸时维持 eta_m
        self.escape_restart_frac = escape_restart_frac  # None/0 表示不重启
        # div_eps 在线自动标定（v1.8/K.18）：预热期观测 div_ema 分布自动标定，
        # 替代按 problem_name 查表。auto 时 div_eps 预热后在线设置。
        if div_eps_auto is None:
            div_eps_auto = False
        self.div_eps_auto = div_eps_auto
        self.div_eps_warmup = div_eps_warmup
        self.div_eps_quantile = (0.25 if div_eps_quantile is None
                                 else div_eps_quantile)
        if div_eps_auto:
            div_eps = None   # 自动标定模式忽略显式 div_eps，避免预热期误触发
        escape_enabled = ((div_eps is not None or div_eps_auto)
                          and (self.escape_delta_phi_scale != 1.0
                               or escape_phi_min is not None
                               or escape_eta_m is not None
                               or bool(escape_restart_frac)))

        # 控制器
        self.controller = DominanceController(
            rho_star=rho_star, ki=ki, kd=kd, phi_min=phi_min,
            stall_patience=stall_patience, phi_relax=phi_relax,
            conv_eps=conv_eps if conv_eps is not None else 1e-3,
            div_eps=div_eps,
            escape_on_div_stall=escape_enabled,
            verbose=verbose,
        )
        self.delta_phi = delta_phi
        self.phi_min = phi_min
        self.use_differentiation = use_differentiation

        # 遗传算子参数
        self.eta_c = eta_c
        self.eta_m = eta_m

        # 历史记录
        self.history = {
            "phi": [],
            "rho": [],
            "gamma": [],
            "n_rank1": [],
            "niches": [],
            "igd": [],
            "fe": [],
        }

    # ---------------- 遗传算子 ----------------
    def sbx_crossover(self, parent1, parent2, prob=0.9):
        """模拟二进制交叉 (SBX)"""
        xl, xu = self.problem.xl, self.problem.xu
        c1, c2 = parent1.copy(), parent2.copy()
        if self.rng.random() > prob:
            return c1, c2
        for i in range(len(parent1)):
            if self.rng.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1, x2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    rand = self.rng.random()
                    beta = 1.0 + (2.0 * (x1 - xl[i]) / (x2 - x1))
                    alpha = 2.0 - beta ** -(self.eta_c + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1))
                    c1[i] = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))

                    beta = 1.0 + (2.0 * (xu[i] - x2) / (x2 - x1))
                    alpha = 2.0 - beta ** -(self.eta_c + 1)
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1))
                    c2[i] = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))
        return np.clip(c1, xl, xu), np.clip(c2, xl, xu)

    def polynomial_mutation(self, x, prob=None, eta=None):
        """多项式变异。eta: 覆盖分布指数（逃逸期传更小值产生大步长变异）"""
        xl, xu = self.problem.xl, self.problem.xu
        eta_m = self.eta_m if eta is None else eta
        if prob is None:
            prob = 1.0 / len(x)
        y = x.copy()
        for i in range(len(x)):
            if self.rng.random() < prob:
                delta1 = (y[i] - xl[i]) / (xu[i] - xl[i])
                delta2 = (xu[i] - y[i]) / (xu[i] - xl[i])
                rand = self.rng.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow
                y[i] = y[i] + delta_q * (xu[i] - xl[i])
        return np.clip(y, xl, xu)

    # ---------------- 交配选择 ----------------
    def binary_tournament(self, ranks, dom, n_select, niche_count=None):
        """
        二元锦标赛：Pareto 等级 → 强化支配裁决 → 小生境稀疏度 → 随机。
        niche_count: (n,) 每个解所在方向的成员数，用于稀疏方向优先。
        压力只作用于交配选择，不杀伤环境选择的多样性。
        """
        n = len(ranks)
        parents = []
        pairs = self.rng.integers(0, n, size=(n_select, 2))
        for i, j in pairs:
            if ranks[i] != ranks[j]:
                parents.append(i if ranks[i] < ranks[j] else j)
            elif dom[i, j] and not dom[j, i]:
                parents.append(i)
            elif dom[j, i] and not dom[i, j]:
                parents.append(j)
            elif niche_count is not None and niche_count[i] != niche_count[j]:
                parents.append(i if niche_count[i] < niche_count[j] else j)
            else:
                parents.append(self.rng.choice([i, j]))
        return parents

    # ---------------- 主循环 ----------------
    def run(self, fe_max, callback=None):
        """
        运行算法。
        fe_max: 最大函数评估次数
        callback: 每代回调函数, 签名 callback(pop_X, pop_F, self)
        返回: (pop_X, pop_F)
        """
        # 初始化
        xl, xu = self.problem.xl, self.problem.xu
        D = self.problem.n_var
        X = self.rng.uniform(xl, xu, size=(self.N, D))
        F = np.asarray(self.problem.evaluate(X), dtype=float)
        fe = self.N

        # 初始归一化、Pareto 排序与强化支配矩阵
        z_star = np.min(F, axis=0)
        F_shift = F - z_star
        fronts, ranks = strengthened_nd_sort(F_shift, np.full(self.N, 0.5))
        span = estimate_nadir_extreme(F_shift[fronts[0]], z_star)
        F_norm = F_shift / span
        phi_per_x, assoc, g_hat, lag = direction_differentiation(
            F_norm, self.W, 0.5, self.delta_phi, self.phi_min
        )
        dom = dominance_matrix(F_norm, phi_per_x)
        rho0 = float(np.mean(dom.sum(axis=0) == 0))
        self.controller.e_prev = rho0 - self.controller.rho_star
        assoc_pop = assoc
        niche_cnt = None
        prev_ghat_mean = float(np.mean(g_hat))
        lag_ema = lag.copy()          # P2-a: lag EMA 平滑，α=0.3
        prev_div = float(np.mean(np.std(F_norm, axis=0)))  # 多样性基准值

        t = 0
        # div_eps 在线自动标定状态：预热期采集 div_ema 样本，结束后按分位数标定
        warmup_fe = (int(fe_max * self.div_eps_warmup)
                     if self.div_eps_auto else 0)
        div_samples = []
        div_calibrated = False
        # 方案 A（PG-EF）：niche_fill 分期边界 + 空方向连续空缺计数（跨代状态）
        fe_fill_until = int(fe_max * self.niche_fill_until_frac)
        self.empty_streak = np.zeros(len(self.W), dtype=np.int64)
        # 方案 C：角点档案（纯观察者）+ 末代种群归一化目标（输出期冗余判定用）
        archive = (CornerArchive(self.m, policy=self.archive_policy, W=self.W)
                   if self.corner_archive else None)
        F_norm_last = F_norm
        while fe < fe_max:
            t += 1
            # 1. 交配选择 + 交叉变异生成子代
            parent_idx = self.binary_tournament(ranks, dom, self.N, niche_cnt)
            # 逃逸期变异提升：eta_m 降低（如 20→5）产生大步长变异，穿越谷地
            eta_now = (self.escape_eta_m
                       if (self.controller.escape_active
                           and self.escape_eta_m is not None) else None)
            QX = []
            for i in range(0, self.N, 2):
                p1, p2 = X[parent_idx[i]], X[parent_idx[i + 1]]
                c1, c2 = self.sbx_crossover(p1, p2)
                QX.append(self.polynomial_mutation(c1, eta=eta_now))
                if i + 1 < self.N:
                    QX.append(self.polynomial_mutation(c2, eta=eta_now))
            QX = np.array(QX[: self.N])
            QF = np.asarray(self.problem.evaluate(QX), dtype=float)
            fe += len(QX)

            # 2. 合并 + 归一化（NSGA-III 式两遍法）
            RX = np.vstack([X, QX])
            RF = np.vstack([F, QF])
            z_star_prev = z_star.copy()
            z_star = np.minimum(z_star, np.min(RF, axis=0))
            F_shift = RF - z_star
            # 环境选择用标准 Pareto 排序（第一层同时用于 nadir 估计）
            fronts_R, ranks_R = strengthened_nd_sort(
                F_shift, np.full(len(RF), 0.5)
            )
            span = estimate_nadir_extreme(F_shift[fronts_R[0]], z_star)
            RF_norm = F_shift / span
            # 方案 C：角点档案旁路观察（纯读，不写回任何状态 → 轨迹不变性）
            if archive is not None:
                archive.observe(RX, RF, RF_norm=RF_norm, front1=fronts_R[0])
            # 理想点相对改进量（收敛门控信号）
            improvement = float(np.mean(
                np.maximum(0.0, z_star_prev - z_star) / (span + 1e-12)
            ))

            # 3. 强化支配矩阵（方向差异化 phi，用于交配选择与压力观测）
            # phi_base 来源三选一：FRRMAB 在线选臂 > 固定 phi > PD 闭环控制器
            if self.phi_bandit is not None:
                self._bandit_arm = self.phi_bandit.select()
                phi_base = self.phi_bandit.phi_of(self._bandit_arm)
            elif self.fixed_phi is not None:
                phi_base = self.fixed_phi
            else:
                phi_base = self.controller.get_phi()
            # 逃逸模式：Δφ 倍增 + phi_min 压低，对滞后方向施加更强扩张
            if self.controller.escape_active:
                dphi_eff = self.delta_phi * self.escape_delta_phi_scale
                pmin_eff = (self.escape_phi_min
                            if self.escape_phi_min is not None else self.phi_min)
            else:
                dphi_eff, pmin_eff = self.delta_phi, self.phi_min
            phi_diff, assoc, g_hat, lag_raw = direction_differentiation(
                RF_norm, self.W, phi_base, dphi_eff, pmin_eff
            )
            # P2-a: lag EMA 平滑（α=0.3），滤除单代波动，提升 adaptive 保护稳定性
            lag_ema = 0.3 * lag_raw + 0.7 * lag_ema
            lag = lag_ema
            if self.use_differentiation:
                phi_per_x = phi_diff
            else:
                phi_per_x = np.full(len(RF), phi_base)
            dom_R = dominance_matrix(RF_norm, phi_per_x)
            rho = float(np.mean(dom_R.sum(axis=0) == 0))
            # 综合改进量：理想点坐标改进 + 小生境标量化值改进（横向收敛）
            ghat_mean = float(np.mean(g_hat))
            g_improve = max(0.0, prev_ghat_mean - ghat_mean) / (
                abs(prev_ghat_mean) + 1e-12)
            improvement += g_improve
            prev_ghat_mean = ghat_mean

            # 4. 环境选择：Pareto 等级逐层填入 + R2 贪心截断（多样性受保护）
            # 方案 A：niche_fill 仅在分期窗口（前 until_frac 比例 FE）内生效
            fill_on = self.niche_fill and fe < fe_fill_until
            next_idx = []
            for front in fronts_R:
                if len(next_idx) + len(front) <= self.N:
                    next_idx.extend(front)
                else:
                    k = self.N - len(next_idx)
                    chosen = r2_greedy_truncation(
                        RF_norm, next_idx, front, k, self.Lambda,
                        mode=self.r2_mode, theta_w=self.theta_w,
                        alpha=self.r2_alpha,
                        assoc_dirs=self.W if self.niche_protect else None,
                        protect_mode=self.protect_mode,
                        lag=lag,
                        lag_thresh=self.lag_thresh,
                        lag_quantile=self.lag_quantile,
                        d_perp_tiebreak=self.d_perp_tiebreak,
                        use_lazy_greedy=self.use_lazy_greedy,
                        niche_fill=fill_on,
                        empty_streak=self.empty_streak if fill_on else None,
                        persist_gens=self.niche_fill_persist_gens,
                    )
                    next_idx.extend(chosen)
                    break

            # 5. 多样性变化率（环境选择后种群的归一化目标分布离散度）
            # 种群困于局部前沿时分布趋于稳定，变化率持续趋零 → 停滞信号
            div_now = float(np.mean(np.std(RF_norm[next_idx], axis=0)))
            div_change = abs(div_now - prev_div) / (abs(prev_div) + 1e-12)
            prev_div = div_now

            # 6. 更新控制器（含收敛门控信号 + 多样性停滞信号）
            # bandit 模式仍更新控制器以维持逃逸/多样性门控，仅 phi_base 改由 bandit 定
            if self.fixed_phi is None:
                self.controller.update(rho, improvement, t, div_change=div_change)
            # FRRMAB：记录本代所选臂的 reward（综合改进量），供下一代 UCB 选择
            if self.phi_bandit is not None:
                self.phi_bandit.store(self._bandit_arm, improvement)

            # 6b. div_eps 在线自动标定：预热期采集 div_ema，结束后按分位数标定
            if self.div_eps_auto and not div_calibrated:
                if fe <= warmup_fe:
                    div_samples.append(float(self.controller.div_ema))
                else:
                    div_eps_val = 0.0
                    if len(div_samples) >= 3:
                        div_eps_val = float(np.quantile(
                            np.asarray(div_samples, dtype=float),
                            self.div_eps_quantile))
                    # 数值下限，避免阈值恰为 0 时 div_active 恒为 False
                    div_eps_final = max(div_eps_val, 1e-6)
                    self.controller.div_eps = div_eps_final
                    if self.controller.verbose:
                        arr = np.asarray(div_samples, dtype=float)
                        qs = np.quantile(arr, [0.0, 0.05, 0.1, 0.25, 0.5])
                        print(f"[div_eps 标定] 预热样本数={len(div_samples)}, "
                              f"分位数={self.div_eps_quantile}, "
                              f"最终 div_eps={div_eps_final:.6g}")
                        print(f"[div_eps 分布] min={qs[0]:.6g} p5={qs[1]:.6g} "
                              f"p10={qs[2]:.6g} p25={qs[3]:.6g} p50={qs[4]:.6g}")
                    div_calibrated = True

            # 7. 形成新一代
            X, F = RX[next_idx], RF[next_idx]
            if archive is not None:
                F_norm_last = RF_norm[next_idx]
            ranks = ranks_R[next_idx]
            dom = dom_R[np.ix_(next_idx, next_idx)]
            assoc_pop = assoc[next_idx]
            if self.niche_mating:
                counts = np.bincount(assoc_pop, minlength=len(self.W))
                niche_cnt = counts[assoc_pop]
            else:
                niche_cnt = None

            # 8. 逃逸期部分重启：随机替换一定比例个体，注入新遗传物质
            # 注入后 ranks/dom/assoc 失效，需重建交配选择状态
            if (self.controller.escape_active and self.escape_restart_frac
                    and fe < fe_max):
                k = max(1, int(self.escape_restart_frac * self.N))
                k = min(k, (fe_max - fe))          # 不超 FE 预算
                if k > 0:
                    ridx = self.rng.choice(self.N, size=k, replace=False)
                    X[ridx] = self.rng.uniform(xl, xu, size=(k, D))
                    F[ridx] = np.asarray(self.problem.evaluate(X[ridx]),
                                         dtype=float)
                    fe += k
                    # 重建交配选择状态
                    z_star = np.minimum(z_star, np.min(F, axis=0))
                    F_shift_p = F - z_star
                    fronts_p, ranks = strengthened_nd_sort(
                        F_shift_p, np.full(self.N, 0.5))
                    span_p = estimate_nadir_extreme(F_shift_p[fronts_p[0]],
                                                    z_star)
                    F_norm_p = F_shift_p / span_p
                    phi_p, assoc_pop, _, _ = direction_differentiation(
                        F_norm_p, self.W, phi_base, dphi_eff, pmin_eff)
                    dom = dominance_matrix(F_norm_p, phi_p)
                    if archive is not None:
                        F_norm_last = F_norm_p
                    if self.niche_mating:
                        counts = np.bincount(assoc_pop, minlength=len(self.W))
                        niche_cnt = counts[assoc_pop]

            # 9. 记录历史
            self.history["phi"].append(
                phi_base if self.phi_bandit is not None
                else self.controller.get_phi())
            self.history["rho"].append(rho)
            self.history["gamma"].append(self.controller.get_gamma())
            self.history["n_rank1"].append(len(fronts_R[0]))
            self.history["niches"].append(len(np.unique(assoc[next_idx])))
            self.history["fe"].append(fe)
            if callback is not None:
                callback(X, F, self)

        # 方案 C：输出期一次性合并（演化已结束，不影响动力学）。
        # 合并前保留原始种群快照，供实验断言轨迹不变性
        if archive is not None:
            self.pop_X_raw, self.pop_F_raw = X.copy(), F.copy()
            X, F = archive.finalize(X, F, F_norm_last, self.W,
                                    alpha=self.r2_alpha,
                                    mode=self.archive_out_mode,
                                    budget=self.archive_budget,
                                    union_cap=self.archive_union_cap)
        return X, F
