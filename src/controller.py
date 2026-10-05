import numpy as np

class DominanceController:
    """
    闭环支配强度控制器（v1.3：收敛门控 + 滞回 + 压力下限）。
    被控量: 等级-1 解占比 rho（仅在进化活跃期有意义）
    门控信号: 综合改进量的 EMA（理想点坐标改进 + 小生境平均标量化值改进，
              后者捕捉沿 PF 的横向收敛）
    滞回: 连续停滞 stall_patience 代后门控才打开；改进恢复即时关闭，
          防止局部前沿上短暂无改进就彻底释放压力（M3 失败根因）
    压力下限: 收敛后 phi 弛豫到 phi_relax（默认 0.45）而非 0.5，
              始终保留最低交配压力，避免多峰问题锁死在局部前沿
    多样性停滞检测（v1.4）: 综合改进量对多峰题的局部前沿不敏感（间歇性小幅
              横向收敛使 conv_ema 无法低于 conv_eps），补充种群多样性变化率
              信号 div_ema —— 种群困于局部前沿时目标分布离散度趋于稳定，
              变化率持续趋零。两个信号任一低于阈值即视为停滞（或逻辑）
    逃逸模式（v1.5，反转响应）: K.5 实证表明多样性停滞时弛豫（释放压力）
              会在多峰题上削弱逃逸能力（dtlz3 m=5 IGD +11.8%）。因此当停滞由
              多样性信号触发时（div_ema ≤ div_eps），不弛豫，而是进入逃逸模式：
              gamma 保持 1.0，由算法层临时提升 Δφ 并压低 phi_min，对滞后方向
              施加更强扩张以跳出局部前沿；信号恢复即退出。仅改进量信号停滞
              （多样性仍活跃）视为真收敛，维持原弛豫行为
    输出: 基准强度参数 phi
    """

    def __init__(self, rho_star=0.15, ki=0.5, kd=0.1,
                 phi_min=0.15, phi_init=0.5, dead_zone=0.05,
                 conv_eps=1e-3, relax_rate=0.1, ema_alpha=0.1,
                 stall_patience=10, phi_relax=0.45, div_eps=None,
                 escape_on_div_stall=False, verbose=False):
        self.rho_star = rho_star
        self.ki = ki
        self.kd = kd
        self.phi_min = phi_min
        self.phi = phi_init
        self.dead_zone = dead_zone
        self.conv_eps = conv_eps
        self.relax_rate = relax_rate
        self.ema_alpha = ema_alpha
        self.stall_patience = stall_patience   # 滞回：连续停滞代数阈值
        self.phi_relax = phi_relax             # 压力下限
        self.div_eps = div_eps                 # 多样性停滞阈值（None 禁用）
        self.escape_on_div_stall = escape_on_div_stall  # 多样性停滞→逃逸（非弛豫）
        self.verbose = verbose                 # 是否打印 div_eps 标定/逃逸迁移日志
        self.escape_active = False             # 逃逸模式状态
        self.conv_ema = 1.0                    # 初始假设进化活跃
        self.div_ema = 1.0                     # 多样性变化率 EMA，初始假设活跃
        self.stall_counter = 0
        self.e_prev = 0.0
        self.history = []                      # (t, phi, rho, gamma, stall, conv_ema, div_ema, escape)

    def update(self, rho, improvement=1.0, t=None, div_change=None):
        """
        rho: 当前等级-1 解占比
        improvement: 本代综合相对改进量（理想点 + 小生境标量化值）
        div_change: 本代种群多样性相对变化率（可选，需 div_eps 启用）
        返回: 新的 phi
        """
        a = self.ema_alpha
        self.conv_ema = (1 - a) * self.conv_ema + a * improvement
        if div_change is not None:
            # div_ema 始终跟踪（含自动标定预热期，此时 div_eps 尚为 None）
            self.div_ema = (1 - a) * self.div_ema + a * div_change

        # 滞回状态机：改进量 或 多样性变化率 任一低于阈值即停滞
        prev_escape = self.escape_active
        conv_active = self.conv_ema > self.conv_eps
        div_active = True if self.div_eps is None else (self.div_ema > self.div_eps)
        if conv_active and div_active:
            self.stall_counter = 0
            self.escape_active = False       # 信号恢复，退出逃逸
            gamma = 1.0
        else:
            self.stall_counter += 1
            if self.stall_counter >= self.stall_patience:
                if self.escape_on_div_stall and not div_active:
                    # 反转响应：多样性停滞 → 逃逸模式，保持全压不弛豫
                    self.escape_active = True
                    gamma = 1.0
                else:
                    # 仅改进量停滞（真收敛）→ 弛豫
                    gamma = min(1.0, self.conv_ema / self.conv_eps)
                    if self.div_eps is not None:
                        gamma = min(gamma, self.div_ema / self.div_eps)
            else:
                gamma = 1.0        # 观察期内保持加压

        # 逃逸状态迁移日志：记录每次触发/退出时的 div_ema 与 div_eps
        if self.verbose:
            t_now = t if t is not None else len(self.history)
            if self.escape_active and not prev_escape:
                print(f"[div_eps 逃逸触发] t={t_now} div_ema={self.div_ema:.6g} "
                      f"div_eps={self.div_eps:.6g} stall={self.stall_counter}")
            elif prev_escape and not self.escape_active:
                print(f"[div_eps 逃逸退出] t={t_now} div_ema={self.div_ema:.6g} "
                      f"div_eps={self.div_eps:.6g}")

        e = rho - self.rho_star
        if abs(e) <= self.dead_zone:
            ctrl_delta = 0.0
        else:
            ctrl_delta = -self.ki * e - self.kd * (e - self.e_prev)

        new_phi = (self.phi
                   + gamma * ctrl_delta
                   + (1.0 - gamma) * self.relax_rate * (self.phi_relax - self.phi))
        new_phi = np.clip(new_phi, self.phi_min, 0.5)
        self.e_prev = e
        self.phi = new_phi
        self.history.append((t if t is not None else len(self.history),
                             new_phi, rho, gamma,
                             self.stall_counter, self.conv_ema, self.div_ema,
                             float(self.escape_active)))
        return new_phi

    def get_phi(self):
        return self.phi

    def get_gamma(self):
        conv_active = self.conv_ema > self.conv_eps
        div_active = True if self.div_eps is None else (self.div_ema > self.div_eps)
        if conv_active and div_active:
            return 1.0
        if self.stall_counter >= self.stall_patience:
            if self.escape_on_div_stall and not div_active:
                return 1.0    # 逃逸模式不弛豫
            gamma = min(1.0, self.conv_ema / self.conv_eps)
            if self.div_eps is not None:
                gamma = min(gamma, self.div_ema / self.div_eps)
            return gamma
        return 1.0

    def get_history(self):
        return np.array(self.history) if self.history else np.empty((0, 8))
