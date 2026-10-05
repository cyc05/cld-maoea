"""
CLD-MaOEA 版本配置：四个预定义版本，覆盖不同应用场景。

版本说明：
- HIGH_DIM (H): 高维稳定版，m>=8 优先，M4 champion 保护
- LOW_DIM (L): 低维均衡版，m<=5 优先，adaptive + 动态分位数
- MULTI_MODAL (M): 多峰专项版，DTLZ3/7、WFG7/9 等多峰/不连续 PF
- SPEED (S): 速度优先版，大规模问题，惰性贪心加速

用法：
    from src.variants import CLDVariant, get_variant_config

    # 方式1：直接指定版本
    alg = CLDMaOEA(problem, m=10, variant=CLDVariant.HIGH_DIM)

    # 方式2：自动推荐
    variant = auto_variant("dtlz3", m=10, n_var=29)
    alg = CLDMaOEA(problem, m=10, variant=variant)

    # 方式3：手动覆盖（版本配置为默认，显式参数优先）
    alg = CLDMaOEA(problem, m=10, variant=CLDVariant.HIGH_DIM,
                   protect_mode="adaptive")  # 覆盖保护策略

    # 方式4：直接传配置文件路径
    alg = CLDMaOEA(problem, m=10, variant="configs/high_dim.json")

配置文件：configs/ 目录下的 JSON 文件定义各版本参数。
- 标量值直接生效；
- 条件值用 {"default": x, "m>=10": y} 或 {"dtlz": a, "wfg": b, "default": c}
  按 m / problem_type 解析；
- 文件缺失时回退到代码内置默认值。
"""
import json
import os

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "configs")


class CLDVariant:
    """CLD-MaOEA 预定义版本标识。"""
    HIGH_DIM = "high_dim"           # 高维稳定版
    LOW_DIM = "low_dim"             # 低维均衡版
    MULTI_MODAL = "multi_modal"     # 多峰专项版
    SPEED = "speed"                 # 速度优先版
    CUSTOM = "custom"               # 自定义（无预定义配置）


# 版本描述
VARIANT_DESCRIPTIONS = {
    CLDVariant.HIGH_DIM: "高维稳定版 (m>=8): M4 champion 保护，DTLZ d⊥，WFG 无 d⊥",
    CLDVariant.LOW_DIM: "低维均衡版 (m<=5): adaptive 保护 + 动态分位数 lag，无 d⊥",
    CLDVariant.MULTI_MODAL: "多峰专项版: champion 保护，lag_thresh=0.4，d⊥ 启用，dtlz3 逃逸变异提升",
    CLDVariant.SPEED: "速度优先版: champion 保护，固定 lag_thresh=0.3，无 d⊥，惰性贪心",
}


def _parse_override_key(key):
    """解析 override 键，返回 (problem_name, m_condition)。
    m_condition 为 (op, val) 如 ('>=', 10) 或 None 表示不限制。"""
    if "+" not in key:
        return key, None
    parts = key.split("+")
    pname = parts[0]
    mcond = "+".join(parts[1:])  # e.g. "m=10" or "m>=10"
    if mcond.startswith("m>="):
        return pname, (">=", int(mcond[3:]))
    elif mcond.startswith("m="):
        return pname, ("=", int(mcond[2:]))
    return key, None


def _check_override(problem_name, m, pname, mcond):
    """检查当前 (problem_name, m) 是否匹配一个 override 条件。"""
    if problem_name is None:
        return False
    if problem_name.lower() != pname.lower():
        return False
    if mcond is None:
        return True
    op, val = mcond
    if op == ">=":
        return m is not None and m >= val
    elif op == "=":
        return m is not None and m == val
    return False


def _resolve_conditional(value, m, problem_type):
    """
    解析 JSON 中的条件值。
    - {"default": x, "m>=10": y}: 按目标数解析
    - {"dtlz": a, "wfg": b, "default": c}: 按问题类型解析
    其他值原样返回。
    """
    if not isinstance(value, dict):
        return value
    if "m>=10" in value:
        if m is not None and m >= 10:
            return value["m>=10"]
        return value.get("default")
    if "dtlz" in value or "wfg" in value:
        if problem_type in value:
            return value[problem_type]
        return value.get("default")
    return value.get("default")


def load_config_file(path, m=None, problem_type=None, problem_name=None):
    """从 JSON 配置文件加载参数，解析条件值与 overrides 后返回参数字典。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = {}
    for k, v in raw.items():
        if k in ("name", "description", "overrides", "_notes"):
            continue
        cfg[k] = _resolve_conditional(v, m, problem_type)
    # 应用 overrides（按问题名+目标数精细覆盖）
    overrides = raw.get("overrides", {})
    for key, patch in overrides.items():
        pname, mcond = _parse_override_key(key)
        if _check_override(problem_name, m, pname, mcond):
            for pk, pv in patch.items():
                cfg[pk] = _resolve_conditional(pv, m, problem_type)
    return cfg


def get_variant_config(variant, m=None, problem_type=None, problem_name=None):
    """
    获取版本对应的配置参数字典。

    参数
    ----
    variant : str
        CLDVariant 枚举值，或 JSON 配置文件路径（优先从
        configs/<variant>.json 加载，缺失时回退内置默认值）
    m : int, optional
        目标数，用于 lag_thresh 自适应
    problem_type : str, optional
        "dtlz" | "wfg" | None，用于 d_perp_tiebreak 判断
    problem_name : str, optional
        问题名如 "dtlz1", "wfg3"，用于 overrides 匹配

    返回
    ----
    dict : 可直接传入 CLDMaOEA 的参数字典
    """
    # 直接传入配置文件路径
    if isinstance(variant, str) and variant.endswith(".json"):
        return load_config_file(variant, m, problem_type, problem_name)
    # 预定义版本：优先读 configs/<variant>.json
    cfg_path = os.path.join(_CONFIG_DIR, f"{variant}.json")
    if os.path.isfile(cfg_path):
        cfg = load_config_file(cfg_path, m, problem_type, problem_name)
        cfg.setdefault("problem_type", problem_type)
        cfg.setdefault("problem_name", problem_name)
        return cfg
    # 回退内置默认值
    if variant == CLDVariant.HIGH_DIM:
        return _high_dim_config(m, problem_type)
    elif variant == CLDVariant.LOW_DIM:
        return _low_dim_config(m, problem_type)
    elif variant == CLDVariant.MULTI_MODAL:
        return _multi_modal_config(m, problem_type)
    elif variant == CLDVariant.SPEED:
        return _speed_config(m, problem_type)
    else:
        return {}


def _high_dim_config(m, problem_type):
    """高维稳定版：M4 champion 保护，稳定性优先。"""
    return {
        "protect_mode": "champion",
        "lag_thresh": 0.2 if (m is not None and m >= 10) else 0.3,
        "lag_quantile": None,  # 禁用动态分位数
        "d_perp_tiebreak": problem_type == "dtlz" if problem_type else False,
        "problem_type": problem_type,
        "problem_name": None,
        "stall_patience": 10,
        "phi_relax": 0.45,
        "conv_eps": 0.01,  # P0: 提高收敛阈值，激活弛豫路径
    }


def _low_dim_config(m, problem_type):
    """低维均衡版：adaptive + 动态分位数，覆盖均匀性优先。"""
    return {
        "protect_mode": "adaptive",
        "lag_thresh": 0.3,  # 兜底值，动态分位数优先
        "lag_quantile": None,  # None 表示启用动态分位数逻辑（r2select 内部处理）
        "d_perp_tiebreak": False,  # 禁用 d⊥，避免牺牲覆盖
        "problem_type": problem_type,
        "problem_name": None,
        "stall_patience": 10,
        "phi_relax": 0.45,
        "conv_eps": 0.01,
    }


def _multi_modal_config(m, problem_type):
    """多峰专项版：champion + 高 lag_thresh + d⊥，逃逸局部前沿。"""
    return {
        "protect_mode": "champion",  # 固定 champion，非 adaptive
        "lag_thresh": 0.4,  # 提高保护比例
        "lag_quantile": None,
        "d_perp_tiebreak": True,  # 启用 d⊥ 聚焦方向
        "problem_type": problem_type,
        "problem_name": None,
        "stall_patience": 15,  # 更长观察期
        "phi_relax": 0.40,  # 保留更多压力
        "conv_eps": 0.01,  # P0: 提高收敛阈值，激活弛豫路径
    }


def _speed_config(m, problem_type):
    """速度优先版：简化计算，惰性贪心加速。"""
    return {
        "protect_mode": "champion",
        "lag_thresh": 0.3,  # 固定，无动态计算
        "lag_quantile": None,
        "d_perp_tiebreak": False,
        "problem_type": problem_type,
        "problem_name": None,
        "stall_patience": 8,  # 稍短观察期
        "phi_relax": 0.45,
        "conv_eps": 0.01,
        # 注：惰性贪心在 r2select.py 中实现，此处通过参数标记
        "use_lazy_greedy": True,
    }


def auto_variant(problem_name, m, n_var=None):
    """
    根据问题特征自动推荐版本。

    参数
    ----
    problem_name : str
        问题名称，如 "dtlz1", "wfg3"
    m : int
        目标数
    n_var : int, optional
        变量数，用于 SPEED 判断

    返回
    ----
    str : CLDVariant 枚举值
    """
    problem_name = problem_name.lower()
    is_dtlz = problem_name.startswith("dtlz")
    is_wfg = problem_name.startswith("wfg")
    is_multi_modal = problem_name in ("dtlz3", "dtlz7", "wfg7", "wfg9")

    # 大规模问题优先速度
    if n_var is not None and n_var > 50:
        return CLDVariant.SPEED

    # 高维优先稳定
    if m >= 8:
        return CLDVariant.HIGH_DIM

    # 低维多峰专项
    if m <= 5 and is_multi_modal:
        return CLDVariant.MULTI_MODAL

    # 低维均衡
    if m <= 5:
        return CLDVariant.LOW_DIM

    # 中维默认高维稳定
    return CLDVariant.HIGH_DIM


def list_variants():
    """列出所有可用版本及其描述。"""
    return {
        v: VARIANT_DESCRIPTIONS[v]
        for v in [CLDVariant.HIGH_DIM, CLDVariant.LOW_DIM,
                  CLDVariant.MULTI_MODAL, CLDVariant.SPEED]
    }
