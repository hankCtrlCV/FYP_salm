import yaml, pathlib, copy

ROOT = pathlib.Path(__file__).resolve().parents[1]        # FYP_salm/

# ----------------------------------------------------------------------
def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _flatten(dic: dict, out: dict, prefix=""):
    """
    递归下压，所有嵌套键都展平成一级。
    prefix 用于保留父路径信息（此处暂不用，可为空)。
    """
    for k, v in dic.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, out, prefix="")        # 继续下压
        else:
            out.setdefault(key, v)

# ----------------------------------------------------------------------
def load_common() -> dict:
    """通用 + Measurement + Builder 都能用"""
    data = _read(ROOT / "configs/slam_common.yaml")
    flat = {}
    _flatten(data, flat)
    return flat

# ---------- 下面是 GBP 专用 ---------------------------------------------------
def _translate_performance_keys(src: dict) -> dict:
    """
    performance.* → Builder 里的真实键
    """
    mapping = {                       # <YAML 键> : <Builder 键>
        "enable_vectorization":   "enable_true_vectorization",
        "enable_batch_processing": "enable_batch_processing",
        "enable_factor_reordering": "enable_factor_reordering",
    }
    out = {}
    for k, v in src.items():
        if k in mapping:
            out[mapping[k]] = v
    return out

def load_gbp() -> dict:
    """
    common + gbp_ut.yaml → 扁平 dict
    gbp_ut.yaml 结构：
        ut: { ... }
        performance: { ... }
    """
    cfg = load_common()

    # 读取 gbp_ut.yaml
    ub = _read(ROOT / "configs/gbp_ut.yaml")

    # 1. ut.* 直接平铺
    if "ut" in ub and isinstance(ub["ut"], dict):
        _flatten(ub["ut"], cfg)

    # 2. numerical_config 放进去（避免被 _flatten 毁掉类型）
    if "numerical_config" in ub.get("ut", {}):
        cfg["numerical_config"] = ub["ut"]["numerical_config"]

    # 3. performance.* -> Builder 真实键
    if "performance" in ub and isinstance(ub["performance"], dict):
        cfg.update(_translate_performance_keys(ub["performance"]))

    return cfg
