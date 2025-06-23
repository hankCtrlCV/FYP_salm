import yaml, pathlib, copy

ROOT = pathlib.Path(__file__).resolve().parents[1]  # FYP_salm/

def _read(f):
    with open(f, 'r') as fp:
        return yaml.safe_load(fp)

def load_common() -> dict:
    return _read(ROOT / "configs/common.yaml")

def load_gbp() -> dict:
    # 在公共 cfg 上再 overlay gbp 专用键
    common = load_common()
    gbp    = _read(ROOT / "configs/ut.yaml")
    merged = copy.deepcopy(common)
    merged.update(gbp)          # 简单扁平合并即可
    return merged
