import numpy as np, json

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k].tolist() if isinstance(d[k], np.ndarray) and
                         d[k].dtype == object else d[k]
            for k in d.files}
