# zhanlan/ivfflat/zhanlan_retrieval/faiss_utils.py
import faiss
import numpy as np


def set_faiss_nprobe(index, nprobe=64):
    try:
        base = index
        while hasattr(base, "index"):
            base = base.index
        if hasattr(base, "nprobe") and hasattr(base, "nlist"):
            base.nprobe = min(int(nprobe), int(base.nlist))
            print(f"[FAISS] set nprobe={base.nprobe}/{base.nlist}")
    except Exception:
        pass


def faiss_scores_from_D(index, D: np.ndarray) -> np.ndarray:
    try:
        mt = index.metric_type
    except Exception:
        mt = None
    if mt == faiss.METRIC_L2 or mt == 1:
        D = D.astype(np.float32, copy=False)
        return np.float32(1.0) / (np.float32(1.0) + D)
    else:
        return D
