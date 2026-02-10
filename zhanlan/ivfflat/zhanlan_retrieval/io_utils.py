# zhanlan/ivfflat/zhanlan_retrieval/io_utils.py
from pathlib import Path
import numpy as np
import cv2


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def imread_unicode(p: str):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
