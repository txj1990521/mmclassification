# zhanlan/ivfflat/zhanlan_retrieval/image_decode.py
"""
image_decode.py
---------------
把各种输入统一解码成 np.ndarray(BGR)
"""

import base64
import numpy as np
import cv2


def decode_image(inp):
    """
    支持：
    - 文件路径 (str)
    - base64 字符串（支持 data:image/...;base64,xxx）
    - np.ndarray (BGR)
    """
    if isinstance(inp, np.ndarray):
        return inp

    if isinstance(inp, str):
        # 1) 尝试当成文件路径
        img = cv2.imread(inp)
        if img is not None:
            return img

        # 2) 尝试 base64
        try:
            b64 = inp
            if "," in b64 and b64.strip().startswith("data:image"):
                b64 = b64.split(",", 1)[1]

            data = base64.b64decode(b64)
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass

    raise ValueError("Input image must be file path, base64 string, or np.ndarray(BGR)")
