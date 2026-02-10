# zhanlan/ivfflat/zhanlan_retrieval/api.py
"""
api.py
------
对外唯一入口：ZhanlanRetriever.search(img)->indices

外部调用方不需要知道：
- YOLO / FAISS / RRF / HeadGate / GeomRerank
"""

from __future__ import annotations

from typing import Optional, List
import numpy as np

from .config import RuntimeConfig
from .service import ZhanlanRetrievalService


class ZhanlanRetriever:
    """
    Public API (黑盒)
    """

    def __init__(self, cfg: Optional[RuntimeConfig] = None):
        self._service = ZhanlanRetrievalService(cfg or RuntimeConfig())

    def search(self, img, topk: Optional[int] = None) -> List[int]:
        """
        对外黑盒接口：
        输入：
          - img: str(path) 或 np.ndarray(BGR)
        输出：
          - List[int]: image indices
        """
        return self._service.search(img, topk=topk)
