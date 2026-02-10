# zhanlan/ivfflat/zhanlan_retrieval/service.py
"""
service.py
----------
服务层：保持单例 + 缓存 assets
"""

from __future__ import annotations

from typing import Optional, List
import threading
import numpy as np
import cv2

from .config import RuntimeConfig
from .assets import RetrievalAssets, load_assets
from .pipeline_core import search_indices
from .image_decode import decode_image

class ZhanlanRetrievalService:
    """
    单例服务：
    - 全进程只加载一次 assets
    - 线程安全（用锁保护初始化）
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cfg: Optional[RuntimeConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg: Optional[RuntimeConfig] = None):
        if getattr(self, "_initialized", False):
            return

        self.cfg = cfg or RuntimeConfig()
        self.assets: RetrievalAssets = load_assets(self.cfg)
        self._initialized = True

    def search(self, img, topk: Optional[int] = None) -> List[int]:
        q = decode_image(img)
        k = int(topk) if topk is not None else int(self.cfg.topk)
        return search_indices(q, cfg=self.cfg, assets=self.assets, topk=k)
