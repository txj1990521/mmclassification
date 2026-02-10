# zhanlan/ivfflat/zhanlan_retrieval/assets.py
"""
assets.py
---------
一次性加载重资源（FAISS index / metadata / backbone model / YOLO seg provider）

目标：
- 服务启动时加载一次
- 后续每次 query 只做纯计算
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import faiss

from .config import RuntimeConfig
from .faiss_utils import set_faiss_nprobe
from .model_build import build_model
from .yolo_seg import YoloSegProvider


@dataclass
class RetrievalAssets:
    # --- FAISS indices ---
    g_index: faiss.Index
    p_index_s: faiss.Index
    p_index_g: faiss.Index

    # --- metas ---
    img_paths: np.ndarray          # global_meta: img_id -> path
    patch_meta_s: np.ndarray       # patch_stripe_meta
    patch_meta_g: np.ndarray       # patch_grid_meta

    # --- backbone model & preprocess ---
    model: object
    mean: np.ndarray
    std: np.ndarray
    to_rgb: bool

    # --- YOLO seg provider ---
    yolo: YoloSegProvider


def load_assets(cfg: RuntimeConfig, *, nprobe: int = 64) -> RetrievalAssets:
    """
    一次性加载所有重资源。
    注意：这个函数应当只在服务启动 / 第一次使用时调用一次。
    """

    # 1) FAISS
    g_index = faiss.read_index(cfg.global_index)
    p_index_s = faiss.read_index(cfg.patch_stripe_index)
    p_index_g = faiss.read_index(cfg.patch_grid_index)

    set_faiss_nprobe(g_index, nprobe)
    set_faiss_nprobe(p_index_s, nprobe)
    set_faiss_nprobe(p_index_g, nprobe)

    # 2) metas
    img_paths = np.load(cfg.global_meta, allow_pickle=True)
    patch_meta_s = np.load(cfg.patch_stripe_meta, allow_pickle=True)
    patch_meta_g = np.load(cfg.patch_grid_meta, allow_pickle=True)

    # 3) backbone
    model, mean, std, to_rgb = build_model(cfg.cfg_path, cfg.ckpt_path, device=cfg.device)

    # 4) yolo provider
    yolo = YoloSegProvider(cfg.yolo_seg_weights)

    return RetrievalAssets(
        g_index=g_index,
        p_index_s=p_index_s,
        p_index_g=p_index_g,
        img_paths=img_paths,
        patch_meta_s=patch_meta_s,
        patch_meta_g=patch_meta_g,
        model=model,
        mean=mean,
        std=std,
        to_rgb=to_rgb,
        yolo=yolo,
    )
