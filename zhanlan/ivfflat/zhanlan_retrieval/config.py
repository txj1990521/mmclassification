# zhanlan/ivfflat/zhanlan_retrieval/config.py
from dataclasses import dataclass
import os
import torch

# from your existing shared utils
from hybrid_shared import STRIPE_LONG_EDGE


@dataclass
class RuntimeConfig:
    # ---------- device ----------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seg_device: str = "cuda:0"
    yolo_device = 0  # ultralytics typical: 0 / "cpu" / "cuda:0"

    # ---------- segmentation ----------
    seg_score_thr: float = 0.6
    seg_use_classes = None

    # ---------- model ----------
    cfg_path: str = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
    ckpt_path: str = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

    # ---------- YOLO seg ----------
    yolo_seg_weights: str = r"D:\zhanlanProject\ultralyticsV8\runs\huaxing\exp12\weights\best.pt"
    yolo_imgsz: int = 640
    yolo_iou: float = 0.5
    yolo_retina_masks: bool = True
    yolo_max_det: int = 100

    # ---------- query / index ----------
    query_img: str = r"D:\zhanlan\segment_data\花色随机拍摄照片\IMG_20260129_114047(1).jpg"

    index_dir: str = r"D:\zhanlan\faiss_database_hybrid_new_data"
    out_dir: str = r"D:\zhanlan\search_vis"
    topk: int = 12

    # derived paths
    @property
    def global_index(self) -> str:
        return os.path.join(self.index_dir, "global.index")

    @property
    def global_meta(self) -> str:
        return os.path.join(self.index_dir, "global_img_paths.npy")

    @property
    def patch_stripe_index(self) -> str:
        return os.path.join(self.index_dir, "patch_stripe.index")

    @property
    def patch_grid_index(self) -> str:
        return os.path.join(self.index_dir, "patch_grid.index")

    @property
    def patch_stripe_meta(self) -> str:
        return os.path.join(self.index_dir, "patch_stripe_meta.npy")

    @property
    def patch_grid_meta(self) -> str:
        return os.path.join(self.index_dir, "patch_grid_meta.npy")

    # =========================
    # STRIPE_SHARED_CONSTANTS
    # =========================
    stripe_long_edge: int = int(STRIPE_LONG_EDGE)
    stripe_ar_thr: float = 2.7

    stripe_win_h: int = 224
    stripe_stride: int = 48
    stripe_max_patches: int = 24
    stripe_center_frac: float = 0.92
    stripe_jitter: int = 8
    stripe_win_w_frac: float = 0.85
    stripe_win_w_min: int = 160
    stripe_win_w_max: int = 256
    stripe_seed: int = 123

    # ---------- retrieval ----------
    topg: int = 2000
    patch_topk_per_qpatch: int = 800
    top_patch_images: int = 4000
    rrf_k: int = 60

    # ---------- geom rerank (ORB RANSAC) ----------
    geom_topn: int = 120
    min_inliers: int = 8
    ransac_thresh: float = 5.0

    # ---------- global multi-view ----------
    views_per_image: int = 12
    resize_short: int = 256
    crop_size: int = 224
    view_plan = [
        (0,   1, 5),
        (-15, 1, 1),
        (15,  1, 1),
        (-30, 1, 0),
        (30,  1, 0),
    ]
    view_batch: int = 256

    # ---------- patch rerank / feature-map patches ----------
    feat_level: int = -2
    rmac_input_size: int = 512
    keep_patches: int = 256
    border: float = 0.05

    # ROI selection
    q_roi_frac: float = 0.18
    c_roi_frac: float = 0.18
    q_roi_ratio: float = 0.50
    c_roi_ratio: float = 1.00

    # Geom score / periodic
    margin: float = 0.015
    min_keep: int = 6
    bin_size: float = 0.05
    topk_core: int = 64

    periodic_peak_thr: float = 0.25
    periodic_cover_thr: float = 0.22
    periodic_cover_topm_thr: float = 0.70
    periodic_cover_xy_thr: float = 0.22
    topm: int = 6
