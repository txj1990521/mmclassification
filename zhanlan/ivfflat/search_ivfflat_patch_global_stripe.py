#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import faiss
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS
from rembg import remove, new_session
import multiprocessing as mp
from mmdet.apis import DetInferencer
from zhanlan.utils.seg_cropper import SegMaskCropper
from pycocotools import mask as maskUtils
import time


mp.set_start_method("spawn", force=True)


# ========== SEG CONFIG ==========
SEG_MODEL_CONFIG = r"D:/zhanlanProject/mmdetection/zhanlan/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
SEG_WEIGHTS      = r"D:/zhanlanProject/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_12.pth"
SEG_DEVICE       = "cuda:0"  # or "cpu"

SEG_SCORE_THR    = 0.6       # 推理阈值
SEG_USE_CLASSES  = None      # 例如 [0,1,2] 只保留这些label；None 表示不筛
# ---------- Rerank ----------
FEAT_LEVEL = -2
RMAC_INPUT_SHORT = 512
KEEP_PATCHES = 256
BORDER = 0.05

MARGIN = 0.015      # 0.01~0.02
MIN_KEEP = 6        # 6~10
BIN_SIZE = 0.05
TOPK_CORE = 64

PERIODIC_PEAK_THR  = 0.25
PERIODIC_COVER_THR = 0.22
PERIODIC_COVER_TOPM_THR = 0.70  # 新增：topM bins 覆盖占比
PERIODIC_COVER_XY_THR   = 0.22  # 你现在这个更像空间覆盖阈值

TOPM = 6
# ============================================================
# CONFIG
# ============================================================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

QUERY_IMG = r"D:\zhanlan\qurrey_data\微信图片_20260128110352_32_7.jpg"

INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid_new_data"
GLOBAL_INDEX = os.path.join(INDEX_DIR, "global.index")
PATCH_INDEX  = os.path.join(INDEX_DIR, "patch.index")
GLOBAL_META  = os.path.join(INDEX_DIR, "global_img_paths.npy")
PATCH_META   = os.path.join(INDEX_DIR, "patch_meta.npy")

OUT_DIR = r"D:\zhanlan\search_vis"
TOPK = 12

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- retrieval ----------
TOPG = 2000
PATCH_TOPK_PER_QPATCH = 400
TOP_PATCH_IMAGES = 4000
RRF_K = 60

# ---------- geom rerank ----------
GEOM_TOPN = 200        # 只在 RRF 后 topN 上跑
MIN_INLIERS = 8        # RANSAC 最少内点
RANSAC_THRESH = 5.0
# ---------- rerank config ----------
FEAT_LEVEL = -2
RMAC_INPUT_SIZE = 512
KEEP_PATCHES = 256
BORDER = 0.05

# ROI / selection
Q_ROI_FRAC = 0.18      # query roi energy box fraction
C_ROI_FRAC = 0.18      # candidate roi energy box fraction
Q_ROI_RATIO = 0.50     # query: ROI patches ratio (0~1)
C_ROI_RATIO = 1.00     # candidate: use ROI if exists, else fallback

def pad_to_square(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_REFLECT101)

_SEG_INFER = None
def get_seg_inferencer():
    global _SEG_INFER
    if _SEG_INFER is None:
        _SEG_INFER = DetInferencer(
            model=SEG_MODEL_CONFIG,
            weights=SEG_WEIGHTS,
            device=SEG_DEVICE,
            palette="random",
        )
    return _SEG_INFER
# 0) 新增：给 head 用的“干净图”（不 feather，不 mean 背景）
# 放在 cand_gate_score 上面/附近即可
# =========================
def make_head_view(img_bgr: np.ndarray, prefer_gray=True):
    """
    给 stripe_grid_head_v21 用：尽量保留纹理方向，不要 feather/mean 背景干扰。
    - 如果你想更强：可以把 blur 去掉或改弱（在 stripe_grid_head_v21 内部）
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    # 只用于 head：取中间区域，避开样卡上下边/LOGO/手
    H, W = img_bgr.shape[:2]
    y1 = int(H * 0.15)
    y2 = int(H * 0.85)
    x1 = int(W * 0.10)
    x2 = int(W * 0.90)
    crop = img_bgr[y1:y2, x1:x2].copy()
    return crop


def _largest_cc(mask_u8: np.ndarray):
    """mask_u8: 0/255"""
    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = 1 + int(np.argmax(areas))
    out = (labels_cc == best_i).astype(np.uint8) * 255
    return out

def _safe_pad_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W-1, int(x1)))
    y1 = max(0, min(H-1, int(y1)))
    x2 = max(1, min(W,   int(x2)))
    y2 = max(1, min(H,   int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def _fill_background(crop_bgr, crop_mask_u8, mode="mean"):
    """
    mode:
      - "mean": 背景填充为mask内像素均值（推荐，最不引入新模式）
      - "white": 背景填白
      - "edge": 先用原图，再把背景轻微模糊（尽量不引入强边界）
    """
    if crop_bgr is None or crop_mask_u8 is None:
        return crop_bgr

    m = crop_mask_u8.astype(bool)
    if m.sum() < 10:
        return crop_bgr

    out = crop_bgr.copy()
    if mode == "white":
        out[~m] = (255, 255, 255)
        return out

    if mode == "edge":
        # 背景做轻微模糊，减少“硬边框”特征
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out[~m] = blur[~m]
        return out

    # default: mean
    mean_color = out[m].mean(axis=0)
    out[~m] = mean_color
    return out

def crop_by_mmdet_mask_final(
    img_bgr: np.ndarray,
    pad: int = 10,
    min_area_frac: float = 0.06,
    score_thr: float = 0.6,
    use_classes=None,
    merge_all: bool = True,

    # --- 新增：矫正 ---
    do_rectify: bool = True,
    rectify_pad: int = 10,
    warp_border: str = "reflect",   # "reflect" | "replicate"

    # --- 新增：背景处理 ---
    bg_mode: str = "mean",          # "mean" | "white" | "edge"

    debug_dir: str = None
):
    """
    返回:
      crop_bgr, crop_mask_u8
    - crop_bgr: 已用 mask 抠图且背景已处理（不再是黑边）
    - crop_mask_u8: 0/255, 与 crop_bgr 对齐
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, None

    H, W = img_bgr.shape[:2]
    infer = get_seg_inferencer()

    res = infer(
        inputs=[img_bgr],
        pred_score_thr=float(score_thr),
        batch_size=1,
        show=False,
        no_save_vis=True,
        no_save_pred=True,
        print_result=False,
        out_dir=""
    )

    inst = _extract_instances_from_inferencer_result(res)
    if inst is None:
        return img_bgr, None

    labels = inst.get("labels", None)
    scores = inst.get("scores", None)
    masks  = inst.get("masks", None)
    if scores is None or masks is None:
        return img_bgr, None

    scores = np.asarray(scores, dtype=np.float32)
    masks_bool = _decode_mmdet_masks(masks, H, W)
    if masks_bool is None:
        return img_bgr, None

    keep = scores >= float(score_thr)
    if labels is not None and use_classes is not None:
        labels = np.asarray(labels)
        keep = keep & np.isin(labels, np.asarray(use_classes))

    idx = np.where(keep)[0]
    if idx.size == 0:
        return img_bgr, None

    if not merge_all:
        best = idx[np.argmax(scores[idx])]
        m = masks_bool[best]
    else:
        m = np.any(masks_bool[idx].astype(bool), axis=0)

    mask_u8 = (m.astype(np.uint8) * 255)

    # --- 清理 ---
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  ker, iterations=1)

    # 最大连通域
    mask_u8 = _largest_cc(mask_u8)
    if mask_u8 is None:
        return img_bgr, None

    area = float((mask_u8 > 0).sum())
    if area < float(min_area_frac) * (H * W):
        return img_bgr, None

    # --- 轮廓/最小外接旋转矩形 ---
    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 20:
        return img_bgr, None

    if do_rectify:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
        (cx, cy), (rw, rh), ang = rect

        # OpenCV 的 angle 规则：一般要把长边对齐到水平
        if rw < rh:
            ang = ang + 90.0

        # 旋转矩阵
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)

        borderMode = cv2.BORDER_REFLECT101 if warp_border == "reflect" else cv2.BORDER_REPLICATE

        rot_img = cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=borderMode)
        rot_msk = cv2.warpAffine(mask_u8, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 旋转后用 mask 的 tight bbox 裁切（注意：这是“mask bbox”，不是 det bbox）
        ys2, xs2 = np.where(rot_msk > 0)
        if ys2.size < 20:
            return img_bgr, None

        x1, x2 = xs2.min() - rectify_pad, xs2.max() + 1 + rectify_pad
        y1, y2 = ys2.min() - rectify_pad, ys2.max() + 1 + rectify_pad

        crop_img = _safe_pad_crop(rot_img, x1, y1, x2, y2)
        crop_msk = _safe_pad_crop(rot_msk, x1, y1, x2, y2)
    else:
        # 不矫正：直接按 mask bbox 裁切（仍然不是 det bbox）
        x1, x2 = xs.min() - pad, xs.max() + 1 + pad
        y1, y2 = ys.min() - pad, ys.max() + 1 + pad
        crop_img = _safe_pad_crop(img_bgr, x1, y1, x2, y2)
        crop_msk = _safe_pad_crop(mask_u8, x1, y1, x2, y2)

    if crop_img is None or crop_msk is None or crop_img.size == 0:
        return img_bgr, None

    # --- 关键：用 mask 做抠图，避免黑边干扰 ---
    crop_msk_u8 = (crop_msk > 0).astype(np.uint8) * 255
    crop_msk_bool = crop_msk_u8.astype(bool)

    # 先做背景填充，再把前景保留（减少“硬边界”）
    filled = _fill_background(crop_img, crop_msk_u8, mode=bg_mode)
    out = filled.copy()
    out[~crop_msk_bool] = filled[~crop_msk_bool]  # 背景
    out[crop_msk_bool]  = crop_img[crop_msk_bool] # 前景原像素
    tag = f"q_{int(time.time())}"
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_mask.png"), mask_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_mask.png"), crop_msk_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_out.png"), out)

    return out, crop_msk_u8


def apply_mask_cut(
    img_bgr: np.ndarray,
    mask_uint8: np.ndarray,      # 0/255
    pad: int = 10,
    bg_mode: str = "mean",       # "mean" | "gray" | "blur"
    feather: int = 9,            # 边缘羽化核大小，0=不羽化
):
    H, W = img_bgr.shape[:2]

    # 1) 最大连通域后，你可以先做个 bbox 用来裁掉大黑边（可选但很有用）
    #    注意：这不是“用bbox当结果”，只是为了减少空背景面积
    ys, xs = np.where(mask_uint8 > 0)
    if len(xs) == 0:
        return img_bgr
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad); y1 = min(H - 1, y1 + pad)

    img_crop = img_bgr[y0:y1+1, x0:x1+1].copy()
    m = mask_uint8[y0:y1+1, x0:x1+1].copy()

    # 2) feather：把 mask 变成 0~1 的 alpha，并做一点高斯模糊
    alpha = (m.astype(np.float32) / 255.0)
    if feather and feather > 0:
        k = int(feather)
        if k % 2 == 0: k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    alpha3 = np.repeat(alpha[:, :, None], 3, axis=2)

    # 3) 生成背景
    if bg_mode == "mean":
        bg_color = img_crop.reshape(-1, 3).mean(axis=0).astype(np.float32)  # BGR
        bg = np.zeros_like(img_crop, dtype=np.float32)
        bg[:] = bg_color
    elif bg_mode == "gray":
        bg = np.zeros_like(img_crop, dtype=np.float32)
        bg[:] = (127, 127, 127)
    elif bg_mode == "blur":
        bg = cv2.GaussianBlur(img_crop, (0, 0), 15).astype(np.float32)
    else:
        raise ValueError("bg_mode must be mean/gray/blur")

    fg = img_crop.astype(np.float32)
    out = fg * alpha3 + bg * (1.0 - alpha3)
    return out.astype(np.uint8)

def _decode_mmdet_masks(masks_obj, H: int, W: int):
    """
    输入:
      masks_obj 可能是：
        1) (N,H,W) bool/uint8
        2) list[dict] / np.ndarray(dtype=object): COCO RLE {"size":[H,W],"counts":...}
    输出:
      masks_bool: np.ndarray (N,H,W) bool
      若失败返回 None
    """
    if masks_obj is None:
        return None

    # case1: already dense
    if isinstance(masks_obj, np.ndarray) and masks_obj.ndim == 3:
        m = masks_obj
        if m.shape[1] != H or m.shape[2] != W:
            return None
        return m.astype(bool)

    # case2: list/obj array of RLE dicts
    if isinstance(masks_obj, (list, tuple)):
        rles = list(masks_obj)
    elif isinstance(masks_obj, np.ndarray) and masks_obj.dtype == object and masks_obj.ndim == 1:
        rles = masks_obj.tolist()
    else:
        # some versions may store as BitmapMasks-like object with .to_ndarray()
        if hasattr(masks_obj, "to_ndarray"):
            m = masks_obj.to_ndarray()
            if m.ndim == 3 and m.shape[1] == H and m.shape[2] == W:
                return m.astype(bool)
        return None

    if len(rles) == 0:
        return None

    decoded = []
    for rle in rles:
        if isinstance(rle, dict) and "counts" in rle and "size" in rle:
            # decode returns (H,W,1) or (H,W)
            dm = maskUtils.decode(rle)
            if dm.ndim == 3:
                dm = dm[:, :, 0]
            # pycocotools uses Fortran order internally, but decode result is correct spatially
            if dm.shape[0] != H or dm.shape[1] != W:
                # 有时 size 里是 (W,H)？少见，但这里兜底处理
                if dm.shape[0] == W and dm.shape[1] == H:
                    dm = dm.T
                else:
                    return None
            decoded.append(dm.astype(bool))
        else:
            return None

    return np.stack(decoded, axis=0)  # (N,H,W)

def _extract_instances_from_inferencer_result(res):
    """
    兼容不同 mmdet/mmengine 版本输出结构，尽量拿到 instances dict：
      instances = {"scores": ..., "labels": ..., "masks": ..., "bboxes": ...}
    """
    preds = res.get("predictions", None) or res.get("preds", None)
    if preds is None:
        return None

    # batch 情况：取第一张
    if isinstance(preds, list) and len(preds) > 0:
        p0 = preds[0]
    else:
        p0 = preds

    # 常见结构：p0["pred_instances"] 或 p0["instances"]
    if isinstance(p0, dict):
        if "pred_instances" in p0:
            return p0["pred_instances"]
        if "instances" in p0:
            return p0["instances"]
        # 有的版本直接平铺
        if ("masks" in p0) or ("scores" in p0):
            return p0
    return None

def stripe_grid_head_v21(img_bgr, resize_long=512, nbins=36):
    if img_bgr is None or img_bgr.size == 0:
        return {"stripe_score":0.0, "grid_score":0.0, "ori_peakedness":0.0}

    h, w = img_bgr.shape[:2]
    s = resize_long / float(max(h, w))
    if s < 1.0:
        img = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    else:
        img = img_bgr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)

    # ✅ A：阈值更低，保留弱方向
    thr = np.percentile(mag, 55)   # 50~60
    mask = mag > thr
    if mask.sum() < 200:
        return {"stripe_score":0.0, "grid_score":0.0, "ori_peakedness":0.0}

    mag = mag[mask]
    ang = ang[mask]

    hist, _ = np.histogram(ang, bins=nbins, range=(0, np.pi), weights=mag)
    hist = hist.astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(1,-1), (1,5), 0).ravel()

    eps = 1e-6
    p = hist / (hist.sum() + eps)

    peaked = float(p.max() / (p.mean() + eps))

    k1 = int(np.argmax(p))
    # ✅ B：屏蔽更窄
    ban = max(1, nbins // 18)  # 36->2
    p2 = p.copy()
    p2[max(0,k1-ban):min(nbins,k1+ban+1)] = 0
    k2 = int(np.argmax(p2))

    peak1 = float(p[k1])
    peak2 = float(p[k2])

    a1 = k1 / nbins * np.pi
    a2 = k2 / nbins * np.pi
    d = abs(a1 - a2)
    d = min(d, np.pi - d)

    # ✅ C：正交容忍更宽
    ortho = np.exp(-((d - (np.pi/2))**2) / (2*(0.40**2)))

    stripe_score = peak1 * (peaked / 6.0)
    stripe_score = float(np.clip(stripe_score, 0.0, 1.0))

    # grid 不要只看 peak2，给它一点“弱峰补偿”
    grid_score = (0.65*peak1 + 0.35*peak2) * ortho * (peaked / 6.0)
    grid_score = float(np.clip(grid_score, 0.0, 1.0))

    return {"stripe_score":stripe_score, "grid_score":grid_score, "ori_peakedness":peaked}

@torch.no_grad()
def make_single_tensor_for_rerank(img_bgr: np.ndarray, mean, std, to_rgb: bool):
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = pad_to_square(img_rgb)
    img_rgb = cv2.resize(img_rgb, (RMAC_INPUT_SIZE, RMAC_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)

@torch.no_grad()
def extract_featmap(model, batch_tensor: torch.Tensor, prefer_level: int):
    out = model.backbone(batch_tensor)

    if isinstance(out, dict):
        if "feat" in out:
            out = out["feat"]
        elif "features" in out:
            out = out["features"]
        else:
            out = list(out.values())[-1]

    if isinstance(out, (tuple, list)):
        n = len(out)
        lvl = prefer_level
        if lvl < -n: lvl = -n
        if lvl > n - 1: lvl = n - 1
        return out[lvl]

    if isinstance(out, torch.Tensor):
        return out

    raise TypeError(f"Unsupported backbone output type: {type(out)}")

def energy_roi_box(fm_1chw: torch.Tensor, frac=0.18):
    """
    fm_1chw: (C,H,W)   (注意不是 batch)
    在能量图里取 top frac 区域，返回 bbox (x1,y1,x2,y2) in featmap coords
    """
    e = fm_1chw.pow(2).sum(dim=0)  # (H,W)
    flat = e.flatten()
    k = max(1, int(flat.numel() * frac))
    thr = torch.topk(flat, k=k, largest=True).values.min()
    m = (e >= thr)
    ys, xs = torch.where(m)
    if ys.numel() < 10:
        return None
    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1
    return (x1, y1, x2, y2)


@torch.no_grad()
def select_top_patches_with_xy(feat_map_1bchw: torch.Tensor, keep=256, border=0.05, roi_fbox=None):
    """
    feat_map_1bchw: (1,C,H,W)
    return:
      patches: (K,C) L2
      xy:      (K,2) float (x,y)
    """
    fm = feat_map_1bchw[0]  # (C,H,W)
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)  # (H,W)

    # border mask
    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))
    mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
    mask[y1b:y2b, x1b:x2b] = True

    # roi intersect
    if roi_fbox is not None:
        rx1, ry1, rx2, ry2 = roi_fbox
        rx1 = max(0, min(W-1, int(rx1)))
        ry1 = max(0, min(H-1, int(ry1)))
        rx2 = max(1, min(W,   int(rx2)))
        ry2 = max(1, min(H,   int(ry2)))
        if rx2 > rx1 and ry2 > ry1:
            roi_mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
            roi_mask[ry1:ry2, rx1:rx2] = True
            mask = mask & roi_mask

    idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        # fallback to full
        idx_all = torch.arange(H * W, device=energy.device)

    k = min(int(keep), int(idx_all.numel()))
    vals = energy.flatten()[idx_all]
    top_local = torch.topk(vals, k=k, largest=True).indices
    idx = idx_all[top_local]

    patches = fm.flatten(1).t()[idx]        # (k,C)
    patches = F.normalize(patches, p=2, dim=1)

    ys = (idx // W).float()
    xs = (idx % W).float()

    # 归一化到 0~1（避免不同层/不同输入尺寸导致阈值失效）
    xs = xs / max(1.0, float(W - 1))
    ys = ys / max(1.0, float(H - 1))

    xy = torch.stack([xs, ys], dim=1)
    return patches, xy

# 建议做一个全局 session，避免每次 init 很慢
_REMBG_SESSION = None

def _get_rembg_session(model_name: str = "u2net"):
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session(model_name)
    return _REMBG_SESSION

# ============================================================
# Utils
# ============================================================
def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

def rotate_bgr(img, deg):
    if deg == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += (nw/2) - cx
    M[1,2] += (nh/2) - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

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

# 2) 改：geom_score_compatible 的默认参数（最小 diff：只改默认值）
# 你原来 tex_weight=0.35 太低，margin/min_keep偏紧
# =========================
@torch.no_grad()
def geom_score_compatible(q_desc, q_xy, c_desc, c_xy,
                          margin=0.012, min_keep=5,
                          bin_size=0.05, topM=6, topk_core=64,
                          periodic_peak_thr=0.25,
                          periodic_cover_topM_thr=0.70,
                          periodic_cover_xy_thr=0.22,
                          tex_topk_core=128, tex_min_pairs=12,
                          tex_weight=0.85, return_ng0=False):
    """
    返回一个最终的“几何相关”分数（兼容结构/纹理）
    - Ng0>=min_keep：走结构 geom_score_adaptive
    - 否则：走 texture_score（并且 tex_weight 不要太低）
    """
    Ng0 = compute_Ng0(q_desc, c_desc, margin=margin)
    if Ng0 >= min_keep:
        s = geom_score_adaptive(
            q_desc, q_xy, c_desc, c_xy,
            margin=margin, min_keep=min_keep,
            bin_size=bin_size, topM=topM, topk_core=topk_core,
            periodic_peak_thr=periodic_peak_thr,
            periodic_cover_topM_thr=periodic_cover_topM_thr,
            periodic_cover_xy_thr=periodic_cover_xy_thr,
            dbg=False
        )
        return (float(s), Ng0) if return_ng0 else float(s)

    s_tex = texture_score(
        q_desc, q_xy, c_desc, c_xy,
        bin_size=bin_size, topM=topM,
        topk_core=tex_topk_core, min_pairs=tex_min_pairs
    )
    out = float(s_tex * tex_weight)
    return (out, Ng0) if return_ng0 else out

@torch.no_grad()
def texture_score(q_desc, q_xy, c_desc, c_xy,
                  bin_size=0.05, topM=6,
                  topk_core=128, min_pairs=12):
    sim = q_desc @ c_desc.t()
    q_bestv, q_best = torch.max(sim, dim=1)        # (Nq,)

    K = min(int(topk_core), q_desc.shape[0])
    sel = torch.topk(q_bestv, k=K, largest=True).indices
    if K < min_pairs:
        return 0.0

    qg = q_xy[sel]
    cg = c_xy[q_best[sel]]
    core = float(q_bestv[sel].mean().item())

    d = cg - qg
    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(K)
    m = min(int(topM), int(cnt.numel()))
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(K)
    if peak_ratio > 0.85 and cover_topM < 0.35:
        # 高峰但覆盖窄 → 很像“局部强纹理误匹配”
        return 0.0
    if core < 0.45 and peak_ratio > 0.6:
        return 0.0
    # 纹理：更看重 cover_topM（集中在少数峰）
    return float(core * (0.35 + 0.65 * cover_topM) * (0.5 + 0.5 * peak_ratio))


@torch.no_grad()
def geom_score_with_stats(q_desc, q_xy, c_desc, c_xy, **kw):
    margin   = kw.get("margin", 0.015)
    min_keep = kw.get("min_keep", 6)

    sim = q_desc @ c_desc.t()
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0.0, {
            "Ng0": 0, "min_keep": int(min_keep), "margin": float(margin),
            "core_sim_mean": 0.0, "best_gap_mean": 0.0,
        }

    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best  = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng0 = int(good.sum().item())

    score_struct = geom_score_adaptive(q_desc, q_xy, c_desc, c_xy, **kw)

    stats = {
        "Ng0": Ng0,
        "min_keep": int(min_keep),
        "margin": float(margin),
        "core_sim_mean": float(q_bestv.mean().item()),
        "best_gap_mean": float((q_bestv - q_2ndv).mean().item()),
    }
    return score_struct, stats


# ============================================================
# Model
# ============================================================

@torch.no_grad()
def geom_score_adaptive(
    q_desc, q_xy, c_desc, c_xy,
    margin=0.02, min_keep=8,
    bin_size=4.0, topM=6, topk_core=64,
    periodic_peak_thr=0.22,
    periodic_cover_topM_thr=0.70,
    periodic_cover_xy_thr=0.18,
    dbg=False
):
    sim = q_desc @ c_desc.t()  # (Nq, Nc)
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        if dbg:
            print("[DBG] early return: Nc<2 or Nq<1")
        return 0.0

    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best  = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng0 = int(good.sum().item())
    if Ng0 < min_keep:
        if dbg:
            print(f"[DBG] early return: Ng0={Ng0} < min_keep={min_keep} (margin={margin})")
        return 0.0

    good_idx = torch.nonzero(good, as_tuple=False).squeeze(1)

    K = min(topk_core, good_idx.numel())
    sel = torch.topk(q_bestv[good_idx], k=K, largest=True).indices
    good_idx = good_idx[sel]

    mi = q_best[good_idx]
    qg = q_xy[good_idx]
    cg = c_xy[mi]
    Ng = int(good_idx.numel())
    if Ng < min_keep:
        if dbg:
            print(f"[DBG] early return: Ng={Ng} < min_keep={min_keep}")
        return 0.0

    P = min(64, Ng)
    qg2 = qg[:P]
    cg2 = cg[:P]

    ratios = []
    for i in range(P):
        j = (i * 7 + 13) % P
        dq = torch.norm(qg2[i] - qg2[j]) + 1e-6
        if float(dq.item()) < 0.08:
            continue
        dc = torch.norm(cg2[i] - cg2[j]) + 1e-6
        rr = (dc / dq).clamp(0.25, 4.0)
        ratios.append(torch.log(rr))

    if len(ratios) < 8:
        if dbg:
            print(f"[DBG] early return: ratios={len(ratios)} < 8 (Ng={Ng}, P={P})")
        return 0.0

    ratio_std = float(torch.stack(ratios).std().item())

    if ratio_std > 1.2:
        shape_gate = 0.40
    elif ratio_std > 0.8:
        shape_gate = 0.70
    else:
        shape_gate = 1.0

    shape_scale = float(np.exp(-ratio_std / 0.55)) * float(shape_gate)

    d = cg - qg

    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(Ng)
    m = min(topM, cnt.numel())
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(Ng)

    qx = qg[:, 0]
    qy = qg[:, 1]
    cover_x = float((qx.max() - qx.min()).item())
    cover_y = float((qy.max() - qy.min()).item())
    cover_xy = min(cover_x, cover_y)

    core = float(q_bestv[good_idx].mean().item())
    ng_scale = float(min(1.0, Ng / 32.0))

    is_periodic = (
        (peak_ratio < periodic_peak_thr) and
        (cover_topM > periodic_cover_topM_thr) and
        (cover_xy < periodic_cover_xy_thr)
    )

    score = core * peak_ratio * ng_scale * shape_scale
    if is_periodic:
        score *= (0.25 + 0.75 * peak_ratio)

    if dbg:
        print(
            f"[DBG] Ng={Ng} core={core:.3f} peak_ratio={peak_ratio:.3f} "
            f"cover_topM={cover_topM:.3f} cover_xy={cover_xy:.3f} "
            f"ratio_std={ratio_std:.3f} shape_scale={shape_scale:.3f} "
            f"periodic={is_periodic} score={score:.3f}"
        )

    return float(score)

@torch.no_grad()
def build_model(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(DEVICE)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]),
                    dtype=np.float32).reshape(1,1,3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]),
                    dtype=np.float32).reshape(1,1,3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb
@torch.no_grad()
def select_query_patches(q_fm_1bchw: torch.Tensor,
                         keep=KEEP_PATCHES, border=BORDER,
                         roi_frac=Q_ROI_FRAC, roi_ratio=Q_ROI_RATIO):
    """
    Query 用 ROI + 全图混合，避免 query 太局部导致 Ng 不够
    """
    q_roi = energy_roi_box(q_fm_1bchw[0], frac=roi_frac)

    k_roi = int(round(keep * roi_ratio))
    k_full = max(1, keep - k_roi)

    desc_list, xy_list = [], []

    # ROI part
    if q_roi is not None and k_roi >= 4:
        d1, x1 = select_top_patches_with_xy(q_fm_1bchw, keep=k_roi, border=border, roi_fbox=q_roi)
        if d1 is not None and d1.shape[0] >= 4:
            desc_list.append(d1); xy_list.append(x1)

    # Full part (always)
    d2, x2 = select_top_patches_with_xy(q_fm_1bchw, keep=k_full, border=border, roi_fbox=None)
    desc_list.append(d2); xy_list.append(x2)

    q_desc = torch.cat(desc_list, dim=0)
    q_xy   = torch.cat(xy_list, dim=0)
    return q_desc, q_xy


@torch.no_grad()
def select_candidate_patches(c_fm_1bchw: torch.Tensor,
                             keep=KEEP_PATCHES, border=BORDER,
                             roi_frac=C_ROI_FRAC):
    """
    Candidate：优先 ROI（能量集中区域），但 ROI 失败就回退全图
    """
    c_roi = energy_roi_box(c_fm_1bchw[0], frac=roi_frac)

    d, x = select_top_patches_with_xy(c_fm_1bchw, keep=keep, border=border, roi_fbox=c_roi)
    # 兜底：roi 交集为空时 select_top_patches_with_xy 内部已 fallback 到 full
    return d, x

@torch.no_grad()
def extract_feat(model, imgs):
    feat = model.backbone(imgs)
    if isinstance(feat, (tuple, list)):
        feat = feat[-1]
    if feat.dim() == 4:
        feat = feat.mean(dim=(2,3))
    return F.normalize(feat, p=2, dim=1)

# ============================================================
# Simple query embedding (global)
# ============================================================
def get_query_global_feat(model, mean, std, to_rgb, img_bgr):
    img = cv2.resize(img_bgr, (224,224))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) - mean) / std
    x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return extract_feat(model, x).cpu().numpy()
# ============================================================
# Query patch embedding (FIX PATCH SEARCH BUG)
# ============================================================
def _resize_long_edge(img_bgr, long_edge=768):
    h, w = img_bgr.shape[:2]
    s = long_edge / float(max(h, w))
    if s >= 1.0:
        return img_bgr
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def mask_cover(patch_xyxy, qmask):
    x1,y1,x2,y2 = patch_xyxy
    return qmask[y1:y2, x1:x2].mean() / 255.0

def _extract_patches_grid(img_bgr, patch_sizes=(256,384,512), stride_ratio=0.5,
                          max_patches=64, border_frac=0.02, roi_xyxy=None,
                          return_xyxy=False):
    H, W = img_bgr.shape[:2]
    if roi_xyxy is not None:
        rx1, ry1, rx2, ry2 = roi_xyxy
        rx1 = max(0,int(rx1)); ry1=max(0,int(ry1)); rx2=min(W,int(rx2)); ry2=min(H,int(ry2))
    else:
        rx1, ry1, rx2, ry2 = 0,0,W,H

    crop = img_bgr[ry1:ry2, rx1:rx2]
    HH, WW = crop.shape[:2]

    patches = []
    for ps in patch_sizes:
        if min(HH, WW) < ps:
            continue
        stride = max(1, int(ps * stride_ratio))
        y0 = int(HH * border_frac); x0 = int(WW * border_frac)
        y1m = max(y0, HH - int(HH * border_frac) - ps)
        x1m = max(x0, WW - int(WW * border_frac) - ps)

        ys = list(range(y0, y1m + 1, stride)) if y1m >= y0 else [max(0, (HH-ps)//2)]
        xs = list(range(x0, x1m + 1, stride)) if x1m >= x0 else [max(0, (WW-ps)//2)]

        for yy in ys:
            for xx in xs:
                patch = crop[yy:yy+ps, xx:xx+ps]
                if patch.shape[0]==ps and patch.shape[1]==ps:
                    if return_xyxy:
                        x1 = rx1 + xx
                        y1 = ry1 + yy
                        x2 = x1 + ps
                        y2 = y1 + ps
                        patches.append((patch, (x1,y1,x2,y2)))
                    else:
                        patches.append(patch)
                if len(patches) >= max_patches:
                    return patches

    if not patches:
        if return_xyxy:
            patches.append((crop, (rx1, ry1, rx2, ry2)))
        else:
            patches.append(crop)
    return patches


def get_query_patch_feats(model, mean, std, to_rgb, qimg_bgr,
                          patch_sizes=(256, 384, 512),
                          stride_ratio=0.5,
                          max_patches=64,
                          long_edge=1024,
                          batch_size=64,
                          qmask=None,
                          min_mask_cover=0.35):
    qimg = _resize_long_edge(qimg_bgr, long_edge=long_edge)

    # 如果传了 qmask，也同步 resize 到 qimg 尺寸
    qmask_rs = None
    if qmask is not None:
        qmask_rs = cv2.resize(qmask, (qimg.shape[1], qimg.shape[0]), interpolation=cv2.INTER_NEAREST)

    qx512 = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    qfm = extract_featmap(model, qx512, FEAT_LEVEL)
    roi_f = energy_roi_box(qfm[0], frac=0.18)

    if roi_f is not None:
        _, _, Hf, Wf = qfm.shape
        x1, y1, x2, y2 = roi_f
        x1 = x1 / (Wf - 1); x2 = x2 / (Wf - 1)
        y1 = y1 / (Hf - 1); y2 = y2 / (Hf - 1)
        H, W = qimg.shape[:2]
        roi_xyxy = (x1 * W, y1 * H, x2 * W, y2 * H)
    else:
        roi_xyxy = None

    patches_with_xy = _extract_patches_grid(
        qimg,
        patch_sizes=patch_sizes,
        stride_ratio=stride_ratio,
        max_patches=max_patches,
        roi_xyxy=roi_xyxy,
        return_xyxy=True
    )

    # ✅ 过滤：mask 覆盖率太低的 patch 丢掉
    filtered = []
    for patch, xyxy in patches_with_xy:
        if qmask_rs is not None:
            x1,y1,x2,y2 = xyxy
            x1 = max(0, min(qmask_rs.shape[1]-1, int(x1)))
            y1 = max(0, min(qmask_rs.shape[0]-1, int(y1)))
            x2 = max(1, min(qmask_rs.shape[1],   int(x2)))
            y2 = max(1, min(qmask_rs.shape[0],   int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            cover = float(qmask_rs[y1:y2, x1:x2].mean()) / 255.0
            if cover < float(min_mask_cover):
                continue
        filtered.append(patch)

    if not filtered:
        # 兜底：别一个都没有
        filtered = [p for p, _ in patches_with_xy[:1]]

    tensors = []
    for p in filtered:
        p224 = cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA)
        if to_rgb:
            p224 = cv2.cvtColor(p224, cv2.COLOR_BGR2RGB)
        x = (p224.astype(np.float32) - mean) / std
        x = torch.from_numpy(x.transpose(2, 0, 1))
        tensors.append(x)

    feats_all = []
    with torch.no_grad():
        for st in range(0, len(tensors), batch_size):
            bt = torch.stack(tensors[st:st+batch_size], dim=0).to(DEVICE)
            fv = extract_feat(model, bt)
            feats_all.append(fv.cpu())

    feats = torch.cat(feats_all, dim=0).numpy().astype("float32")
    return feats, len(filtered)


# ============================================================
# Patch aggregation → image score
# ============================================================
def aggregate_patch_hits(patch_ids, patch_scores, patch_meta,
                         top_images=4000, topM=8, tau=0.15):
    """
    同图多patch累计：对每张图取 topM 个 patch 分数做 soft-sum
    tau 越小越接近 max；越大越接近 sum
    """
    meta_is_2d = isinstance(patch_meta, np.ndarray) and patch_meta.ndim == 2

    img_scores = {}
    for pid, s in zip(patch_ids, patch_scores):
        if pid < 0:
            continue
        img_id = int(patch_meta[pid, 0]) if meta_is_2d else int(patch_meta[pid][0])
        img_scores.setdefault(img_id, []).append(float(s))

    fused = []
    for img_id, ss in img_scores.items():
        ss.sort(reverse=True)
        ss = ss[:topM]
        # softmax-like sum (数值稳定/抗偶然max)
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        score = m + max(tau, 1e-6) * np.log(v + 1e-9)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]

def rank_to_rrf_score(rank_list, k=60):
    s = {}
    for r, i in enumerate(rank_list, 1):
        s[i] = 1.0 / (k + r)
    return s


def faiss_scores_from_D(index, D: np.ndarray) -> np.ndarray:
    # 返回“越大越好”的 score
    try:
        mt = index.metric_type
    except Exception:
        mt = None

    # faiss.METRIC_L2 == 1, faiss.METRIC_INNER_PRODUCT == 0（不同版本也可能有 enum）
    if mt == faiss.METRIC_L2 or mt == 1:
        D = D.astype(np.float32, copy=False)
        return np.float32(1.0) / (np.float32(1.0) + D)
    else:
        # inner product 本来就是越大越好
        return D

def rrf_fuse(rankA, rankB, k=60):
    score = {}
    for r,i in enumerate(rankA,1):
        score[i] = score.get(i,0)+1/(k+r)
    for r,i in enumerate(rankB,1):
        score[i] = score.get(i,0)+1/(k+r)
    return [i for i,_ in sorted(score.items(), key=lambda x:x[1], reverse=True)]


# ============================================================
# Geometry rerank (ORB + RANSAC)
# ============================================================
orb = cv2.ORB_create(2000)

def geom_score(qimg, cimg):
    gq = cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    kq, dq = orb.detectAndCompute(gq, None)
    kc, dc = orb.detectAndCompute(gc, None)
    if dq is None or dc is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(dq, dc, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_INLIERS:
        return 0.0

    pts_q = np.float32([kq[m.queryIdx].pt for m in good])
    pts_c = np.float32([kc[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_q, pts_c, cv2.RANSAC, RANSAC_THRESH)
    if mask is None:
        return 0.0

    inliers = int(mask.sum())
    return inliers / (len(good) + 1e-6)


# ============================================================
# Visualization (Grid)
# ============================================================
def _put_text(img, text, org=(8, 26), font_scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _fit_square(img_bgr, tile=320):
    """等比缩放到 tile 内 + 居中 pad 到 tile*tile"""
    h, w = img_bgr.shape[:2]
    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

@torch.no_grad()
def compute_Ng0(q_desc, c_desc, margin=0.015):
    sim = q_desc @ c_desc.t()                      # (Nq, Nc)
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0
    topv, topi = torch.topk(sim, k=2, dim=1)       # q->c top1/top2
    q_best  = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)              # c->q top1
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    return int(good.sum().item())

# 1) 计算一个“global置信度”：top聚集度（用top20里同一前缀/同一组的集中度替代也行）
def global_confidence(global_rank, img_paths, topn=20):
    names = [os.path.basename(str(img_paths[i])) for i in global_rank[:topn]]
    # 例：按 IMG_483x 这种系列聚集（你可以换成更通用的：相似度gap/熵）
    prefix = [n.split('_')[0] if '_' in n else n[:4] for n in names]
    # 计算最常见prefix占比
    from collections import Counter
    c = Counter(prefix).most_common(1)[0][1]
    return c / max(1, len(prefix))
def visualize_grid(query_bgr, top_imgs_bgr, top_scores, out_path,
                   tile=320, gap=10, header=44):
    """
    query_bgr: BGR
    top_imgs_bgr: list of BGR
    top_scores: list of float
    out_path: output png/jpg
    """
    # 1) prepare tiles
    tiles = [_fit_square(query_bgr, tile)]
    labels = ["QUERY"]

    for i, (img, s) in enumerate(zip(top_imgs_bgr, top_scores), 1):
        if img is None:
            img = np.zeros((tile, tile, 3), np.uint8)
        tiles.append(_fit_square(img, tile))
        labels.append(f"#{i}  {float(s):.3f}")

    # 2) grid size
    n = len(tiles)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # 3) paste
    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)

        canvas[y:y+header, x:x+tile] = 0  # header bg
        _put_text(canvas, labels[idx], org=(x + 8, y + 28), font_scale=0.7, thickness=2)
        canvas[y+header:y+header+tile, x:x+tile] = tiles[idx]

    cv2.imwrite(out_path, canvas)
    return out_path

def clean_rank(rank_list):
    seen = set()
    out = []
    for x in rank_list:
        if x is None:
            continue
        x = int(x)
        if x < 0:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# 1) 改：cand_gate_score -> 连续 gate（不再依赖 q_is）
# 最小 diff：保持函数名不变，但增加 q_head 传入；main 里改调用
# =========================
def cand_gate_score(cimg_bgr, q_head: dict):
    ch = stripe_grid_head_v21(make_head_view(cimg_bgr, prefer_gray=False))

    # peakedness 降权：避免“吊牌/边缘”把 sim 抬高
    wP = 0.25
    qs = np.array([q_head["stripe_score"], q_head["grid_score"],
                   wP * np.clip(q_head["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)
    cs = np.array([ch["stripe_score"], ch["grid_score"],
                   wP * np.clip(ch["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)

    den = float(np.linalg.norm(qs) * np.linalg.norm(cs) + 1e-6)
    sim = float((qs * cs).sum() / den)
    sim = max(0.0, min(1.0, sim))

    # sim 阈值可以略抬高一点（你现在 0.18 很松）
    if sim < 0.28:
        return 0.0

    # “类型一致性”这两条保留
    if q_head["grid_score"] > 0.15 and ch["grid_score"] < 0.06:
        return 0.0
    if q_head["stripe_score"] > 0.15 and ch["stripe_score"] < 0.06:
        return 0.0

    return float(0.60 + 0.55 * sim)

# ============================================================
# Main
# ============================================================
def main():
    print("[PID]", os.getpid())
    ensure_dir(OUT_DIR)

    g_index = faiss.read_index(GLOBAL_INDEX)
    p_index = faiss.read_index(PATCH_INDEX)
    set_faiss_nprobe(g_index, 64)
    set_faiss_nprobe(p_index, 64)

    img_paths = np.load(GLOBAL_META, allow_pickle=True)
    patch_meta = np.load(PATCH_META, allow_pickle=True)

    model, mean, std, to_rgb = build_model(CONFIG, CKPT)

    qimg = imread_unicode(QUERY_IMG)

    qimg, qmask = crop_by_mmdet_mask_final(
        qimg,
        score_thr=SEG_SCORE_THR,
        use_classes=SEG_USE_CLASSES,
        merge_all=True,
        do_rectify=True,
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=OUT_DIR
    )

    # # 调试用：看裁剪后的 query 到底长啥样
    cv2.imwrite(os.path.join(OUT_DIR, "dbg_crop.png"), qimg)

    q_head_img = make_head_view(qimg, prefer_gray=True)
    q_head = stripe_grid_head_v21(q_head_img)
    is_stripe_like = (q_head["stripe_score"] > 0.12)
    is_grid_like = (q_head["grid_score"] > 0.08)
    q_is = (is_stripe_like or is_grid_like)

    print(
        f"[HEADv2.1] stripe={q_head['stripe_score']:.3f} "
        f"grid={q_head['grid_score']:.3f} peaked={q_head['ori_peakedness']:.2f}"
    )
    qvec = get_query_global_feat(model, mean, std, to_rgb, qimg)

    # -------- Global search
    _, gids = g_index.search(qvec, TOPG)
    # -------- Patch search (FIXED)
    q_patch_vecs, n_qpatch = get_query_patch_feats(
        model, mean, std, to_rgb, qimg,
        patch_sizes=(256, 384, 512),
        stride_ratio=0.5,
        max_patches=64,
        long_edge=1024,
        batch_size=64,
        qmask=qmask,  # ✅ 新增
        min_mask_cover=0.22  # ✅ 可调：0.25~0.45
    )

    print(f"[PATCH] query patches = {n_qpatch}, vecs shape = {q_patch_vecs.shape}")

    def show_ids(ids, img_paths, title, n=15):
        print(f"\n== {title} ==")
        for k, i in enumerate(ids[:n], 1):
            p = str(img_paths[i])
            print(f"{k:02d}  id={i:<5d}  name={os.path.basename(p)}  path={p}")
    # 对每个 query patch 去搜 patch index
    patch_ids_all, patch_scores_all = [], []
    D, I = p_index.search(q_patch_vecs, PATCH_TOPK_PER_QPATCH)  # D/I: (P, K)
    # D: (P,K) 对每个 query patch 内部做归一化，避免某个 patch 分值尺度异常
    D = D.astype(np.float32)
    S = faiss_scores_from_D(p_index, D)

    # 聚合所有 patch hit
    patch_ids_all = I.reshape(-1).tolist()
    patch_scores_all = S.reshape(-1).tolist()

    patch_rank = aggregate_patch_hits(
        patch_ids_all, patch_scores_all, patch_meta, top_images=TOP_PATCH_IMAGES,tau=0.15
    )
    global_rank = clean_rank(gids[0].tolist())
    patch_rank = clean_rank(patch_rank)
    print("[PATCH] top patch-rank ids:", patch_rank[:10])
    print("[GLOBAL] top global-rank ids:", global_rank[:10])

    # -------- RRF fusion
    fused = rrf_fuse(global_rank, patch_rank, RRF_K)
    fused = clean_rank(fused)[:GEOM_TOPN]
    rrf_g = rank_to_rrf_score(global_rank, k=RRF_K)
    rrf_p = rank_to_rrf_score(patch_rank, k=RRF_K)
    conf_g = global_confidence(global_rank, img_paths, topn=20)
    # conf_g 越大，越信global
    w_g = 0.6 + 0.35 * conf_g  # 大概落在[0.6, 0.95]
    w_p = 1.0 - w_g
    if is_stripe_like or is_grid_like:
        # 强制偏向 patch_rank（条纹库）
        w_g = 0.20
        w_p = 0.80
    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    q_desc_list = []
    q_xy_list = []

    for ang in [-30, -15, 0, 15, 30]:
        qimg_r = rotate_bgr(qimg, ang)
        qx = make_single_tensor_for_rerank(qimg_r, mean, std, to_rgb=to_rgb).to(DEVICE)
        q_fm = extract_featmap(model, qx, FEAT_LEVEL)
        q_desc, q_xy = select_query_patches(q_fm)
        q_desc_list.append(q_desc)
        q_xy_list.append(q_xy)

    show_ids(global_rank, img_paths, "GLOBAL top")
    show_ids(patch_rank, img_paths, "PATCH  top")
    show_ids(fused, img_paths, "FUSED  top")
    cimg_cache = {}

    scored = []
    fused2 = []
    for img_id in fused:
        cimg0 = imread_unicode(img_paths[img_id])
        if cimg0 is None or cimg0.size == 0:
            continue
        g0 = cand_gate_score(cimg0, q_head)
        if g0 <= 0.0:
            continue
        fused2.append(img_id)
    print("[FUSED] before:", len(fused), "after head gate:", len(fused2))

    fused = fused2
    for img_id in fused:
        cimg = imread_unicode(img_paths[img_id])
        if cimg is None or cimg.size == 0:
            continue
        # ✅ candidate 不分割：直接缓存原图
        cimg_cache[img_id] = cimg

        cx = make_single_tensor_for_rerank(cimg, mean, std, to_rgb=to_rgb).to(DEVICE)
        c_fm = extract_featmap(model, cx, FEAT_LEVEL)
        c_desc, c_xy = select_candidate_patches(c_fm)

        geom_best = 0.0
        cnt = 0
        for q_desc, q_xy in zip(q_desc_list, q_xy_list):
            s, ng0= geom_score_compatible(
                q_desc, q_xy, c_desc, c_xy,
                margin=0.012, min_keep=5,  # ✅ 放宽 Ng0 通道
                bin_size=BIN_SIZE, topM=TOPM, topk_core=TOPK_CORE,
                periodic_peak_thr=PERIODIC_PEAK_THR,
                periodic_cover_topM_thr=PERIODIC_COVER_TOPM_THR,
                periodic_cover_xy_thr=PERIODIC_COVER_XY_THR,
                tex_topk_core=128, tex_min_pairs=12,
                tex_weight=0.85 , # ✅ 提高 texture 权重,
                return_ng0=True
            )
            if q_head["grid_score"] > 0.10 and ng0 < 5:
                s = 0.0
            if s > 0:
                geom_best += s
                cnt += 1
        geom_best = geom_best / max(1, cnt)
        scored.append((img_id, geom_best))
    scored = [(i, s) for (i, s) in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    geom_vals = [s for _, s in scored]
    gmax = max(geom_vals) if geom_vals else 1.0
    gmin = min(geom_vals) if geom_vals else 0.0

    def norm_g(g):
        return (g - gmin) / (gmax - gmin + 1e-9)

    beta = 0.15  # 0.1~0.5 之间先试

    final = []
    for img_id, gs in scored:
        a = 0.35
        b = 0.20
        boost = 1.0 + a * q_head["grid_score"] + b * q_head["stripe_score"]
        fs = final_rrf.get(img_id, 0.0) * (1.0 + beta * norm_g(gs)) * boost
        final.append((img_id, fs, gs))

    final2 = []
    for img_id, fs, gs in final:
        cimg = cimg_cache.get(img_id, None)
        if cimg is None:
            cimg = imread_unicode(img_paths[img_id])
            if cimg is None or cimg.size == 0:
                continue
            cimg_cache[img_id] = cimg

        g = cand_gate_score(cimg, q_head)
        final2.append((img_id, fs * g, gs))

    final2.sort(key=lambda x: x[1], reverse=True)
    final = final2

    scores = [fs for _, fs, _ in final]
    s_min = min(scores)
    s_max = max(scores) + 1e-9

    final_norm = []
    for img_id, fs, gs in final:
        fs_norm = (fs - s_min) / (s_max - s_min)  # 0~1
        final_norm.append((img_id, fs_norm, gs))

    final_norm.sort(key=lambda x: x[1], reverse=True)
    top = final_norm[:TOPK]

    # top = final[:TOPK]

    for r, (img_id, fs, gs) in enumerate(top, 1):
        print(f"{r:02d}  final={fs:.4f}  geom={gs:.4f}  {img_paths[img_id]}")

    # -------- Visualize
    imgs = []
    scores = []
    for img_id, fs, gs in top:
        imgs.append(imread_unicode(img_paths[img_id]))
        scores.append(fs)

    out = os.path.join(OUT_DIR, "result_grid1.png")
    visualize_grid(qimg, imgs, scores, out, tile=320)

    print("Top results:")
    for r,(img_id, fs, gs) in enumerate(top,1):
        print(f"{r:02d}  score={fs:.3f}  {img_paths[img_id]}")
    print("Saved:", out)

if __name__ == "__main__":
    main()
