#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zhanlan Hybrid Retrieval Pipeline (Global + Patch + RRF + HeadGate + GeomRerank)
Refactored layout: group functions by responsibility.
NOTE: Logic kept as-is; only re-ordered and de-duplicated where safe.
"""

# ============================================================
# 0. Imports & Global Runtime Settings
# ============================================================
import os
import json
import time
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import random
import os
from matplotlib import pyplot as plt
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS
from rembg import remove, new_session
from pycocotools import mask as maskUtils
from torch.cuda.amp import autocast
from ultralytics import YOLO
from hybrid_shared import gen_patch_windows_unified, patch_to_model_input, STRIPE_LONG_EDGE
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x 对 matmul 更快

# mp
mp.set_start_method("spawn", force=True)

# ============================================================
# 1. Global Config / Constants
# ============================================================
# ---------- device ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEG_DEVICE = "cuda:0"  # or "cpu"
YOLO_DEVICE = 0        # 0 / "cpu" / "cuda:0" (ultralytics typical)

# ---------- segmentation ----------
SEG_SCORE_THR = 0.6
SEG_USE_CLASSES = None

# ---------- model ----------
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

# ---------- YOLO seg ----------
YOLO_SEG_WEIGHTS = r"D:\zhanlanProject\ultralyticsV8\runs\huaxing\exp12\weights\best.pt"
_YOLO_SEG = None

# ---------- query / index ----------
QUERY_IMG = r"D:\zhanlan\segment_data\花色随机拍摄照片\IMG_20260129_114047(1).jpg"

INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid_new_data"
GLOBAL_INDEX = os.path.join(INDEX_DIR, "global.index")

GLOBAL_META  = os.path.join(INDEX_DIR, "global_img_paths.npy")


PATCH_STRIPE_INDEX = os.path.join(INDEX_DIR, "patch_stripe.index")
PATCH_GRID_INDEX   = os.path.join(INDEX_DIR, "patch_grid.index")

PATCH_STRIPE_META  = os.path.join(INDEX_DIR, "patch_stripe_meta.npy")
PATCH_GRID_META    = os.path.join(INDEX_DIR, "patch_grid_meta.npy")

OUT_DIR = r"D:\zhanlan\search_vis"
TOPK = 12


# =========================
# STRIPE_SHARED_CONSTANTS
# =========================
STRIPE_AR_THR = 2.7

STRIPE_WIN_H = 224
STRIPE_STRIDE = 48
STRIPE_MAX_PATCHES = 24
STRIPE_CENTER_FRAC = 0.92
STRIPE_JITTER = 8        # 关键：和建库一致
STRIPE_WIN_W_FRAC = 0.85
STRIPE_WIN_W_MIN = 160
STRIPE_WIN_W_MAX = 256
STRIPE_SEED = 123


# ---------- retrieval ----------
TOPG = 2000
PATCH_TOPK_PER_QPATCH = 800
TOP_PATCH_IMAGES = 4000
RRF_K = 60

# ---------- geom rerank (ORB RANSAC) ----------
GEOM_TOPN = 120
MIN_INLIERS = 8
RANSAC_THRESH = 5.0

VIEWS_PER_IMAGE = 12
RESIZE_SHORT = 256
CROP_SIZE = 224
VIEW_PLAN = [
    (0,   1, 5),
    (-15, 1, 1),
    (15,  1, 1),
    (-30, 1, 0),
    (30,  1, 0),
]
VIEW_BATCH = 256  # batch in "views"

# ---------- patch rerank / feature-map patches ----------
FEAT_LEVEL = -2
RMAC_INPUT_SIZE = 512
KEEP_PATCHES = 256
BORDER = 0.05

# ROI selection
Q_ROI_FRAC  = 0.18
C_ROI_FRAC  = 0.18
Q_ROI_RATIO = 0.50
C_ROI_RATIO = 1.00

# Geom score / periodic
MARGIN = 0.015
MIN_KEEP = 6
BIN_SIZE = 0.05
TOPK_CORE = 64

PERIODIC_PEAK_THR  = 0.25
PERIODIC_COVER_THR = 0.22
PERIODIC_COVER_TOPM_THR = 0.70
PERIODIC_COVER_XY_THR   = 0.22
TOPM = 6


# ============================================================
# 2. Debug / Logging Utilities
# ============================================================

# ============================================================
# 3. Basic IO / Geometry Utils
# ============================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

def rotate_bgr(img, deg):
    if deg == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

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

def _safe_pad_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(1, min(W, int(x2)))
    y2 = max(1, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def _largest_cc(mask_u8: np.ndarray):
    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = 1 + int(np.argmax(areas))
    return (labels_cc == best_i).astype(np.uint8) * 255


# ============================================================
# 4. YOLO Seg / Crop Pipeline
# ============================================================
def get_yolo_seg():
    global _YOLO_SEG
    if _YOLO_SEG is None:
        _YOLO_SEG = YOLO(YOLO_SEG_WEIGHTS)
    return _YOLO_SEG

def _fill_background(crop_bgr, crop_mask_u8, mode="mean"):
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
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out[~m] = blur[~m]
        return out

    mean_color = out[m].mean(axis=0)
    out[~m] = mean_color
    return out

def _yolo_extract_mask_u8(result, H, W, conf_thr=0.6, use_classes=None, merge_all=True):
    if result is None:
        return None

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if masks is None or boxes is None or masks.data is None or len(masks.data) == 0:
        return None

    m = masks.data
    conf = boxes.conf if boxes.conf is not None else torch.ones((m.shape[0],), device=m.device)
    cls = boxes.cls

    keep = conf >= float(conf_thr)
    if use_classes is not None and cls is not None:
        use = torch.tensor(use_classes, device=m.device, dtype=cls.dtype)
        keep = keep & torch.isin(cls, use)

    idx = torch.where(keep)[0]
    if idx.numel() == 0:
        return None

    m_keep = m[idx]
    if not merge_all:
        best_local = torch.argmax(conf[idx]).item()
        mm = m_keep[best_local]
    else:
        mm = torch.any(m_keep > 0.5, dim=0)

    mm = mm.float().unsqueeze(0).unsqueeze(0)
    mm = torch.nn.functional.interpolate(mm, size=(H, W), mode="nearest")
    mm = mm[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    return mm

def crop_by_mmdet_mask_final(
        img_bgr: np.ndarray,
        pad: int = 10,
        min_area_frac: float = 0.06,
        score_thr: float = 0.6,
        use_classes=None,
        merge_all: bool = True,
        do_rectify: bool = True,
        rectify_pad: int = 10,
        warp_border: str = "reflect",   # "reflect" | "replicate"
        bg_mode: str = "mean",          # "mean" | "white" | "edge"
        debug_dir: str = None,
        yolo_imgsz: int = 640,
        yolo_iou: float = 0.5,
        yolo_retina_masks: bool = True,
):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, None, None

    H, W = img_bgr.shape[:2]
    default_mask = np.ones((H, W), dtype=np.uint8) * 255
    default_raw = img_bgr

    # 1) YOLO seg
    model = get_yolo_seg()
    results = model.predict(
        source=img_bgr,
        conf=float(score_thr),
        iou=float(yolo_iou),
        imgsz=int(yolo_imgsz),
        device=YOLO_DEVICE,
        max_det=100,
        retina_masks=bool(yolo_retina_masks),
        classes=use_classes,
        verbose=False,
        stream=False,
        save=False,
        show=False
    )
    if results is None or len(results) == 0:
        return img_bgr, default_mask, default_raw

    r0 = results[0]
    mask_u8 = _yolo_extract_mask_u8(r0, H, W, conf_thr=float(score_thr),
                                    use_classes=use_classes, merge_all=merge_all)
    if mask_u8 is None:
        return img_bgr, default_mask, default_raw

    # 2) cleanup mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, ker, iterations=1)

    mask_u8 = _largest_cc(mask_u8)
    if mask_u8 is None:
        return img_bgr, default_mask, default_raw

    area = float((mask_u8 > 0).sum())
    if area < float(min_area_frac) * (H * W):
        return img_bgr, default_mask, default_raw

    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 20:
        return img_bgr, default_mask, default_raw

    # 3) rectify + crop
    if do_rectify:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang = rect
        if rw < rh:
            ang = ang + 90.0

        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        borderMode = cv2.BORDER_REFLECT101 if warp_border == "reflect" else cv2.BORDER_REPLICATE

        rot_img = cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=borderMode)
        rot_msk = cv2.warpAffine(mask_u8, M, (W, H), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        ys2, xs2 = np.where(rot_msk > 0)
        if ys2.size < 20:
            return img_bgr, default_mask, default_raw

        x1, x2 = xs2.min() - rectify_pad, xs2.max() + 1 + rectify_pad
        y1, y2 = ys2.min() - rectify_pad, ys2.max() + 1 + rectify_pad
        crop_img = _safe_pad_crop(rot_img, x1, y1, x2, y2)
        crop_msk = _safe_pad_crop(rot_msk, x1, y1, x2, y2)
    else:
        x1, x2 = xs.min() - pad, xs.max() + 1 + pad
        y1, y2 = ys.min() - pad, ys.max() + 1 + pad
        bw = (x2 - x1)
        bh = (y2 - y1)
        ar = max(bh / (bw + 1e-9), bw / (bh + 1e-9))
        if ar > 8.0:  # ✅ 只有非常极端才修正
            x1, y1, x2, y2 = _expand_bbox_to_limit_ar(x1, y1, x2, y2, H, W, max_ar=8.0)

        crop_img = _safe_pad_crop(img_bgr, x1, y1, x2, y2)
        crop_msk = _safe_pad_crop(mask_u8, x1, y1, x2, y2)

    if crop_img is None or crop_msk is None or crop_img.size == 0:
        return img_bgr, default_mask, default_raw

    # 4) fill bg
    crop_msk_u8 = (crop_msk > 0).astype(np.uint8) * 255
    crop_msk_bool = crop_msk_u8.astype(bool)

    filled = _fill_background(crop_img, crop_msk_u8, mode=bg_mode)
    out = filled.copy()
    out[crop_msk_bool] = crop_img[crop_msk_bool]

    tag = f"q_{int(time.time())}"
    if debug_dir is not None:
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_mask.png"), mask_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_mask.png"), crop_msk_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_raw.png"), crop_img)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_out.png"), out)

    return out, crop_msk_u8, crop_img


# ============================================================
# 5. Stripe/Grid Head Features (FFT + Orientation Histogram)
# ============================================================
def aggregate_patch_hits_stripe(
        patch_ids, patch_scores, patch_meta,
        top_images=4000,
        topM=10,
        tau=0.15,
        pos_bin=500,          # 10000/500=20 bins
        min_cover_bins=3,     # 覆盖太少的直接惩罚
        w_cover=0.12,         # 覆盖加成权重
        w_cont=0.10,          # 连续性加成权重
        only_ptype=1          # 只用 stripe patch
):
    meta = patch_meta
    assert isinstance(meta, np.ndarray) and meta.ndim == 2 and meta.shape[1] >= 8

    img_hits = {}   # img_id -> list of (score, pos_bin)
    for pid, s in zip(patch_ids, patch_scores):
        if pid < 0:
            continue
        pid = int(pid)
        img_id = int(meta[pid, 0])
        ptype = int(meta[pid, 6])
        if only_ptype is not None and ptype != int(only_ptype):
            continue
        pos = int(meta[pid, 7])
        b = int(pos // pos_bin)
        img_hits.setdefault(img_id, []).append((float(s), b))

    fused = []
    for img_id, sb in img_hits.items():
        # 1) similarity 融合（保留你原来的 softmax 思路）
        sb.sort(key=lambda x: x[0], reverse=True)
        sb = sb[:topM]
        ss = [x[0] for x in sb]
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        base = m + max(tau, 1e-6) * np.log(v + 1e-9)

        # 2) 位置覆盖：命中 bin 的数量 / 总 bin（上限 1）
        bins = [x[1] for x in sb]
        ub = sorted(set(bins))
        cover = min(1.0, len(ub) / 8.0)  # 经验：topM=10 时 8 个 bin 覆盖已经很不错了

        # 3) 连续性：最长连续 bin 段 / 覆盖 bin 数
        longest = 1
        cur = 1
        for i in range(1, len(ub)):
            if ub[i] == ub[i-1] + 1:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 1
        cont = longest / max(1, len(ub))

        # 覆盖太少，明显是“蹭到一点纹理”的，惩罚
        if len(ub) < min_cover_bins:
            base *= 0.72

        score = base * (1.0 + w_cover * cover + w_cont * cont)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]

def make_head_view(img_bgr: np.ndarray, prefer_gray=True):
    """head 检测用的“干净图”：裁中间区域减少干扰"""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    H, W = img_bgr.shape[:2]
    y1 = int(H * 0.15); y2 = int(H * 0.85)
    x1 = int(W * 0.10); x2 = int(W * 0.90)
    return img_bgr[y1:y2, x1:x2].copy()

def fft_peak_mass(gray, resize=512, topk_frac=0.002):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32); g -= g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    H, W = mag.shape; cy, cx = H//2, W//2
    r0 = int(min(H, W) * 0.03)
    mag[cy-r0:cy+r0+1, cx-r0:cx+r0+1] = 0

    flat = mag.reshape(-1)
    K = int(topk_frac * flat.size)
    K = max(50, min(K, 4000))
    topk = np.partition(flat, -K)[-K:]
    return float(topk.sum() / (flat.sum() + 1e-6))

def fft_peak_radius(gray, resize=512):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32); gray -= gray.mean()
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    H, W = mag.shape
    cy, cx = H//2, W//2
    r0 = int(min(H, W) * 0.03)
    mag[cy-r0:cy+r0+1, cx-r0:cx+r0+1] = 0
    y, x = np.unravel_index(np.argmax(mag), mag.shape)
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    r_norm = r / (0.5 * min(H, W) + 1e-6)
    return float(r_norm)

def fft_stripe_grid_head(gray, resize=512):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32)
    gray -= gray.mean()
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))

    H, W = mag.shape
    cy, cx = H//2, W//2
    r = int(min(H, W) * 0.03)
    mag[cy-r:cy+r+1, cx-r:cx+r+1] = 0

    band = int(min(H, W) * 0.04)
    horiz = mag[cy-band:cy+band+1, :].mean()
    vert  = mag[:, cx-band:cx+band+1].mean()
    allm  = mag.mean() + 1e-6

    stripe = float(max(horiz, vert) / allm)
    grid   = float(min(horiz, vert) / allm)

    stripe_score = float(np.clip((stripe - 1.05) / 0.6, 0, 1))
    grid_score   = float(np.clip((grid   - 1.02) / 0.6, 0, 1))
    return {"stripe_score": stripe_score, "grid_score": grid_score}

def stripe_grid_head_v21(img_bgr, resize_long=512, nbins=36):
    if img_bgr is None or img_bgr.size == 0:
        return {"stripe_score": 0.0, "grid_score": 0.0, "ori_peakedness": 0.0}

    h, w = img_bgr.shape[:2]
    s = resize_long / float(max(h, w))
    img = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img_bgr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)

    thr = np.percentile(mag, 50)
    mask = mag > thr
    if mask.sum() < 200:
        return {"stripe_score": 0.0, "grid_score": 0.0, "ori_peakedness": 0.0}

    mag2 = mag[mask]
    ang2 = ang[mask]

    hist, _ = np.histogram(ang2, bins=nbins, range=(0, np.pi), weights=mag2)
    hist = hist.astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 5), 0).ravel()

    eps = 1e-6
    p = hist / (hist.sum() + eps)

    bin_angles = (np.arange(nbins) + 0.5) / nbins * np.pi
    d0 = np.minimum(np.abs(bin_angles - 0), np.pi - np.abs(bin_angles - 0))
    d90 = np.abs(bin_angles - (np.pi / 2))
    d = np.minimum(d0, d90)
    w = np.exp(-(d ** 2) / (2 * (0.18 ** 2))).astype(np.float32)
    axis_align = float((p * w).sum())

    peaked = float(p.max() / (p.mean() + eps))

    k1 = int(np.argmax(p))
    ban = max(1, nbins // 18)
    p2 = p.copy()
    p2[max(0, k1-ban):min(nbins, k1+ban+1)] = 0
    k2 = int(np.argmax(p2))

    peak1 = float(p[k1])
    peak2 = float(p[k2])

    a1 = k1 / nbins * np.pi
    a2 = k2 / nbins * np.pi
    dd = abs(a1 - a2)
    dd = min(dd, np.pi - dd)

    ortho = np.exp(-((dd - (np.pi/2))**2) / (2*(0.40**2)))

    stripe_score = peak1 * (peaked / 6.0)
    stripe_score = float(np.clip(stripe_score, 0.0, 1.0))

    grid_score = (0.65*peak1 + 0.35*peak2) * ortho * (peaked / 6.0)
    grid_score = float(np.clip(grid_score, 0.0, 1.0))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r_peak = fft_peak_radius(gray2, resize=512)
    fft_h = fft_stripe_grid_head(gray2)
    peak_mass = fft_peak_mass(gray2, resize=512)

    stripe_score = max(stripe_score, fft_h["stripe_score"])
    grid_score = max(grid_score, fft_h["grid_score"])

    mag_full = np.sqrt(gx * gx + gy * gy)
    thr2 = np.percentile(mag_full, 80)
    edge_density = float((mag_full > thr2).mean())
    if edge_density < 0.035:
        stripe_score *= 0.3
        grid_score *= 0.3

    return {
        "stripe_score": stripe_score,
        "grid_score": grid_score,
        "ori_peakedness": peaked,
        "axis_align": axis_align,
        "r_peak": r_peak,
        "peak1": peak1,
        "peak2": peak2,
        "ortho": float(ortho),
        "peak_mass": peak_mass
    }

def is_grid_like(h: dict):
    g  = h.get("grid_score", 0.0)
    ax = h.get("axis_align", 0.0)
    pk = h.get("ori_peakedness", 0.0)
    ort= h.get("ortho", 0.0)
    p2 = h.get("peak2", 0.0)
    pm = h.get("peak_mass", 0.0)

    if g >= 0.22 and pm > 0.010:
        return True
    if g < 0.10:
        return False
    if ort < 0.25 or p2 < 0.015:
        return False
    if pk < 2.5:
        return (ax > 0.22) and (pm > 0.008)
    return True


# ============================================================
# 6. Model Build / Feature Extraction
# ============================================================
@torch.no_grad()
def extract_backbone_last(model, batch_tensor: torch.Tensor):
    feat_map = model.backbone(batch_tensor)

    if isinstance(feat_map, dict):
        if 'feat' in feat_map:
            feat_map = feat_map['feat']
        elif 'features' in feat_map:
            feat_map = feat_map['features']
        else:
            feat_map = list(feat_map.values())[-1]

    if isinstance(feat_map, (tuple, list)):
        feat_map = feat_map[-1]

    if feat_map.dim() == 2:
        feat = feat_map
    else:
        feat = feat_map.mean(dim=(2, 3))

    feat = F.normalize(feat, p=2, dim=1)
    return feat
def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))

def resize_long_edge(img_bgr: np.ndarray, max_long: int):
    h, w = img_bgr.shape[:2]
    s = max_long / max(h, w)
    if s >= 1.0:
        return img_bgr
    nh, nw = int(round(h*s)), int(round(w*s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def center_crop(img_rgb: np.ndarray, size=224):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img_rgb[y1:y1+size, x1:x1+size]

def random_crop(img_rgb: np.ndarray, size=224):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img_rgb[y:y+size, x:x+size]
def to_tensor_from_rgb(img_rgb_crop: np.ndarray, mean, std):
    x = img_rgb_crop.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def rotate_bound(img_rgb: np.ndarray, deg: float):
    if deg == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img_rgb, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
def resize_short_edge(img_rgb: np.ndarray, short=256):
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def make_views_for_global(img_bgr: np.ndarray, mean, std, to_rgb: bool):
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()
    img_rgb = resize_short_edge(img_rgb, RESIZE_SHORT)

    views = []
    for deg, n_center, n_rand in VIEW_PLAN:
        rot = rotate_bound(img_rgb, deg)
        for _ in range(n_center):
            views.append(to_tensor_from_rgb(center_crop(rot, CROP_SIZE), mean, std))
        for _ in range(n_rand):
            views.append(to_tensor_from_rgb(random_crop(rot, CROP_SIZE), mean, std))
    return views[:VIEWS_PER_IMAGE]

@torch.no_grad()
def aggregate_views_to_one(feats_view: torch.Tensor):
    # You used max pooling; keep consistent with your system.
    # agg = feats_view.max(dim=0).values
    agg = feats_view.mean(dim=0)

    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

@torch.no_grad()
def build_model(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(DEVICE)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb

# @torch.no_grad()
# def extract_feat(model, imgs):
#     feat = model.backbone(imgs)
#     if isinstance(feat, (tuple, list)):
#         feat = feat[-1]
#     if feat.dim() == 4:
#         feat = feat.mean(dim=(2, 3))
#     return F.normalize(feat, p=2, dim=1)

# def get_query_global_feat(model, mean, std, to_rgb, img_bgr):
#     img = cv2.resize(img_bgr, (224, 224))
#     if to_rgb:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     x = (img.astype(np.float32) - mean) / std
#     x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         return extract_feat(model, x).cpu().numpy()

def get_query_global_feat(model, mean, std, to_rgb, img_bgr, device=DEVICE):
    # 建议：seed 用“和建库一致的输入尺度”
    # 统一做 seed 的源图
    seed_src = resize_long_edge(img_bgr, max_long=STRIPE_LONG_EDGE)
    random.seed(seed_from_image(seed_src, base=0))

    # views 用同一个 seed_src（或者至少同一份 resize 结果）
    views = make_views_for_global(seed_src, mean, std, to_rgb)

    if len(views) == 0:
        raise RuntimeError("No global views generated for query.")

    bt = torch.stack(views, dim=0).to(device)              # (V,3,224,224)
    feats_v = extract_backbone_last(model, bt)             # (V,D) torch, 已 L2

    q = aggregate_views_to_one(feats_v)                    # (D,) torch
    return q.unsqueeze(0).detach().cpu().numpy().astype("float32")  # (1,D)


# ============================================================
# 7. Feature-map Patch Selection (for Geom Rerank)
# ============================================================
@torch.no_grad()
def batch_candidate_desc_xy(model, img_ids, img_cache, img_paths, mean, std, to_rgb,
                            device=DEVICE, batch_size=32, feat_level=FEAT_LEVEL):
    # 返回 dict: img_id -> (c_desc, c_xy)
    out = {}
    buf_ids = []
    buf_tensors = []

    for img_id in img_ids:
        cimg = img_cache.get(img_id, None)
        if cimg is None:
            cimg = imread_unicode(img_paths[img_id])
            if cimg is None:
                continue
            img_cache[img_id] = cimg

        t = make_single_tensor_for_rerank(cimg, mean, std, to_rgb)  # (1,3,512,512) on CPU
        buf_ids.append(img_id)
        buf_tensors.append(t)

        if len(buf_ids) >= batch_size:
            bt = torch.cat(buf_tensors, dim=0).to(device, non_blocking=True)  # (B,3,512,512)
            with autocast(enabled=(DEVICE.startswith("cuda"))):
                fm = extract_featmap(model, bt, feat_level)                       # (B,C,H,W)
            for i, _id in enumerate(buf_ids):
                d, x = select_candidate_patches(fm[i:i + 1])
                out[_id] = (d.float(), x.float())

            buf_ids, buf_tensors = [], []

    if buf_ids:
        bt = torch.cat(buf_tensors, dim=0).to(device, non_blocking=True)
        with autocast(enabled=(DEVICE.startswith("cuda"))):
            fm = extract_featmap(model, bt, feat_level)
        for i, _id in enumerate(buf_ids):
            d, x = select_candidate_patches(fm[i:i + 1])
            out[_id] = (d.float(), x.float())

    return out

def _expand_bbox_to_limit_ar(x1, y1, x2, y2, H, W, max_ar=3.0):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    ar = bh / bw
    if ar <= max_ar:
        return x1, y1, x2, y2

    # 需要的目标宽度，使 bh / new_bw <= max_ar
    target_bw = int(np.ceil(bh / max_ar))
    extra = target_bw - bw
    pad_l = extra // 2
    pad_r = extra - pad_l

    nx1 = max(0, x1 - pad_l)
    nx2 = min(W, x2 + pad_r)

    # 如果到边界还不够，就尽量再扩
    cur_bw = nx2 - nx1
    if cur_bw < target_bw:
        lack = target_bw - cur_bw
        nx1 = max(0, nx1 - lack // 2)
        nx2 = min(W, nx2 + (lack - lack // 2))

    return nx1, y1, nx2, y2

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
    e = fm_1chw.pow(2).sum(dim=0)
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
    fm = feat_map_1bchw[0]
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)

    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))
    mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
    mask[y1b:y2b, x1b:x2b] = True

    if roi_fbox is not None:
        rx1, ry1, rx2, ry2 = roi_fbox
        rx1 = max(0, min(W-1, int(rx1))); ry1 = max(0, min(H-1, int(ry1)))
        rx2 = max(1, min(W,   int(rx2))); ry2 = max(1, min(H,   int(ry2)))
        if rx2 > rx1 and ry2 > ry1:
            roi_mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
            roi_mask[ry1:ry2, rx1:rx2] = True
            mask = mask & roi_mask

    idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        idx_all = torch.arange(H * W, device=energy.device)

    k = min(int(keep), int(idx_all.numel()))
    vals = energy.flatten()[idx_all]
    top_local = torch.topk(vals, k=k, largest=True).indices
    idx = idx_all[top_local]

    patches = fm.flatten(1).t()[idx]
    patches = F.normalize(patches, p=2, dim=1)

    ys = (idx // W).float()
    xs = (idx % W).float()
    xs = xs / max(1.0, float(W - 1))
    ys = ys / max(1.0, float(H - 1))
    xy = torch.stack([xs, ys], dim=1)
    return patches, xy

@torch.no_grad()
def select_query_patches(q_fm_1bchw: torch.Tensor,
                         keep=KEEP_PATCHES, border=BORDER,
                         roi_frac=Q_ROI_FRAC, roi_ratio=Q_ROI_RATIO):
    q_roi = energy_roi_box(q_fm_1bchw[0], frac=roi_frac)
    k_roi = int(round(keep * roi_ratio))
    k_full = max(1, keep - k_roi)

    desc_list, xy_list = [], []
    if q_roi is not None and k_roi >= 4:
        d1, x1 = select_top_patches_with_xy(q_fm_1bchw, keep=k_roi, border=border, roi_fbox=q_roi)
        if d1 is not None and d1.shape[0] >= 4:
            desc_list.append(d1); xy_list.append(x1)

    d2, x2 = select_top_patches_with_xy(q_fm_1bchw, keep=k_full, border=border, roi_fbox=None)
    desc_list.append(d2); xy_list.append(x2)

    return torch.cat(desc_list, dim=0), torch.cat(xy_list, dim=0)

@torch.no_grad()
def select_candidate_patches(c_fm_1bchw: torch.Tensor,
                             keep=KEEP_PATCHES, border=BORDER,
                             roi_frac=C_ROI_FRAC):
    c_roi = energy_roi_box(c_fm_1bchw[0], frac=roi_frac)
    d, x = select_top_patches_with_xy(c_fm_1bchw, keep=keep, border=border, roi_fbox=c_roi)
    return d, x

def sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()

# ============================================================
# 8. Patch Extraction for FAISS Patch Index Query
# ============================================================
def gen_stripe_windows(
        H: int, W: int,
        max_patches: int,
        seed: int,
        win_w: int = 192,       # “条带宽度方向”的窗口厚度（像素）
        win_h: int = 384,       # 沿长边窗口长度（像素）
        stride: int = None,     # 沿长边步长；None -> win_h//4
        center_frac: float = 0.92,
        jitter: int = 8,
):
    """
    返回 windows: list of (x1,y1,x2,y2,win,patch_type,pos_int)
    - patch_type=1 (stripe)
    - pos_int: 0..10000 表示沿长边位置
    自动判断竖条/横条，并沿长边滑窗。
    """
    rng = np.random.default_rng(seed)
    vertical = (H >= W)

    windows = []
    if stride is None:
        stride = max(32, win_h // 4)

    if vertical:
        # 竖条：裁左右，只在中间区域取窗，减少背景
        Wu = int(round(W * center_frac))
        x0 = max(0, (W - Wu) // 2)
        # 在裁剪后的坐标系里算窗，再映射回原图
        x_base = x0 + max(0, (Wu - min(win_w, Wu)) // 2)
        ww = min(win_w, Wu)
        hh = min(win_h, H)

        if ww < 16 or hh < 16:
            return []

        y = 0
        while y + hh <= H and len(windows) < max_patches:
            dx = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            x1 = int(np.clip(x_base + dx, x0, x0 + Wu - ww))
            y1 = int(y)
            x2 = x1 + ww
            y2 = y1 + hh

            # pos: 沿长边（y轴）中心位置归一化
            center = (y1 + y2) * 0.5
            pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            y += stride
        # ensure tail coverage
        if len(windows) > 0:
            last = windows[-1]
            if last[1] != max(0, H - hh):
                last_y2 = windows[-1][3]
                if last_y2 < H:
                    y1 = max(0, H - hh)
                    x1 = windows[-1][0]
                    x2 = x1 + ww
                    y2 = y1 + hh
                    center = (y1 + y2) * 0.5
                    pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
                    windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    else:
        # 横条：裁上下，只在中间区域取窗
        Hu = int(round(H * center_frac))
        y0 = max(0, (H - Hu) // 2)

        # 这里沿长边是 x 方向；厚度是 win_w，长度是 win_h
        hh = min(win_w, Hu)     # 厚度
        ww = min(win_h, W)      # 长度
        if hh < 16 or ww < 16:
            return []

        y_base = y0 + max(0, (Hu - hh) // 2)

        x = 0
        while x + ww <= W and len(windows) < max_patches:
            dy = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            y1 = int(np.clip(y_base + dy, y0, y0 + Hu - hh))
            x1 = int(x)
            x2 = x1 + ww
            y2 = y1 + hh

            center = (x1 + x2) * 0.5
            pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            x += stride
        # ensure tail coverage
        # ensure tail coverage (horizontal stripe: slide along x, thickness along y)
        if len(windows) > 0:
            last_x2 = windows[-1][2]
            if last_x2 < W:
                x1 = max(0, W - ww)
                y1 = windows[-1][1]  # keep same y band
                x2 = x1 + ww
                y2 = y1 + hh
                center = (x1 + x2) * 0.5
                pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
                windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    return windows

def seed_from_image(img_bgr: np.ndarray, base: int = 999) -> int:
    if img_bgr is None or img_bgr.size == 0:
        return base & 0x7fffffff
    h = hashlib.md5(img_bgr.tobytes()).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff

def seed_from_path(p: str, base: int = 999) -> int:
    h = hashlib.md5(str(p).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff

def _resize_long_edge(img_bgr, long_edge=1536):
    h, w = img_bgr.shape[:2]
    s = long_edge / float(max(h, w))
    if s >= 1.0:
        return img_bgr
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

@torch.no_grad()
def get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg_bgr,
        qmask=None,
        long_edge=1536,
        stripe_ar_thr=STRIPE_AR_THR,
        stripe_win_h=STRIPE_WIN_H,
        stripe_stride=STRIPE_STRIDE,
        stripe_max_patches=STRIPE_MAX_PATCHES,
        stripe_center_frac=STRIPE_CENTER_FRAC,
        stripe_jitter=STRIPE_JITTER,
        min_mask_cover=0.0,
        batch_size=64,
):
    qimg = _resize_long_edge(qimg_bgr, long_edge=long_edge)
    H, W = qimg.shape[:2]

    qmask_rs = None
    if qmask is not None:
        qmask_rs = cv2.resize(qmask, (W, H), interpolation=cv2.INTER_NEAREST)

    # ar = max(W / (H + 1e-6), H / (W + 1e-6))
    # is_stripe = (ar >= stripe_ar_thr)
    #
    # patches = []
    q_seed = seed_from_image(qimg, base=999)
    qimg_resized, windows, is_stripe = gen_patch_windows_unified(
        qimg_bgr,
        max_long=long_edge,
        stripe_ar_thr=stripe_ar_thr,
        seed_base=999,
        stripe_max_patches=stripe_max_patches,
        stripe_win_h=stripe_win_h,
        stripe_stride=stripe_stride,
        stripe_center_frac=stripe_center_frac,
        stripe_jitter=stripe_jitter,
        grid_sizes=(256, 384, 512, 768),
        grid_stride_ratio=0.5,
        max_patches=60,
    )

    patches = []
    for (x1, y1, x2, y2, win, ptype, pos) in windows:
        patch = qimg_resized[y1:y2, x1:x2]
        patches.append(patch)
    if is_stripe:
        win_w = int(np.clip(STRIPE_WIN_W_FRAC * W, STRIPE_WIN_W_MIN, STRIPE_WIN_W_MAX))
        windows = gen_stripe_windows(
            H, W,
            max_patches=stripe_max_patches,
            seed=q_seed,
            win_w=win_w,
            win_h=stripe_win_h,
            stride=stripe_stride,
            center_frac=stripe_center_frac,
            jitter=stripe_jitter,
        )

        if len(windows) < min(6, stripe_max_patches):
            ww = min(max(win_w, 160), W)
            hh = min(max(stripe_win_h, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            windows.append((x1, y1, x1 + ww, y1 + hh, int(max(ww, hh)), 1, 5000))

        # mask 过滤（要完全对齐建库就 min_mask_cover=0）
        if qmask_rs is not None and min_mask_cover > 0:
            filtered = []
            for (x1, y1, x2, y2, win, ptype, pos) in windows:
                cover = float(qmask_rs[y1:y2, x1:x2].mean()) / 255.0
                if cover >= min_mask_cover:
                    filtered.append((x1, y1, x2, y2, win, ptype, pos))
            if len(filtered) >= 16:
                windows = filtered

        for (x1, y1, x2, y2, *_rest) in windows:
            patches.append(qimg[y1:y2, x1:x2])

    else:
        patches_with_xy = _extract_patches_grid(
            qimg,
            patch_sizes=(256, 384, 512, 768),
            stride_ratio=0.5,
            max_patches=60,
            roi_xyxy=None,
            return_xyxy=True
        )
        for patch, (x1, y1, x2, y2) in patches_with_xy:
            if qmask_rs is not None and min_mask_cover > 0:
                cover = float(qmask_rs[y1:y2, x1:x2].mean()) / 255.0
                if cover < min_mask_cover:
                    continue
            patches.append(patch)

    if len(patches) == 0:
        patches = [qimg]

    # --- encode ---
    tensors = []
    for p in patches:
        p224 = cv2.resize(p, (224, 224), interpolation=cv2.INTER_LINEAR)
        if to_rgb:
            p224 = cv2.cvtColor(p224, cv2.COLOR_BGR2RGB)
        x = (p224.astype(np.float32) - mean) / std
        tensors.append(torch.from_numpy(x.transpose(2, 0, 1)))  # (3,224,224) CPU torch

    feats_chunks = []
    for st in range(0, len(tensors), batch_size):
        bt = torch.stack(tensors[st:st + batch_size]).to(DEVICE)
        fv = extract_backbone_last(model, bt)   # (b,D) torch
        feats_chunks.append(fv.detach().cpu())  # keep torch on CPU

    feats = torch.cat(feats_chunks, dim=0).numpy().astype("float32")  # (N,D) numpy
    return feats, len(patches), bool(is_stripe), (H, W)

def _extract_patches_grid(img_bgr, patch_sizes=(256, 384, 512,768), stride_ratio=0.5,
                          max_patches=64, border_frac=0.02, roi_xyxy=None,
                          return_xyxy=False):
    H, W = img_bgr.shape[:2]
    if roi_xyxy is not None:
        rx1, ry1, rx2, ry2 = roi_xyxy
        rx1 = max(0, int(rx1)); ry1 = max(0, int(ry1))
        rx2 = min(W, int(rx2)); ry2 = min(H, int(ry2))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H

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

        ys = list(range(y0, y1m + 1, stride)) if y1m >= y0 else [max(0, (HH - ps)//2)]
        xs = list(range(x0, x1m + 1, stride)) if x1m >= x0 else [max(0, (WW - ps)//2)]

        for yy in ys:
            for xx in xs:
                patch = crop[yy:yy+ps, xx:xx+ps]
                if patch.shape[0] == ps and patch.shape[1] == ps:
                    if return_xyxy:
                        x1 = rx1 + xx; y1 = ry1 + yy
                        x2 = x1 + ps; y2 = y1 + ps
                        patches.append((patch, (x1, y1, x2, y2)))
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
# ============================================================
# 9. FAISS / Ranking Utilities (RRF, patch aggregation)
# ============================================================
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

def aggregate_patch_hits(patch_ids, patch_scores, patch_meta, top_images=4000, topM=8, tau=0.15):
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
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        score = m + max(tau, 1e-6) * np.log(v + 1e-9)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]

def clean_rank(rank_list):
    seen = set()
    out = []
    for x in rank_list:
        if x is None:
            continue
        x = int(x)
        if x < 0 or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def rank_to_rrf_score(rank_list, k=60):
    s = {}
    for r, i in enumerate(rank_list, 1):
        s[i] = 1.0 / (k + r)
    return s

def rrf_fuse(rankA, rankB, k=60):
    score = {}
    for r, i in enumerate(rankA, 1):
        score[i] = score.get(i, 0) + 1 / (k + r)
    for r, i in enumerate(rankB, 1):
        score[i] = score.get(i, 0) + 1 / (k + r)
    return [i for i, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]

def global_confidence(global_rank, img_paths, topn=20):
    names = [os.path.basename(str(img_paths[i])) for i in global_rank[:topn]]
    prefix = [n.split('_')[0] if '_' in n else n[:4] for n in names]
    from collections import Counter
    c = Counter(prefix).most_common(1)[0][1]
    return c / max(1, len(prefix))


# ============================================================
# 10. Geometric Consistency Scoring (patch-based) + Texture fallback
# ============================================================
@torch.no_grad()
def compute_Ng0(q_desc, c_desc, margin=0.015):
    sim = q_desc @ c_desc.t()
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0
    topv, topi = torch.topk(sim, k=2, dim=1)
    q_best = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    return int(good.sum().item())

@torch.no_grad()
def texture_score(q_desc, q_xy, c_desc, c_xy,
                  bin_size=0.05, topM=6,
                  topk_core=128, min_pairs=12):
    sim = q_desc @ c_desc.t()
    q_bestv, q_best = torch.max(sim, dim=1)

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
        return 0.0
    if core < 0.45 and peak_ratio > 0.6:
        return 0.0

    return float(core * (0.35 + 0.65 * cover_topM) * (0.5 + 0.5 * peak_ratio))

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
    sim = q_desc @ c_desc.t()
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0.0

    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng0 = int(good.sum().item())
    if Ng0 < min_keep:
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

    return float(score)

@torch.no_grad()
def geom_score_compatible(q_desc, q_xy, c_desc, c_xy,
                          margin=0.012, min_keep=5,
                          bin_size=0.05, topM=6, topk_core=64,
                          periodic_peak_thr=0.25,
                          periodic_cover_topM_thr=0.70,
                          periodic_cover_xy_thr=0.22,
                          tex_topk_core=128, tex_min_pairs=12,
                          tex_weight=0.85, return_ng0=False):
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

    s_tex = texture_score(q_desc, q_xy, c_desc, c_xy,
                          bin_size=bin_size, topM=topM,
                          topk_core=tex_topk_core, min_pairs=tex_min_pairs)
    out = float(s_tex * tex_weight)
    return (out, Ng0) if return_ng0 else out


# ============================================================
# 11. Head Gate (cand_gate_score)
# ============================================================
def cand_gate_score(cimg_bgr, q_head: dict, return_dbg=False):
    ch = stripe_grid_head_v21(make_head_view(cimg_bgr, prefer_gray=False))

    wP = 0.25
    qs = np.array([q_head["stripe_score"], q_head["grid_score"],
                   wP * np.clip(q_head["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)
    cs = np.array([ch["stripe_score"], ch["grid_score"],
                   wP * np.clip(ch["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)

    qsn = float(np.linalg.norm(qs))
    csn = float(np.linalg.norm(cs))

    if csn < 0.06 or qsn < 1e-6:
        sim = 0.0
    else:
        sim = float((qs * cs).sum() / (qsn * csn + 1e-6))
        sim = max(0.0, min(1.0, sim))

    def _ret(g, reason):
        if return_dbg:
            return float(g), str(reason), float(sim), ch
        return float(g)

    q_is_grid = is_grid_like(q_head)
    if q_is_grid and (not is_grid_like(ch)):
        if q_head.get("grid_score", 0.0) >= 0.35:
            return _ret(0.0, "grid_mismatch_hard")
        return _ret(0.20, "grid_mismatch_soft")

    plaid_penalty = 1.0
    if q_head.get("grid_score", 0.0) > 0.18:
        if not is_grid_like(ch):
            return _ret(0.0, "plaid_miss")
        plaid_penalty = 1.0
    elif q_head.get("grid_score", 0.0) > 0.10:
        if ch.get("grid_score", 0.0) < 0.02:
            plaid_penalty = 0.45

    if q_head.get("stripe_score", 0.0) > 0.15 and ch.get("stripe_score", 0.0) < 0.08:
        return _ret(0.0, "stripe_miss")

    if sim < 0.15:
        return _ret(0.0, "sim_low")

    peak_penalty = 1.0
    if q_head.get("ori_peakedness", 0.0) > 5.5 and ch.get("ori_peakedness", 0.0) < 3.8:
        peak_penalty = 0.4

    scale_penalty = 1.0
    if ("r_peak" in q_head) and ("r_peak" in ch):
        rq = float(q_head.get("r_peak", 0.0))
        rc = float(ch.get("r_peak", 0.0))
        if q_head.get("ori_peakedness", 0.0) >= 2.5 and rq > 1e-6 and rc > 1e-6:
            ratio = rc / (rq + 1e-6)
            if ratio < 0.50 or ratio > 3.00:
                scale_penalty = 0.35
            elif ratio < 0.75 or ratio > 1.60:
                scale_penalty = 0.75

    g = float((0.60 + 0.55 * sim) * peak_penalty * plaid_penalty * scale_penalty)
    return _ret(g, "pass")


# ============================================================
# 12. ORB + RANSAC Geom Score (image-level)
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
    for m, n in matches:
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
# 13. Visualization (result grid)
# ============================================================
def _put_text(img, text, org=(8, 26), font_scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _fit_square(img_bgr, tile=320):
    h, w = img_bgr.shape[:2]
    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def visualize_grid(query_bgr, top_imgs_bgr, top_scores, out_path,
                   tile=320, gap=10, header=44):
    tiles = [_fit_square(query_bgr, tile)]
    labels = ["QUERY"]

    for i, (img, s) in enumerate(zip(top_imgs_bgr, top_scores), 1):
        if img is None:
            img = np.zeros((tile, tile, 3), np.uint8)
        tiles.append(_fit_square(img, tile))
        labels.append(f"#{i}  {float(s):.3f}")

    n = len(tiles)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)

        canvas[y:y+header, x:x+tile] = 0
        _put_text(canvas, labels[idx], org=(x + 8, y + 28), font_scale=0.7, thickness=2)
        canvas[y+header:y+header+tile, x:x+tile] = tiles[idx]

    cv2.imwrite(out_path, canvas)
    return out_path


# ============================================================
def main():
    ensure_dir(OUT_DIR)

    # ---- load index & model
    g_index = faiss.read_index(GLOBAL_INDEX)
    p_index_s = faiss.read_index(PATCH_STRIPE_INDEX)
    p_index_g = faiss.read_index(PATCH_GRID_INDEX)
    set_faiss_nprobe(g_index, 64)
    set_faiss_nprobe(p_index_s, 64)
    set_faiss_nprobe(p_index_g, 64)

    img_paths = np.load(GLOBAL_META, allow_pickle=True)

    # target_path = r"...\IMG_8830(1)_label0_score0.999.png"  # 你认为在库里的那张
    # # 在 img_paths 里找它的 img_id
    #
    # base = os.path.basename(target_path)
    # hits = [i for i, p in enumerate(img_paths) if os.path.basename(str(p)) == base]
    # print("target hits:", hits[:10], "count=", len(hits))

    patch_meta_s = np.load(PATCH_STRIPE_META, allow_pickle=True)
    patch_meta_g = np.load(PATCH_GRID_META, allow_pickle=True)

    model, mean, std, to_rgb = build_model(CONFIG, CKPT)


    # ---- load query
    qimg0 = imread_unicode(QUERY_IMG)
    if qimg0 is None:
        raise RuntimeError(f"Failed to read query image: {QUERY_IMG}")

    # ---- seg & crop
    qimg, qmask, qimg_raw = crop_by_mmdet_mask_final(
        qimg0,
        score_thr=SEG_SCORE_THR,
        use_classes=SEG_USE_CLASSES,
        merge_all=True,
        do_rectify=False,
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=None
    )



    # ---- head feature (query)
    q_head_img = make_head_view(qimg_raw, prefer_gray=True)
    q_head = stripe_grid_head_v21(q_head_img)

    # ---- global feature
    qvec = get_query_global_feat(model, mean, std, to_rgb, qimg)


    q_patch_vecs, n_qpatch, is_stripe2, hw = get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg,
        qmask=qmask,
        long_edge=STRIPE_LONG_EDGE,
        stripe_ar_thr=STRIPE_AR_THR,  # ✅ 不要写死 3.0
        stripe_win_h=STRIPE_WIN_H,
        stripe_stride=STRIPE_STRIDE,
        stripe_max_patches=STRIPE_MAX_PATCHES,
        min_mask_cover=0.0,
        batch_size=64
    )

    is_vertical_stripe = is_stripe2
    print(f"[PATCH] is_stripe={is_stripe2}, resized={hw}, n={n_qpatch}")

    # ---- global search
    _, gids = g_index.search(qvec, TOPG)
    global_rank = clean_rank(gids[0].tolist())

    # ---- patch search
    if is_stripe2:
        D, I = p_index_s.search(q_patch_vecs, PATCH_TOPK_PER_QPATCH)
        S = faiss_scores_from_D(p_index_s, D.astype(np.float32))
        patch_rank = aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            patch_meta_s,
            top_images=TOP_PATCH_IMAGES,
            tau=0.15
        )
    else:
        D, I = p_index_g.search(q_patch_vecs, PATCH_TOPK_PER_QPATCH)
        S = faiss_scores_from_D(p_index_g, D.astype(np.float32))
        patch_rank = aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            patch_meta_g,
            top_images=TOP_PATCH_IMAGES,
            tau=0.15
        )

    # ---- RRF fusion
    rrf_g = rank_to_rrf_score(global_rank, k=RRF_K)
    rrf_p = rank_to_rrf_score(patch_rank,  k=RRF_K)

    # ---- weight for fusion (Fix-2: boost patch when patch is very confident) ----
    conf_g = global_confidence(global_rank, img_paths, topn=20)

    # 你原来的默认策略（global 偏大）
    w_g = 0.6 + 0.35 * conf_g
    w_p = 1.0 - w_g

    # 1) head 强纹理/强格纹：偏向 patch（你原来的逻辑保留）
    if (q_head.get("stripe_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8) or \
            (q_head.get("grid_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8):
        w_g, w_p = 0.20, 0.80

    # 2) patch 很强而 global 很弱：强行提高 patch 权重（Fix-2 核心）
    def _rank_pos(lst, x):
        try:
            return lst.index(x) + 1
        except:
            return None

    # 可选：如果你有 target_id（调试用），可以用它做判定；没有就走“统计判定”
    # ——统计判定：patch 前几名在 global 都很靠后 => global 不可信
    try:
        top_patch_ids = patch_rank[:10]  # 看 patch 前10
        gp = []
        for pid in top_patch_ids:
            p = _rank_pos(global_rank, pid)
            if p is not None:
                gp.append(p)

        # 触发条件：patch 前10 的 median global 排名很差，说明 global 拉胯
        # 你可以调阈值：80/100/150 都行；我给一个相对保守的 120
        if len(gp) >= 5:
            gp_sorted = sorted(gp)
            median_gp = gp_sorted[len(gp_sorted) // 2]

            # 典型情况：patch很准，但global把它们排很后
            if median_gp >= 120:
                # 直接把 patch 权重拉高
                w_g, w_p = 0.15, 0.85

        # 进一步增强：如果 global_confidence 也很低，就更偏 patch
        if conf_g < 0.25:
            w_g, w_p = min(w_g, 0.20), max(w_p, 0.80)

    except Exception:
        pass

    # 最后做一次 clamp，避免数值越界
    w_g = float(np.clip(w_g, 0.05, 0.95))
    w_p = float(np.clip(1.0 - w_g, 0.05, 0.95))

    print(f"[W] conf_g={conf_g:.3f}  w_g={w_g:.3f}  w_p={w_p:.3f}")

    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    # ---- candidate pool

    # 1) 先拿一个“候选集合”（用原 fused 只是为了召回，不用它的排序）
    cand = clean_rank(rrf_fuse(global_rank, patch_rank, RRF_K))

    # 2) 用 final_rrf 给候选重新排序（关键修复点）
    cand.sort(key=lambda i: final_rrf.get(i, 0.0), reverse=True)

    # 3) 取前 GEOM_TOPN 进入 gate / geom
    fused = cand[:GEOM_TOPN]

    fused2 = []
    cimg_cache = {}
    for img_id in fused:
        cimg = imread_unicode(img_paths[img_id])
        if cimg is None:
            continue
        g0 = cand_gate_score(cimg, q_head)
        if g0 > 0:
            fused2.append(img_id)
            cimg_cache[img_id] = cimg



    def rank_pos(lst, x):
        try:
            return lst.index(x) + 1
        except:
            return None
    # target_id = hits[0]
    # print("target global pos:", rank_pos(global_rank, target_id))
    # print("target patch  pos:", rank_pos(patch_rank, target_id))
    # print("target fused  pos:", rank_pos(fused, target_id))
    # print("target fused2 pos:", rank_pos(fused2, target_id))  # fused2 是 gate 后的
    # print("target fused(pos after weighted):", rank_pos(fused, target_id))

    # ---- geom rerank
    angles = [-10, -5, 0, 5, 10] if is_vertical_stripe else [-30, -15, 0, 15, 30]

    # query 侧还是按角度算（数量小，问题不大）
    q_desc_list, q_xy_list = [], []
    for ang in angles:
        qimg_r = rotate_bgr(qimg, ang)
        qx = make_single_tensor_for_rerank(qimg_r, mean, std, to_rgb).to(DEVICE)
        q_fm = extract_featmap(model, qx, FEAT_LEVEL)
        q_desc, q_xy = select_query_patches(q_fm)
        q_desc_list.append(q_desc.float())
        q_xy_list.append(q_xy.float())

    # 候选图：一次 batch forward，拿到每张图的 (c_desc, c_xy)
    cand_desc_xy = batch_candidate_desc_xy(
        model, fused2, cimg_cache, img_paths, mean, std, to_rgb,
        device=DEVICE, batch_size=32, feat_level=FEAT_LEVEL
    )


    scored = []
    for img_id in fused2:
        if img_id not in cand_desc_xy:
            continue
        c_desc, c_xy = cand_desc_xy[img_id]

        geom_best, cnt = 0.0, 0
        for q_desc, q_xy in zip(q_desc_list, q_xy_list):
            s = geom_score_compatible(
                q_desc, q_xy, c_desc, c_xy,
                margin=0.012, min_keep=5,
                bin_size=BIN_SIZE, topM=TOPM, topk_core=TOPK_CORE,
                periodic_peak_thr=PERIODIC_PEAK_THR,
                periodic_cover_topM_thr=PERIODIC_COVER_TOPM_THR,
                periodic_cover_xy_thr=PERIODIC_COVER_XY_THR,
                tex_weight=0.85
            )
            if s > 0:
                geom_best += s
                cnt += 1
        if cnt > 0:
            scored.append((img_id, geom_best / cnt))

    if not scored:
        fallback = fused2[:TOPK] if fused2 else fused[:TOPK]
        imgs = [imread_unicode(img_paths[i]) for i in fallback]
        scores = [1.0 - i / len(fallback) for i in range(len(fallback))]
        out = os.path.join(OUT_DIR, "result_grid1.png")
        visualize_grid(qimg, imgs, scores, out)
        print(f"[DONE] saved {out}")
        return

    # ---- final score
    geom_vals = [s for _, s in scored]
    gmin, gmax = min(geom_vals), max(geom_vals)

    final = []
    for img_id, gs in scored:
        norm_g = (gs - gmin) / (gmax - gmin + 1e-9)
        fs = final_rrf.get(img_id, 0.0) * (1.0 + 0.6 * norm_g)
        final.append((img_id, fs))

    final.sort(key=lambda x: x[1], reverse=True)
    top = final[:TOPK]

    imgs = [cimg_cache[i] if i in cimg_cache else imread_unicode(img_paths[i]) for i, _ in top]
    scores = [s for _, s in top]

    out = os.path.join(OUT_DIR, "result_grid1.png")
    visualize_grid(qimg, imgs, scores, out)
    print(f"[DONE] saved {out}")

# ============================================================
# 15. Entry
# ============================================================
if __name__ == "__main__":
    main()
