#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qt-friendly + Optimized retrieval (RTX 4060 8GB):
- Keep algorithm logic as-is (RRF + geom_score_compatible)
- Candidate rerank featmap extraction -> BATCH forward
- AMP(fp16) for backbone forward
- Two-stage rotation (0° first; only hard cases run extra rotations)
- PATCH_TOPK_PER_QPATCH reduced (default 200) for speed (tune 200~400)
- Context cache: model/faiss/meta loaded once for Qt repeated queries
"""

import os
import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS
from torch.cuda.amp import autocast

# ============================================================
# CONFIG (your paths)
# ============================================================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid_new_data"
GLOBAL_INDEX = os.path.join(INDEX_DIR, "global.index")
PATCH_INDEX  = os.path.join(INDEX_DIR, "patch.index")
GLOBAL_META  = os.path.join(INDEX_DIR, "global_img_paths.npy")
PATCH_META   = os.path.join(INDEX_DIR, "patch_meta.npy")
force_reload_flag=False #是否重新加载库的数据

# ============================================================
# Runtime / device
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For RTX 4060 8GB (safe defaults)
CAND_BATCH = 16                # candidate batch size for featmap forward (8/12/16/20)
USE_AMP = (DEVICE == "cuda")   # enable fp16 amp
torch.backends.cudnn.benchmark = True

# ============================================================
# Retrieval params (aligned to optimized search code)
# ============================================================
TOPG = 300
TOPK_DEFAULT = 12

PATCH_TOPK_PER_QPATCH = 200    # optimized search default
TOP_PATCH_IMAGES = 600
RRF_K = 60
GEOM_TOPN = 200                # rerank only on topN after fusion

# ============================================================
# Rerank / patches selection params
# ============================================================
FEAT_LEVEL = -2
RMAC_INPUT_SIZE = 512
KEEP_PATCHES = 256
BORDER = 0.05

# ROI / selection
Q_ROI_FRAC = 0.18
C_ROI_FRAC = 0.18
Q_ROI_RATIO = 0.50
C_ROI_RATIO = 1.00  # reserved

# geom params
MARGIN = 0.015
MIN_KEEP = 6
BIN_SIZE = 0.05
TOPK_CORE = 64

PERIODIC_PEAK_THR = 0.25
PERIODIC_COVER_TOPM_THR = 0.70
PERIODIC_COVER_XY_THR = 0.22
TOPM = 6

# Two-stage rotation (speed)
ROT_FAST = [0]
ROT_FULL = [-30, -15, 0, 15, 30]
ROT_TRIGGER = 0.10   # if score(0deg) < this, try full rotations

# geom fusion
BETA_GEOM = 0.3

# ============================================================
# Global caches for Qt repeated calls
# ============================================================
_CTX = None
_q_full_cache = {}  # key -> (q_desc_list_full, q_xy_list_full)

# ============================================================
# Utils
# ============================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def imread_unicode(p: str):
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

@torch.no_grad()
def crop_by_feat_energy_qt_safe(
    model,
    img_bgr: np.ndarray,
    mean, std,
    to_rgb: bool,
    prefer_level: int = -2,
    input_size: int = 512,
    top_frac: float = 0.20,
    border_frac: float = 0.08,
    min_area_frac: float = 0.18,
    pad_px: int = 24,
    morph_ks: int = 13,
):
    """
    Qt / 工程稳定版裁剪：
    - fp32
    - no AMP
    - no cudnn benchmark
    - single pad_to_square
    - strong fallback
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    # ---------- 0. 强制确定性 ----------
    torch.backends.cudnn.benchmark = False

    # ---------- 1. pad 一次（唯一一次） ----------
    img_sq = pad_to_square(img_bgr)
    S = img_sq.shape[0]

    # ---------- 2. 构造 512 输入（不走 make_single_tensor_for_rerank） ----------
    img_in = cv2.resize(img_sq, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    if to_rgb:
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_in[:, :, ::-1].copy()

    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    # ---------- 3. backbone（无 AMP） ----------
    out = model.backbone(x)

    if isinstance(out, (tuple, list)):
        fm = out[prefer_level]
    elif isinstance(out, dict):
        fm = list(out.values())[-1]
    else:
        fm = out

    fm = fm.float()[0]   # (C,Hf,Wf)

    # ---------- 4. energy map ----------
    e = fm.pow(2).sum(dim=0, keepdim=True).unsqueeze(0)
    e = F.interpolate(e, size=(input_size, input_size),
                      mode="bilinear", align_corners=False)[0, 0]

    e -= e.min()
    e /= (e.max() + 1e-6)
    e_np = e.cpu().numpy()

    # ---------- 5. 边缘抑制 ----------
    b = int(input_size * border_frac)
    if b > 0:
        e_np[:b, :] *= 0.1
        e_np[-b:, :] *= 0.1
        e_np[:, :b] *= 0.1
        e_np[:, -b:] *= 0.1

    # ---------- 6. top-p mask ----------
    thr = np.quantile(e_np.reshape(-1), 1.0 - top_frac)
    mask = (e_np >= thr).astype(np.uint8) * 255

    # ---------- 7. morphology ----------
    k = max(3, morph_ks | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ker, iterations=1)

    # ---------- 8. 最大连通域 ----------
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return img_bgr

    areas = stats[1:, cv2.CC_STAT_AREA]
    i = 1 + int(np.argmax(areas))
    x0, y0, w0, h0, area = stats[i]

    if area < min_area_frac * (input_size * input_size):
        return img_bgr

    # ---------- 9. 映射回原图 ----------
    x1 = max(0, x0 - pad_px)
    y1 = max(0, y0 - pad_px)
    x2 = min(input_size, x0 + w0 + pad_px)
    y2 = min(input_size, y0 + h0 + pad_px)

    scale = float(S) / float(input_size)
    X1 = int(round(x1 * scale))
    Y1 = int(round(y1 * scale))
    X2 = int(round(x2 * scale))
    Y2 = int(round(y2 * scale))

    X1 = max(0, min(S - 4, X1))
    Y1 = max(0, min(S - 4, Y1))
    X2 = max(X1 + 4, min(S, X2))
    Y2 = max(Y1 + 4, min(S, Y2))

    crop = img_sq[Y1:Y2, X1:X2].copy()

    if crop.shape[0] * crop.shape[1] < 0.15 * (S * S):
        return img_bgr

    return crop


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

def get_ctx(force_reload=False):
    """Load faiss indices + meta + model once (Qt repeated queries)."""
    global _CTX
    if not force_reload and _CTX is not None:
        return _CTX

    # 强制重新加载索引和元数据
    g_index = faiss.read_index(GLOBAL_INDEX)
    p_index = faiss.read_index(PATCH_INDEX)
    set_faiss_nprobe(g_index, 64)
    set_faiss_nprobe(p_index, 64)

    img_paths = np.load(GLOBAL_META, allow_pickle=True)
    patch_meta = np.load(PATCH_META, allow_pickle=True)

    model, mean, std, to_rgb = build_model(CONFIG, CKPT)

    _CTX = (g_index, p_index, img_paths, patch_meta, model, mean, std, to_rgb)
    return _CTX


# ============================================================
# Model helpers
# ============================================================
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

@torch.no_grad()
def extract_feat(model, imgs):
    with autocast(enabled=USE_AMP):
        feat = model.backbone(imgs)
    if isinstance(feat, (tuple, list)):
        feat = feat[-1]
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))
    feat = feat.float()
    return F.normalize(feat, p=2, dim=1)

@torch.no_grad()
def extract_featmap(model, batch_tensor: torch.Tensor, prefer_level: int):
    with autocast(enabled=USE_AMP):
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
        fm = out[lvl]
    elif isinstance(out, torch.Tensor):
        fm = out
    else:
        raise TypeError(f"Unsupported backbone output type: {type(out)}")

    return fm.float()

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
def make_batch_tensor_for_rerank(imgs_bgr: List[np.ndarray], mean, std, to_rgb: bool):
    xs = []
    for img_bgr in imgs_bgr:
        if to_rgb:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr[:, :, ::-1].copy()
        img_rgb = pad_to_square(img_rgb)
        img_rgb = cv2.resize(img_rgb, (RMAC_INPUT_SIZE, RMAC_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = img_rgb.astype(np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        xs.append(torch.from_numpy(x))
    return torch.stack(xs, dim=0)  # (B,3,H,W)

# ============================================================
# Patch selection (energy ROI + top patches)
# ============================================================
def energy_roi_box(fm_1chw: torch.Tensor, frac=0.18):
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
    fm = feat_map_1bchw[0]  # (C,H,W)
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)  # (H,W)

    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))
    mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
    mask[y1b:y2b, x1b:x2b] = True

    if roi_fbox is not None:
        rx1, ry1, rx2, ry2 = roi_fbox
        rx1 = max(0, min(W - 1, int(rx1)))
        ry1 = max(0, min(H - 1, int(ry1)))
        rx2 = max(1, min(W, int(rx2)))
        ry2 = max(1, min(H, int(ry2)))
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

    patches = fm.flatten(1).t()[idx]        # (k,C)
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

    q_desc = torch.cat(desc_list, dim=0)
    q_xy   = torch.cat(xy_list, dim=0)
    return q_desc, q_xy

@torch.no_grad()
def select_candidate_patches(c_fm_1bchw: torch.Tensor,
                             keep=KEEP_PATCHES, border=BORDER,
                             roi_frac=C_ROI_FRAC):
    c_roi = energy_roi_box(c_fm_1bchw[0], frac=roi_frac)
    d, x = select_top_patches_with_xy(c_fm_1bchw, keep=keep, border=border, roi_fbox=c_roi)
    return d, x

# ============================================================
# Global query feat
# ============================================================
def get_query_global_feat(model, mean, std, to_rgb, img_bgr):
    img = cv2.resize(img_bgr, (224, 224))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE, non_blocking=True)
    with torch.no_grad():
        return extract_feat(model, x).cpu().numpy().astype("float32")

# ============================================================
# Query patch features (for patch index search)
# ============================================================
def _resize_long_edge(img_bgr, long_edge=768):
    h, w = img_bgr.shape[:2]
    s = long_edge / float(max(h, w))
    if s >= 1.0:
        return img_bgr
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def _extract_patches_grid(img_bgr, patch_sizes=(256, 384, 512), stride_ratio=0.5,
                          max_patches=64, border_frac=0.02, roi_xyxy=None):
    H, W = img_bgr.shape[:2]
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(W, int(x2)); y2 = min(H, int(y2))
    else:
        x1, y1, x2, y2 = 0, 0, W, H

    patches = []
    crop = img_bgr[y1:y2, x1:x2]
    HH, WW = crop.shape[:2]

    for ps in patch_sizes:
        if min(HH, WW) < ps:
            continue
        stride = max(1, int(ps * stride_ratio))
        y0 = int(HH * border_frac); x0 = int(WW * border_frac)
        y1m = max(y0, HH - int(HH * border_frac) - ps)
        x1m = max(x0, WW - int(WW * border_frac) - ps)

        ys = list(range(y0, y1m + 1, stride)) if y1m >= y0 else [max(0, (HH - ps) // 2)]
        xs = list(range(x0, x1m + 1, stride)) if x1m >= x0 else [max(0, (WW - ps) // 2)]

        for yy in ys:
            for xx in xs:
                patch = crop[yy:yy + ps, xx:xx + ps]
                if patch.shape[0] == ps and patch.shape[1] == ps:
                    patches.append(patch)
                if len(patches) >= max_patches:
                    return patches

    if not patches:
        patches.append(crop)
    return patches

def get_query_patch_feats(model, mean, std, to_rgb, qimg_bgr,
                          patch_sizes=(256, 384, 512),
                          stride_ratio=0.5,
                          max_patches=64,
                          long_edge=1024,
                          batch_size=64):
    qimg = _resize_long_edge(qimg_bgr, long_edge=long_edge)

    # compute ROI on 512 input
    qx512 = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE, non_blocking=True)
    qfm = extract_featmap(model, qx512, FEAT_LEVEL)  # (1,C,Hf,Wf)
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

    patches = _extract_patches_grid(
        qimg, patch_sizes=patch_sizes, stride_ratio=stride_ratio,
        max_patches=max_patches, roi_xyxy=roi_xyxy
    )

    tensors = []
    for p in patches:
        p224 = cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA)
        if to_rgb:
            p224 = cv2.cvtColor(p224, cv2.COLOR_BGR2RGB)
        x = (p224.astype(np.float32) - mean) / std
        x = torch.from_numpy(x.transpose(2, 0, 1))
        tensors.append(x)

    feats_all = []
    with torch.no_grad():
        for st in range(0, len(tensors), batch_size):
            bt = torch.stack(tensors[st:st + batch_size], dim=0).to(DEVICE, non_blocking=True)
            fv = extract_feat(model, bt)  # (b,D) L2
            feats_all.append(fv.cpu())

    feats = torch.cat(feats_all, dim=0).numpy().astype("float32")
    return feats, len(patches)

# ============================================================
# Patch aggregation → image score
# ============================================================
def aggregate_patch_hits(patch_ids, patch_scores, patch_meta,
                         top_images=4000, topM=8, tau=0.15):
    meta_is_2d = isinstance(patch_meta, np.ndarray) and patch_meta.ndim == 2

    img_scores: Dict[int, List[float]] = {}
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

# ============================================================
# geom scoring core (your original logic)
# ============================================================
@torch.no_grad()
def compute_Ng0(q_desc, c_desc, margin=0.015):
    sim = q_desc @ c_desc.t()
    topv, topi = torch.topk(sim, k=2, dim=1)
    q_best  = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    return int(good.sum().item())

@torch.no_grad()
def geom_score_adaptive(
    q_desc, q_xy, c_desc, c_xy,
    margin=0.02, min_keep=8,
    bin_size=0.05, topM=6, topk_core=64,
    periodic_peak_thr=0.22,
    periodic_cover_topM_thr=0.70,
    periodic_cover_xy_thr=0.18,
    dbg=False
):
    sim = q_desc @ c_desc.t()

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

    qx = qg[:, 0]; qy = qg[:, 1]
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

    return float(core * (0.35 + 0.65 * cover_topM) * (0.5 + 0.5 * peak_ratio))

@torch.no_grad()
def geom_score_compatible(q_desc, q_xy, c_desc, c_xy,
                          margin=0.015, min_keep=6,
                          bin_size=0.05, topM=6, topk_core=64,
                          periodic_peak_thr=0.25,
                          periodic_cover_topM_thr=0.70,
                          periodic_cover_xy_thr=0.22,
                          tex_topk_core=128, tex_min_pairs=12,
                          tex_weight=0.35):
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
        return float(s)

    s_tex = texture_score(
        q_desc, q_xy, c_desc, c_xy,
        bin_size=bin_size, topM=topM,
        topk_core=tex_topk_core, min_pairs=tex_min_pairs
    )
    return float(s_tex * tex_weight)

# ============================================================
# Visualization (Grid)
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
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
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

        canvas[y:y + header, x:x + tile] = 0
        _put_text(canvas, labels[idx], org=(x + 8, y + 28), font_scale=0.7, thickness=2)
        canvas[y + header:y + header + tile, x:x + tile] = tiles[idx]

    cv2.imwrite(out_path, canvas)
    return out_path

# ============================================================
# Some scoring confidence (kept from your script)
# ============================================================
def global_confidence(global_rank, img_paths, topn=20):
    names = [os.path.basename(str(img_paths[i])) for i in global_rank[:topn]]
    prefix = [n.split('_')[0] if '_' in n else n[:4] for n in names]
    from collections import Counter
    c = Counter(prefix).most_common(1)[0][1]
    return c / max(1, len(prefix))

def get_full_rot_query_desc(model, mean, std, to_rgb, qimg, qimg_key: str):
    """Cache FULL rotations query desc/xy (only computed for hard case)."""
    global _q_full_cache
    if qimg_key in _q_full_cache:
        return _q_full_cache[qimg_key]

    q_desc_list_full, q_xy_list_full = [], []
    for ang in ROT_FULL:
        qimg_r = rotate_bgr(qimg, ang)
        qx = make_single_tensor_for_rerank(qimg_r, mean, std, to_rgb=to_rgb).to(DEVICE, non_blocking=True)
        q_fm = extract_featmap(model, qx, FEAT_LEVEL)
        q_desc, q_xy = select_query_patches(q_fm)
        q_desc_list_full.append(q_desc)
        q_xy_list_full.append(q_xy)

    _q_full_cache[qimg_key] = (q_desc_list_full, q_xy_list_full)
    return _q_full_cache[qimg_key]

# ============================================================
# Qt API
# ============================================================
def run_retrieval(query_path: str, out_dir: str, topk: int = TOPK_DEFAULT):
    """
    For Qt GUI:
      return (result_grid_path, top_items)
      top_items: [{"path": str, "score": float, "geom": float, "id": int}, ...]
    """
    ensure_dir(out_dir)
    g_index, p_index, img_paths, patch_meta, model, mean, std, to_rgb = get_ctx(force_reload=force_reload_flag)

    qimg = imread_unicode(query_path)
    qimg = crop_by_feat_energy_qt_safe(
        model, qimg, mean, std, to_rgb,
        prefer_level=FEAT_LEVEL
    )

    if qimg is None:
        raise FileNotFoundError(f"Query image not found: {query_path}")

    # -------- Global search
    qvec = get_query_global_feat(model, mean, std, to_rgb, qimg)
    _, gids = g_index.search(qvec, TOPG)
    global_rank = clean_rank(gids[0].tolist())

    # -------- Patch search
    q_patch_vecs, _ = get_query_patch_feats(
        model, mean, std, to_rgb, qimg,
        patch_sizes=(256, 384, 512),
        stride_ratio=0.5,
        max_patches=64,
        long_edge=1024,
        batch_size=64
    )

    D, I = p_index.search(q_patch_vecs, PATCH_TOPK_PER_QPATCH)
    D = D.astype(np.float32)

    patch_ids_all = I.reshape(-1).tolist()
    patch_scores_all = D.reshape(-1).tolist()

    patch_rank = aggregate_patch_hits(
        patch_ids_all, patch_scores_all, patch_meta, top_images=TOP_PATCH_IMAGES
    )
    patch_rank = clean_rank(patch_rank)

    # -------- RRF fusion (weighted)
    fused = clean_rank(rrf_fuse(global_rank, patch_rank, RRF_K))[:GEOM_TOPN]

    rrf_g = rank_to_rrf_score(global_rank, k=RRF_K)
    rrf_p = rank_to_rrf_score(patch_rank, k=RRF_K)

    conf_g = global_confidence(global_rank, img_paths, topn=20)
    w_g = 0.6 + 0.35 * conf_g
    w_p = 1.0 - w_g

    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    # -------- Precompute query patch descriptors (FAST 0° only)
    q_desc_list_fast, q_xy_list_fast = [], []
    for ang in ROT_FAST:
        qimg_r = rotate_bgr(qimg, ang)
        qx = make_single_tensor_for_rerank(qimg_r, mean, std, to_rgb=to_rgb).to(DEVICE, non_blocking=True)
        q_fm = extract_featmap(model, qx, FEAT_LEVEL)
        q_desc, q_xy = select_query_patches(q_fm)
        q_desc_list_fast.append(q_desc)
        q_xy_list_fast.append(q_xy)

    # -------- Candidate rerank (BATCH featmap forward)
    scored: List[Tuple[int, float]] = []
    fused_valid = [i for i in fused if i is not None and i >= 0]

    for st in range(0, len(fused_valid), CAND_BATCH):
        batch_ids = fused_valid[st:st + CAND_BATCH]

        batch_imgs = []
        real_ids = []
        for img_id in batch_ids:
            p = str(img_paths[img_id])
            cimg = imread_unicode(p)
            if cimg is None:
                continue
            batch_imgs.append(cimg)
            real_ids.append(img_id)

        if not batch_imgs:
            continue

        cx = make_batch_tensor_for_rerank(batch_imgs, mean, std, to_rgb=to_rgb).to(DEVICE, non_blocking=True)
        c_fm_b = extract_featmap(model, cx, FEAT_LEVEL)

        for bi, img_id in enumerate(real_ids):
            c_fm = c_fm_b[bi:bi + 1]
            c_desc, c_xy = select_candidate_patches(c_fm)

            # Stage 1: 0 degree only
            geom_best = 0.0
            for q_desc, q_xy in zip(q_desc_list_fast, q_xy_list_fast):
                s0 = geom_score_compatible(
                    q_desc, q_xy, c_desc, c_xy,
                    margin=MARGIN, min_keep=MIN_KEEP,
                    bin_size=BIN_SIZE, topM=TOPM, topk_core=TOPK_CORE,
                    periodic_peak_thr=PERIODIC_PEAK_THR,
                    periodic_cover_topM_thr=PERIODIC_COVER_TOPM_THR,
                    periodic_cover_xy_thr=PERIODIC_COVER_XY_THR,
                    tex_topk_core=128, tex_min_pairs=12,
                    tex_weight=0.35
                )
                if s0 > geom_best:
                    geom_best = s0

            # Stage 2: Only hard cases -> FULL rotations (cached)
            if geom_best < ROT_TRIGGER:
                q_desc_list_full, q_xy_list_full = get_full_rot_query_desc(
                    model, mean, std, to_rgb, qimg, qimg_key=query_path
                )
                for q_desc, q_xy in zip(q_desc_list_full, q_xy_list_full):
                    s = geom_score_compatible(
                        q_desc, q_xy, c_desc, c_xy,
                        margin=MARGIN, min_keep=MIN_KEEP,
                        bin_size=BIN_SIZE, topM=TOPM, topk_core=TOPK_CORE,
                        periodic_peak_thr=PERIODIC_PEAK_THR,
                        periodic_cover_topM_thr=PERIODIC_COVER_TOPM_THR,
                        periodic_cover_xy_thr=PERIODIC_COVER_XY_THR,
                        tex_topk_core=128, tex_min_pairs=12,
                        tex_weight=0.35
                    )
                    if s > geom_best:
                        geom_best = s

            if geom_best > 0:
                scored.append((img_id, geom_best))

    scored.sort(key=lambda x: x[1], reverse=True)

    # -------- Fuse geom into final score
    geom_vals = [s for _, s in scored]
    gmax = max(geom_vals) if geom_vals else 1.0
    gmin = min(geom_vals) if geom_vals else 0.0

    def norm_g(g):
        return (g - gmin) / (gmax - gmin + 1e-9)

    final = []
    for img_id, gs in scored:
        fs = final_rrf.get(img_id, 0.0) * (1.0 + BETA_GEOM * norm_g(gs))
        final.append((img_id, fs, gs))
    final.sort(key=lambda x: x[1], reverse=True)

    # normalize to 0~1
    scores = [fs for _, fs, _ in final] or [0.0]
    s_min = min(scores)
    s_max = max(scores) + 1e-9
    final_norm = [(img_id, (fs - s_min) / (s_max - s_min), gs) for img_id, fs, gs in final]
    final_norm.sort(key=lambda x: x[1], reverse=True)

    top = final_norm[:int(topk)]

    top_items = []
    for img_id, fs, gs in top:
        top_items.append({
            "path": str(img_paths[img_id]),
            "score": float(fs),
            "geom": float(gs),
            "id": int(img_id),
        })

    # ---- output grid
    imgs, scs = [], []
    for it in top_items:
        imgs.append(imread_unicode(it["path"]))
        scs.append(it["score"])

    result_grid_path = os.path.join(out_dir, "result_grid.png")
    visualize_grid(qimg, imgs, scs, result_grid_path, tile=320)

    return result_grid_path, top_items

# ============================================================
# Optional CLI test
# ============================================================
if __name__ == "__main__":
    # Example manual test (edit your query/out_dir):
    query = r"D:\zhanlan\search_vis\q_1770274344_crop_raw.png"

    out_dir = r"D:\zhanlan\search_vis"
    grid_path, items = run_retrieval(query, out_dir, topk=12)
    print("Saved:", grid_path)
    for i, it in enumerate(items, 1):
        print(i, it["score"], it["geom"], it["path"])
