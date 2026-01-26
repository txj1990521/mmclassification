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
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

QUERY_IMG = r"D:\zhanlan\qurrey_data\S8987B2-90#a.jpg"


INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid"
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
import numpy as np
import torch
import torch.nn.functional as F

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

@torch.no_grad()
def geom_score_compatible(q_desc, q_xy, c_desc, c_xy,
                          # struct params
                          margin=0.015, min_keep=6,
                          bin_size=0.05, topM=6, topk_core=64,
                          periodic_peak_thr=0.25,
                          periodic_cover_topM_thr=0.70,
                          periodic_cover_xy_thr=0.22,
                          # texture params
                          tex_topk_core=128, tex_min_pairs=12,
                          tex_weight=0.35):
    """
    返回一个最终的“几何相关”分数（兼容结构/纹理）
    - 结构模式：用你的 geom_score_adaptive
    - 纹理模式：用 texture_score，并乘 tex_weight 降权（避免误抬）
    """
    Ng0 = compute_Ng0(q_desc, c_desc, margin=margin)

    # 结构模式
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

    # 纹理模式（降权）
    s_tex = texture_score(
        q_desc, q_xy, c_desc, c_xy,
        bin_size=bin_size, topM=topM,
        topk_core=tex_topk_core, min_pairs=tex_min_pairs
    )
    return float(s_tex * tex_weight)

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

    # 纹理：更看重 cover_topM（集中在少数峰）
    return float(core * (0.35 + 0.65 * cover_topM) * (0.5 + 0.5 * peak_ratio))


@torch.no_grad()
def geom_score_with_stats(q_desc, q_xy, c_desc, c_xy, **kw):
    """
    返回:
      score_struct: float  # 你的原 geom_score_adaptive 分数
      stats: dict          # 纹理门控需要的统计量
    """
    # --- 复制你 geom_score_adaptive 的前半段，直到算出 Ng0 ---
    margin   = kw.get("margin", 0.015)
    min_keep = kw.get("min_keep", 6)

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

    # --- 结构分数：仍然调用你的原函数（完全不改它） ---
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
import numpy as np
import torch

@torch.no_grad()
def geom_score_adaptive(
    q_desc, q_xy, c_desc, c_xy,
    margin=0.02, min_keep=8,
    bin_size=4.0, topM=6, topk_core=64,
    periodic_peak_thr=0.22,
    periodic_cover_topM_thr=0.70,   # 注意：这是 cover_topM 的阈值（不是空间覆盖）
    periodic_cover_xy_thr=0.18,     # 新增：空间覆盖率阈值（推荐）
    dbg=False
):
    """
    q_desc: (Nq,D) torch float, L2
    q_xy:   (Nq,2) torch float, 坐标建议是“同一尺度”（最好归一化到[0,1]）
    c_desc: (Nc,D)
    c_xy:   (Nc,2)

    返回：float score
    """

    sim = q_desc @ c_desc.t()  # (Nq, Nc)

    # 1) q->c best and 2nd best
    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best  = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    # 2) c->q best (for mutual)
    c_best = torch.argmax(sim, dim=0)  # (Nc,)

    # 3) mutual NN
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    # 4) ratio/margin test
    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng0 = int(good.sum().item())
    if Ng0 < min_keep:
        if dbg:
            print(f"[DBG] early return: Ng0={Ng0} < min_keep={min_keep} (margin={margin})")
        return 0.0

    good_idx = torch.nonzero(good, as_tuple=False).squeeze(1)

    # 5) 取 topk_core 个最强匹配
    K = min(topk_core, good_idx.numel())
    sel = torch.topk(q_bestv[good_idx], k=K, largest=True).indices
    good_idx = good_idx[sel]

    mi = q_best[good_idx]   # matched candidate indices
    qg = q_xy[good_idx]     # (Ng,2)
    cg = c_xy[mi]           # (Ng,2)
    Ng = int(good_idx.numel())
    if Ng < min_keep:
        if dbg:
            print(f"[DBG] early return: Ng={Ng} < min_keep={min_keep}")
        return 0.0

    # ---- shape consistency: pairwise distance ratio ----
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

    # ---- displacement periodicity (multi-peak) ----
    d = cg - qg  # (Ng,2) 确保 q_xy/c_xy 同坐标系，否则这里没意义

    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(Ng)
    m = min(topM, cnt.numel())
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(Ng)

    # ---- spatial coverage on query side (推荐) ----
    # 这个 cover_xy 是：匹配点在 query 上覆盖的范围（归一化后更稳）
    # 如果你的 q_xy 是像素坐标，也可以先除以图像宽高再传入
    qx = qg[:, 0]
    qy = qg[:, 1]
    cover_x = float((qx.max() - qx.min()).item())
    cover_y = float((qy.max() - qy.min()).item())
    cover_xy = min(cover_x, cover_y)  # 取更保守方向

    # ---- core similarity (用筛过的 good_idx) ----
    core = float(q_bestv[good_idx].mean().item())

    # Ng scale
    ng_scale = float(min(1.0, Ng / 32.0))

    # 周期性判别：主峰小 + 多峰集中 + 覆盖范围还小（典型周期假匹配）
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

# def _extract_patches_grid(img_bgr, patch_sizes=(256, 384, 512), stride_ratio=0.5,
#                           max_patches=64, border_frac=0.02):
#     """
#     多尺度网格切 patch，覆盖局部；返回 list[np.ndarray(BGR)]。
#     - stride_ratio=0.5 => 重叠一半
#     - max_patches 防止 query 太大切太多
#     """
#     H, W = img_bgr.shape[:2]
#     patches = []
#     for ps in patch_sizes:
#         if min(H, W) < ps:
#             continue
#         stride = max(1, int(ps * stride_ratio))
#         y0 = int(H * border_frac)
#         x0 = int(W * border_frac)
#         y1 = max(y0, H - int(H * border_frac) - ps)
#         x1 = max(x0, W - int(W * border_frac) - ps)
#
#         ys = list(range(y0, y1 + 1, stride)) if y1 >= y0 else [max(0, (H-ps)//2)]
#         xs = list(range(x0, x1 + 1, stride)) if x1 >= x0 else [max(0, (W-ps)//2)]
#
#         for y in ys:
#             for x in xs:
#                 patch = img_bgr[y:y+ps, x:x+ps]
#                 if patch.shape[0] == ps and patch.shape[1] == ps:
#                     patches.append(patch)
#                 if len(patches) >= max_patches:
#                     return patches
#     # 兜底：至少给中心 patch
#     if len(patches) == 0:
#         s = min(H, W)
#         y = (H - s) // 2
#         x = (W - s) // 2
#         patch = img_bgr[y:y+s, x:x+s]
#         patches.append(patch)
#     return patches
#
#


def _extract_patches_grid(img_bgr, patch_sizes=(256,384,512), stride_ratio=0.5,
                          max_patches=64, border_frac=0.02, roi_xyxy=None):
    H, W = img_bgr.shape[:2]
    if roi_xyxy is not None:
        x1,y1,x2,y2 = roi_xyxy
        x1 = max(0,int(x1)); y1=max(0,int(y1)); x2=min(W,int(x2)); y2=min(H,int(y2))
    else:
        x1,y1,x2,y2 = 0,0,W,H

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

        ys = list(range(y0, y1m + 1, stride)) if y1m >= y0 else [max(0, (HH-ps)//2)]
        xs = list(range(x0, x1m + 1, stride)) if x1m >= x0 else [max(0, (WW-ps)//2)]

        for yy in ys:
            for xx in xs:
                patch = crop[yy:yy+ps, xx:xx+ps]
                if patch.shape[0]==ps and patch.shape[1]==ps:
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
    """
    1) (可选)把 query 缩到一个合理长边，避免 3K 切 patch 爆炸
    2) 多尺度网格切 patch
    3) 每个 patch resize 到 224 -> backbone -> L2
    return: feats_np (P,D) float32, patches_count
    """
    qimg = _resize_long_edge(qimg_bgr, long_edge=long_edge)

    # 先在512输入上算roi（featmap坐标）
    qx512 = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    qfm = extract_featmap(model, qx512, FEAT_LEVEL)  # (1,C,Hf,Wf)
    roi_f = energy_roi_box(qfm[0], frac=0.18)  # (x1,y1,x2,y2) in featmap coords

    # 把featmap ROI映射到512，再映射回原图
    if roi_f is not None:
        _, _, Hf, Wf = qfm.shape
        x1, y1, x2, y2 = roi_f
        x1 = x1 / (Wf - 1);
        x2 = x2 / (Wf - 1)
        y1 = y1 / (Hf - 1);
        y2 = y2 / (Hf - 1)
        H, W = qimg.shape[:2]
        roi_xyxy = (x1 * W, y1 * H, x2 * W, y2 * H)
    else:
        roi_xyxy = None

    patches = _extract_patches_grid(
        qimg, patch_sizes=patch_sizes, stride_ratio=stride_ratio,
        max_patches=max_patches,roi_xyxy=roi_xyxy
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
            bt = torch.stack(tensors[st:st+batch_size], dim=0).to(DEVICE)
            fv = extract_feat(model, bt)  # (b,D) already L2
            feats_all.append(fv.cpu())

    feats = torch.cat(feats_all, dim=0).numpy().astype("float32")
    return feats, len(patches)

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


# ============================================================
# Main
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    g_index = faiss.read_index(GLOBAL_INDEX)
    p_index = faiss.read_index(PATCH_INDEX)
    set_faiss_nprobe(g_index, 64)
    set_faiss_nprobe(p_index, 64)

    img_paths = np.load(GLOBAL_META, allow_pickle=True)
    patch_meta = np.load(PATCH_META, allow_pickle=True)

    model, mean, std, to_rgb = build_model(CONFIG, CKPT)

    qimg = imread_unicode(QUERY_IMG)
    qvec = get_query_global_feat(model, mean, std, to_rgb, qimg)

    # -------- Global search
    _, gids = g_index.search(qvec, TOPG)
    global_rank = gids[0].tolist()

    # -------- Patch search (FIXED)
    q_patch_vecs, n_qpatch = get_query_patch_feats(
        model, mean, std, to_rgb, qimg,
        patch_sizes=(256, 384, 512),  # 你可以先用这三个尺度
        stride_ratio=0.5,
        max_patches=64,
        long_edge=1024,  # query 很大就缩一下再切
        batch_size=64
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

    # 聚合所有 patch hit
    patch_ids_all = I.reshape(-1).tolist()
    patch_scores_all = D.reshape(-1).tolist()

    patch_rank = aggregate_patch_hits(
        patch_ids_all, patch_scores_all, patch_meta, top_images=TOP_PATCH_IMAGES
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
    # -------- Deep geom rerank (replace ORB)
    qx = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    q_fm = extract_featmap(model, qx, FEAT_LEVEL)
    q_desc, q_xy = select_query_patches(q_fm)  # 你旧脚本里那套 ROI+full 混合即可

    scored = []


    for img_id in fused:
        cimg = imread_unicode(img_paths[img_id])
        if cimg is None:
            continue
        cx = make_single_tensor_for_rerank(cimg, mean, std, to_rgb=to_rgb).to(DEVICE)
        c_fm = extract_featmap(model, cx, FEAT_LEVEL)
        c_desc, c_xy = select_candidate_patches(c_fm)

        geom_best = 0.0
        for q_desc, q_xy in zip(q_desc_list, q_xy_list):
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
        scored.append((img_id, geom_best))
    scored = [(i, s) for (i, s) in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    geom_vals = [s for _, s in scored]
    gmax = max(geom_vals) if geom_vals else 1.0
    gmin = min(geom_vals) if geom_vals else 0.0

    def norm_g(g):
        return (g - gmin) / (gmax - gmin + 1e-9)

    beta = 0.3  # 0.1~0.5 之间先试
    final = []
    for img_id, gs in scored:
        fs = final_rrf.get(img_id, 0.0) * (1.0 + beta * norm_g(gs))
        final.append((img_id, fs, gs))
    final.sort(key=lambda x: x[1], reverse=True)
    # final: [(img_id, fs, gs), ...]

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

    out = os.path.join(OUT_DIR, "result_grid.png")
    visualize_grid(qimg, imgs, scores, out, tile=320)

    print("Top results:")
    for r,(img_id, fs, gs) in enumerate(top,1):
        print(f"{r:02d}  score={fs:.3f}  {img_paths[img_id]}")
    print("Saved:", out)

if __name__ == "__main__":
    main()
