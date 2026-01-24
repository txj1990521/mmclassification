#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from pathlib import Path
from typing import List, Tuple, Dict, Any

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# ============================================================
# CONFIG
# ============================================================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

QUERY_IMG = r"D:\zhanlan\qurrey_data\111a.jpg"

INDEX_DIR = r"D:\zhanlan\faiss_database_hybrid"
GLOBAL_INDEX = os.path.join(INDEX_DIR, "global.index")
PATCH_INDEX  = os.path.join(INDEX_DIR, "patch.index")
GLOBAL_META  = os.path.join(INDEX_DIR, "global_img_paths.npy")
PATCH_META   = os.path.join(INDEX_DIR, "patch_meta.npy")

OUT_VIZ = r"D:\zhanlan\topk_viz_rmac\topk_hybrid_rrf_geom.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Retrieval budget ----------------
TOPK = 10

# global recall
TOPG = 800

# patch recall
QPATCH = 12                  # query 切多少 patch（局部图建议 8~16）
PATCH_SEARCH_K = 800         # 每个 query-patch 检索多少个 patch 向量
PATCH_AGG_TOPM = 6000        # patch 聚合成图片后保留多少图片参与融合

# RRF
RRF_K = 60
FUSED_TOPN = 300             # RRF 融合后只保留 topN 给 geom rerank

# ---------------- Geom rerank ----------------
# 这里默认用 ORB+RANSAC（不依赖颜色，尺度/旋转较稳）
GEOM_ENABLE = True
ORB_NFEATURES = 4000
RANSAC_THRESH = 5.0
MIN_GOOD_MATCH = 15
MIN_INLIERS = 10

# ---------------- Patch slicing ----------------
# 你说库图 3K 左右：对库做 patch 是建库阶段的事，你已经做了
# Query 这里也做 patch
PATCH_SHORT = 768            # query patch 前先把短边缩放到 768（局部图会更稳）
PATCH_SIZES = [256, 320]     # 两种尺度切 patch，兼顾细纹/大花
PATCH_STRIDE_RATIO = 0.50    # stride = size * ratio

# ---------------- Visualization ----------------
TILE = 320

# ============================================================
# Utils
# ============================================================
def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(out_path: str, img_bgr: np.ndarray):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(out_path).suffix.lower()
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(out_path)

def _put_text(img, text, org=(8, 26), font_scale=0.8, thickness=2):
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

def make_topk_viz(query_path: str, top_paths: list, top_scores: list, out_path: str, tile: int = 320):
    qimg = imread_unicode(query_path)
    imgs = [_fit_square(qimg, tile)]
    labels = ["QUERY"]

    for i, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        img = imread_unicode(str(p))
        if img is None:
            img = np.zeros((tile, tile, 3), np.uint8)
        imgs.append(_fit_square(img, tile))
        labels.append(f"#{i}  {float(s):.4f}")

    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    gap = 8
    header = 40

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)
        canvas[y + header:y + header + tile, x:x + tile] = imgs[idx]
        _put_text(canvas, labels[idx], org=(x + 8, y + 26), font_scale=0.7, thickness=2)

    imwrite_unicode(out_path, canvas)
    return out_path

def resize_short_edge(img_bgr: np.ndarray, short: int):
    h, w = img_bgr.shape[:2]
    if min(h, w) == short:
        return img_bgr
    s = short / float(min(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

# ============================================================
# Model
# ============================================================
@torch.no_grad()
def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1,1,3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1,1,3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb

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

def to_tensor_from_bgr(img_bgr: np.ndarray, mean, std, to_rgb: bool, size=224):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = img[:, :, ::-1].copy()
    x = img.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

# ============================================================
# Query: global embedding
# ============================================================
@torch.no_grad()
def query_global_vec(model, mean, std, to_rgb, qimg_bgr):
    x = to_tensor_from_bgr(qimg_bgr, mean, std, to_rgb, size=224).unsqueeze(0).to(DEVICE)
    v = extract_backbone_last(model, x).cpu().numpy().astype("float32")
    return v

# ============================================================
# Query: patches embedding (multi-scale sliding window)
# ============================================================
def extract_query_patches(qimg_bgr: np.ndarray) -> List[np.ndarray]:
    q = resize_short_edge(qimg_bgr, PATCH_SHORT)
    H, W = q.shape[:2]

    patches = []
    for sz in PATCH_SIZES:
        stride = int(sz * PATCH_STRIDE_RATIO)
        if H < sz or W < sz:
            # too small: fallback center crop
            y1 = max(0, (H - sz) // 2)
            x1 = max(0, (W - sz) // 2)
            y2 = min(H, y1 + sz)
            x2 = min(W, x1 + sz)
            crop = q[y1:y2, x1:x2]
            patches.append(crop)
            continue

        for y in range(0, H - sz + 1, stride):
            for x in range(0, W - sz + 1, stride):
                patches.append(q[y:y+sz, x:x+sz])

    # 采样到 QPATCH 个（稳定）
    if len(patches) > QPATCH:
        idx = np.linspace(0, len(patches)-1, QPATCH).astype(int).tolist()
        patches = [patches[i] for i in idx]
    return patches[:QPATCH]

@torch.no_grad()
def query_patch_vecs(model, mean, std, to_rgb, qimg_bgr) -> np.ndarray:
    ps = extract_query_patches(qimg_bgr)
    if not ps:
        ps = [qimg_bgr]
    bt = torch.stack([to_tensor_from_bgr(p, mean, std, to_rgb, size=224) for p in ps], dim=0).to(DEVICE)
    fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
    return fv  # (QPATCH, D)

# ============================================================
# Patch meta parsing
# patch_meta 你建库脚本里一般存：
#   patch_meta[i] = (img_id, x1, y1, x2, y2)  或至少 (img_id, ...)
# 我这里做兼容：只要 patch_meta[i][0] 是 img_id 就行
# ============================================================
def patch_to_img_id(patch_meta: Any, patch_id: int) -> int:
    item = patch_meta[patch_id]
    # item 可能是 np.ndarray / tuple / list
    if isinstance(item, (tuple, list)):
        return int(item[0])
    if isinstance(item, np.ndarray):
        return int(item.flat[0])
    # 最后兜底
    return int(item)

def aggregate_patch_results_to_images(patch_meta, patch_ids: np.ndarray, patch_scores: np.ndarray, topm: int):
    best: Dict[int, float] = {}
    for pid, s in zip(patch_ids.tolist(), patch_scores.tolist()):
        if pid < 0:
            continue
        img_id = patch_to_img_id(patch_meta, int(pid))
        ss = float(s)
        if (img_id not in best) or (ss > best[img_id]):
            best[img_id] = ss
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    img_rank = [i for i,_ in ranked]
    return img_rank[:topm], best

# ============================================================
# RRF
# ============================================================
def rrf_fuse(rankA: List[int], rankB: List[int], k: int = 60) -> Tuple[List[int], Dict[int, float]]:
    score: Dict[int, float] = {}
    for r, i in enumerate(rankA, 1):
        score[i] = score.get(i, 0.0) + 1.0 / (k + r)
    for r, i in enumerate(rankB, 1):
        score[i] = score.get(i, 0.0) + 1.0 / (k + r)
    fused = [i for i,_ in sorted(score.items(), key=lambda x: x[1], reverse=True)]
    return fused, score

# ============================================================
# Geom rerank: ORB + RANSAC homography inlier ratio
# ============================================================
def geom_score_orb(qimg_bgr: np.ndarray, cimg_bgr: np.ndarray,
                   orb_nfeatures=4000, ransac_thresh=5.0,
                   min_good=15, min_inliers=10) -> float:
    gq = cv2.cvtColor(qimg_bgr, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cimg_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=int(orb_nfeatures))
    kq, dq = orb.detectAndCompute(gq, None)
    kc, dc = orb.detectAndCompute(gc, None)
    if dq is None or dc is None or len(kq) < 10 or len(kc) < 10:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(dq, dc, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_good:
        return 0.0

    pts_q = np.float32([kq[m.queryIdx].pt for m in good])
    pts_c = np.float32([kc[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_q, pts_c, cv2.RANSAC, float(ransac_thresh))
    if mask is None:
        return 0.0

    inliers = int(mask.sum())
    if inliers < min_inliers:
        return 0.0

    return float(inliers) / float(len(good) + 1e-6)

# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] device:", DEVICE)

    # load index + meta
    g_index = faiss.read_index(GLOBAL_INDEX)
    p_index = faiss.read_index(PATCH_INDEX)
    global_paths = np.load(GLOBAL_META, allow_pickle=True)
    patch_meta = np.load(PATCH_META, allow_pickle=True)

    print("[INFO] global ntotal:", g_index.ntotal, "patch ntotal:", p_index.ntotal)
    print("[INFO] global_meta:", len(global_paths), "patch_meta:", len(patch_meta))

    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    qimg = imread_unicode(QUERY_IMG)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")

    # ---------- 1) Global search ----------
    qg = query_global_vec(model, mean, std, to_rgb, qimg)
    g_scores, g_ids = g_index.search(qg, min(int(TOPG), int(g_index.ntotal)))
    g_rank = [int(i) for i in g_ids[0].tolist() if i >= 0]

    print("[INFO] global rank size:", len(g_rank), "top5:", g_rank[:5])

    # ---------- 2) Patch search ----------
    qp = query_patch_vecs(model, mean, std, to_rgb, qimg)  # (QPATCH, D)
    all_patch_ids = []
    all_patch_scores = []

    # patch index 一般是 IP / L2；你建的是 FlatIP，所以 score 越大越近
    for i in range(qp.shape[0]):
        scores, ids = p_index.search(qp[i:i+1], min(int(PATCH_SEARCH_K), int(p_index.ntotal)))
        all_patch_ids.append(ids[0])
        all_patch_scores.append(scores[0])

    all_patch_ids = np.concatenate(all_patch_ids, axis=0)
    all_patch_scores = np.concatenate(all_patch_scores, axis=0)

    patch_img_rank, patch_best = aggregate_patch_results_to_images(
        patch_meta, all_patch_ids, all_patch_scores, topm=PATCH_AGG_TOPM
    )

    print("[INFO] patch->img rank size:", len(patch_img_rank), "top5:", patch_img_rank[:5])

    # ---------- 3) RRF fuse ----------
    fused_rank, fused_rrf_score = rrf_fuse(g_rank, patch_img_rank, k=RRF_K)
    fused_rank = fused_rank[:min(FUSED_TOPN, len(fused_rank))]
    print("[INFO] fused top:", fused_rank[:10])

    # ---------- 4) Geom rerank ----------
    final_list = []
    if GEOM_ENABLE:
        for img_id in fused_rank:
            if img_id < 0 or img_id >= len(global_paths):
                continue
            p = str(global_paths[img_id])
            cimg = imread_unicode(p)
            if cimg is None:
                continue
            gs = geom_score_orb(
                qimg, cimg,
                orb_nfeatures=ORB_NFEATURES,
                ransac_thresh=RANSAC_THRESH,
                min_good=MIN_GOOD_MATCH,
                min_inliers=MIN_INLIERS
            )
            # 最终分：RRF + geom（不调场景权重：只做乘法约束）
            s = float(fused_rrf_score.get(img_id, 0.0)) * (0.20 + 0.80 * gs)
            final_list.append((img_id, s, gs))

        final_list.sort(key=lambda x: x[1], reverse=True)
    else:
        final_list = [(i, float(fused_rrf_score.get(i, 0.0)), 0.0) for i in fused_rank]

    top = final_list[:TOPK]
    top_paths = [str(global_paths[i]) for i,_,_ in top]
    top_scores = [float(s) for _,s,_ in top]

    print("\n===== TOPK (RRF + GEOM) =====")
    for r, (img_id, s, gs) in enumerate(top, 1):
        print(f"{r:02d} score={s:.6f} geom={gs:.4f}  {global_paths[img_id]}")

    out = make_topk_viz(QUERY_IMG, top_paths, top_scores, OUT_VIZ, tile=TILE)
    print("[VIZ] saved ->", out)

if __name__ == "__main__":
    main()
