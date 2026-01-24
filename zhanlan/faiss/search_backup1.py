#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coarse-to-Fine Image Retrieval
Stage-1: Old global feature FAISS index for coarse recall (multi-view query).
Stage-2: Intermediate feature map patch matching (R-MAC-like / patch max match) for rerank.
Output: TOPK list + visualization mosaic.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS


# =========================
# CONFIG: 只改这里
# =========================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"
QUERY_IMG = r"D:\zhanlan\qurrey_data\111a.jpg"

# 旧的全局特征库（最后层+GAP）对应的 FAISS index
OUT_INDEX = r"D:\zhanlan\faiss.index"
OUT_META = r"D:\zhanlan\faiss_paths.npy"

TOPK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 输出拼图
OUT_VIZ = r"D:\zhanlan\topk_viz_rmac_backup.jpg"
TILE = 320

# ---------- Stage-1 粗召回 ----------
COARSE_K = 300               # 最终粗候选数（融合所有 view 后截断）
COARSE_BATCH = 64            # query 多视角提特征的 batch
PER_VIEW_SEARCH_K = 200      # 每个 view 在旧 index 里搜索多少候选

# ---------- Query 多视角 ----------
ROT_DEGS = [-30, -15, 0, 15, 30]
FIVE_CROP = True
CROP_SIZE = 224
RESIZE_SHORT = 256

# ---------- R-MAC / Patch-match 参数 ----------
FEAT_LEVEL = -2              # backbone 输出的第几层（-2 常用于 C4）
RMAC_LEVELS = 3              # 仅保留（你原来有 region 实现，但当前主流程用的是 patch match）
RMAC_INPUT_SHORT = 512       # 做 rerank 时将短边 resize 到更大，保结构

# Patch 选择与匹配
KEEP_PATCHES = 512
PATCH_MATCH_TOPK = 64


# =========================
# utils: 中文路径 imread / imwrite
# =========================
def imread_unicode(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(out_path: str, img_bgr: np.ndarray) -> None:
    ext = Path(out_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise ValueError("out_path extension should be an image format like .jpg/.png")
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(out_path)


# =========================
# preprocess helpers
# =========================
def resize_short_edge(img_rgb: np.ndarray, short: int = 256) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def rotate_bound(img_rgb: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(
        img_rgb, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )


def crop_center(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]

    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img_rgb[y1:y1 + size, x1:x1 + size]


def five_crop(img_rgb: np.ndarray, size: int = 224) -> list[np.ndarray]:
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]

    tl = img_rgb[0:size, 0:size]
    tr = img_rgb[0:size, w - size:w]
    bl = img_rgb[h - size:h, 0:size]
    br = img_rgb[h - size:h, w - size:w]
    cc = crop_center(img_rgb, size)
    return [cc, tl, tr, bl, br]


def pad_to_square(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(
        img_rgb, top, bottom, left, right,
        borderType=cv2.BORDER_REFLECT101
    )


def to_tensor_from_rgb(img_rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


# =========================
# model
# =========================
@torch.no_grad()
def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1, 1, 3)
    std = np.array(dp.get("std", [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb


@torch.no_grad()
def extract_global_feat_for_coarse(model, batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    旧 index 对应特征：backbone 最后一层 -> GAP -> L2
    """
    feat_map = model.backbone(batch_tensor)
    if isinstance(feat_map, (tuple, list)):
        feat_map = feat_map[-1]
    feat = feat_map.mean(dim=(2, 3))  # GAP
    feat = F.normalize(feat, p=2, dim=1)
    return feat


@torch.no_grad()
def extract_featmap(model, batch_tensor: torch.Tensor, prefer_level: int) -> torch.Tensor:
    """
    返回 (B,C,H,W) 的特征图。
    兼容 backbone 输出：Tensor / tuple(list) / dict
    """
    out = model.backbone(batch_tensor)

    # dict: 取常见 key 或最后一个 value
    if isinstance(out, dict):
        if "feat" in out:
            out = out["feat"]
        elif "features" in out:
            out = out["features"]
        else:
            out = list(out.values())[-1]

    # tuple/list: clamp prefer_level
    if isinstance(out, (tuple, list)):
        n = len(out)
        lvl = prefer_level
        if lvl < -n:
            lvl = -n
        if lvl > n - 1:
            lvl = n - 1
        feat = out[lvl]

        if not hasattr(extract_featmap, "_printed"):
            print(f"[DEBUG] backbone returns {type(out).__name__} len={n}")
            for i, t in enumerate(out):
                if isinstance(t, torch.Tensor):
                    print(f" [{i}] shape={tuple(t.shape)}")
                else:
                    print(f" [{i}] type={type(t)}")
            print(f"[DEBUG] selected level={lvl} (prefer {prefer_level})")
            extract_featmap._printed = True

        return feat

    if isinstance(out, torch.Tensor):
        if not hasattr(extract_featmap, "_printed"):
            print(f"[DEBUG] backbone returns Tensor shape={tuple(out.shape)}")
            extract_featmap._printed = True
        return out

    raise TypeError(f"Unsupported backbone output type: {type(out)}")


# =========================
# Query views (coarse stage)
# =========================
@torch.no_grad()
def make_query_views_for_coarse(
    img_bgr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
) -> list[torch.Tensor]:
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = resize_short_edge(img_rgb, RESIZE_SHORT)

    views: list[torch.Tensor] = []
    for deg in ROT_DEGS:
        rotated = rotate_bound(img_rgb, deg)
        crops = five_crop(rotated, CROP_SIZE) if FIVE_CROP else [crop_center(rotated, CROP_SIZE)]
        for c in crops:
            views.append(to_tensor_from_rgb(c, mean, std))
    return views


# =========================
# Single tensor (rerank stage)
# =========================
@torch.no_grad()
def make_single_tensor_for_rerank(
    img_bgr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
) -> torch.Tensor:
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = pad_to_square(img_rgb)
    img_rgb = cv2.resize(img_rgb, (RMAC_INPUT_SHORT, RMAC_INPUT_SHORT), interpolation=cv2.INTER_LINEAR)

    x = to_tensor_from_rgb(img_rgb, mean, std)
    return x.unsqueeze(0)  # (1,3,S,S)


# =========================
# Patch match (rerank stage)
# =========================
@torch.no_grad()
def featmap_to_patches(feat_map: torch.Tensor) -> list[torch.Tensor]:
    """
    feat_map: (B,C,H,W)
    return: list length B, each (P,C) normalized, P=H*W
    """
    if feat_map.dim() != 4:
        raise ValueError("feat_map should be (B,C,H,W)")
    B, C, H, W = feat_map.shape
    x = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
    x = F.normalize(x, p=2, dim=2)
    return [x[i] for i in range(B)]


@torch.no_grad()
def select_informative_patches(patches: torch.Tensor, keep: int = 512) -> torch.Tensor:
    """
    patches: (P,C) normalized
    用“均值绝对值”作为信息量 proxy，取前 keep 个 patch
    """
    score = patches.abs().mean(dim=1)  # (P,)
    keep = min(keep, patches.shape[0])
    idx = torch.topk(score, k=keep, largest=True).indices
    return patches[idx]


@torch.no_grad()
def patch_match_score(q_patches: torch.Tensor, c_patches: torch.Tensor, topk: int = 64) -> float:
    """
    q_patches: (Pq,C) normalized
    c_patches: (Pc,C) normalized
    计算：对每个 query patch，找 candidate 中最相似 patch；再取 topk 个平均
    """
    sim = q_patches @ c_patches.t()              # (Pq, Pc)
    best_each_q = sim.max(dim=1).values          # (Pq,)
    k = min(topk, best_each_q.numel())
    return float(torch.topk(best_each_q, k=k, largest=True).values.mean().item())


# =========================
# TopK visualization
# =========================
def _put_text(img, text, org=(8, 26), font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _fit_square(img_bgr: np.ndarray, tile: int = 320) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((tile, tile, 3), dtype=np.uint8)

    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_topk_viz(query_path: str, top_paths: list[str], top_scores: list[float], out_path: str, tile: int = 320) -> str:
    qimg = imread_unicode(query_path)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image for viz: {query_path}")

    imgs = [_fit_square(qimg, tile)]
    labels = ["QUERY"]

    for i, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        img = imread_unicode(str(p))
        if img is None:
            imgs.append(np.zeros((tile, tile, 3), dtype=np.uint8))
            labels.append(f"#{i} {float(s):.4f}\n(read fail)")
        else:
            imgs.append(_fit_square(img, tile))
            labels.append(f"#{i} {float(s):.4f}")

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
        for li, line in enumerate(labels[idx].split("\n")):
            _put_text(canvas, line, org=(x + 8, y + 26 + li * 22), font_scale=0.7, thickness=2)

    imwrite_unicode(out_path, canvas)
    return out_path


# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)

    # load old faiss + meta
    index = faiss.read_index(OUT_INDEX)
    kept_paths = np.load(OUT_META, allow_pickle=True)
    print(f"[INFO] Loaded coarse index ntotal={index.ntotal}, meta={len(kept_paths)}")

    # load model
    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    # read query
    qimg = imread_unicode(QUERY_IMG)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")

    # ---------- Stage-1: coarse recall ----------
    views = make_query_views_for_coarse(qimg, mean, std, to_rgb=to_rgb)
    print(f"[INFO] coarse views = {len(views)}")

    best_score: dict[int, float] = {}  # id -> best score across all views
    for st in range(0, len(views), COARSE_BATCH):
        bt = torch.stack(views[st:st + COARSE_BATCH], dim=0).to(DEVICE)
        feats = extract_global_feat_for_coarse(model, bt).cpu().numpy().astype("float32")

        scores, ids = index.search(feats, PER_VIEW_SEARCH_K)

        for row_s, row_i in zip(scores, ids):
            for s, idx in zip(row_s, row_i):
                if idx < 0:
                    continue
                idx = int(idx)
                s = float(s)
                prev = best_score.get(idx)
                if prev is None or s > prev:
                    best_score[idx] = s

    coarse_items = sorted(best_score.items(), key=lambda x: x[1], reverse=True)[:COARSE_K]
    coarse_ids = [i for i, _ in coarse_items]
    coarse_paths = [str(kept_paths[i]) for i in coarse_ids]
    print(f"[INFO] coarse candidates = {len(coarse_paths)}")

    # ---------- Stage-2: rerank via patch matching ----------
    # query patches
    qx = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    q_fm = extract_featmap(model, qx, FEAT_LEVEL)
    q_patches = featmap_to_patches(q_fm)[0]
    q_patches = select_informative_patches(q_patches, keep=KEEP_PATCHES)

    # load candidates
    cand_tensors: list[torch.Tensor] = []
    valid_paths: list[str] = []
    for p in coarse_paths:
        img = imread_unicode(p)
        if img is None:
            continue
        cand_tensors.append(make_single_tensor_for_rerank(img, mean, std, to_rgb=to_rgb))
        valid_paths.append(p)

    if not cand_tensors:
        raise RuntimeError("All coarse candidate images failed to read.")

    sims: list[float] = []
    bs2 = 8
    for st in range(0, len(cand_tensors), bs2):
        bt = torch.cat(cand_tensors[st:st + bs2], dim=0).to(DEVICE)
        fm = extract_featmap(model, bt, FEAT_LEVEL)
        c_patches_list = featmap_to_patches(fm)

        for c_patches in c_patches_list:
            s = patch_match_score(q_patches, c_patches, topk=PATCH_MATCH_TOPK)
            sims.append(s)

    order = np.argsort(-np.array(sims))[:TOPK]
    top_paths = [valid_paths[i] for i in order]
    top_scores = [float(sims[i]) for i in order]

    print("\n===== TOPK (patch-match rerank) =====")
    for r, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        print(f"{r:02d} score={s:.4f} {p}")

    out = make_topk_viz(QUERY_IMG, top_paths, top_scores, OUT_VIZ, tile=TILE)
    print(f"\n[VIZ] saved -> {out}")


if __name__ == "__main__":
    main()
