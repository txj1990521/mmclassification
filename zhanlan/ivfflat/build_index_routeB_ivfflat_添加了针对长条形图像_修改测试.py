#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build HYBRID FAISS indexes:
1) Global index: 1 embedding per image (same as your current Route-B style).
2) Patch index: multiple patch embeddings per image (multi-scale tiling).

Outputs:
- GLOBAL_INDEX:  global index file
- GLOBAL_META:   image paths (img_id -> path)

- PATCH_INDEX:   patch index file
- PATCH_META:    patch meta array (patch_id -> {img_id,bbox,win})

Notes:
- For small datasets (<20k images / <200k patches) it will fall back to FlatIP.
- For larger patch sets it will use IVF-PQ by default.
"""

import os
import random
import hashlib
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS
from hybrid_shared import (
    STRIPE_AR_THR, STRIPE_LONG_EDGE,
    STRIPE_WIN_H, STRIPE_STRIDE, STRIPE_MAX_PATCHES,
    STRIPE_CENTER_FRAC, STRIPE_JITTER,
    PATCH_SIZES, STRIDE_RATIO, MAX_LONG, MAX_PATCHES_PER_IMAGE,
    resize_long_edge, seed_from_image,
    gen_patch_windows_unified, patch_to_model_input,
)

# =========================
# CONFIG: 只改这里
# =========================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

DATA_ROOT = r"D:\zhanlan\new_data"

# ----- outputs -----
OUT_DIR = r"D:\zhanlan\faiss_database_hybrid_new_data"
GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")




PATCH_STRIPE_INDEX = os.path.join(OUT_DIR, "patch_stripe.index")
PATCH_GRID_INDEX   = os.path.join(OUT_DIR, "patch_grid.index")

PATCH_STRIPE_META  = os.path.join(OUT_DIR, "patch_stripe_meta.npy")
PATCH_GRID_META    = os.path.join(OUT_DIR, "patch_grid_meta.npy")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# Global Route-B config
# =========================
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


# =========================
# STRIPE_SHARED_CONSTANTS
# =========================

STRIPE_WIN_W_FRAC = 0.85
STRIPE_WIN_W_MIN = 160
STRIPE_WIN_W_MAX = 256


# =========================
# Patch tiling config (stable, not scene-specific)
# =========================

# Cap patches per image to control explosion (still generic)
# If too many windows, we sample deterministically.
# Patch encoding: we will center-crop/resize each patch to 224 before model
PATCH_ENC_SIZE = 224

# =========================
# IVF/PQ settings
# =========================
IVF_SEED = 123

# Global index threshold
GLOBAL_FLAT_THRESHOLD = 20000  # images
# Patch index threshold
PATCH_FLAT_THRESHOLD_VECS = 200000  # patches

# IVF nlist heuristic is derived from N (vectors)
def pick_nlist(N: int) -> int:
    nlist = int(4 * np.sqrt(max(N, 1)))
    nlist = max(1024, nlist)
    nlist = min(65536, nlist)
    nlist = min(nlist, max(1, N // 20))
    return max(1, nlist)

def pick_train_size(N: int, nlist: int) -> int:
    t = min(N, max(nlist, 100 * nlist))
    t = min(t, 500_000)
    return int(t)

# IVF-PQ params (common default)
PQ_M = 32     # number of subquantizers (must divide D)
PQ_NBITS = 8  # bits per code


# =========================
# utils
# =========================
def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images_recursive(root: str):
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    paths.sort()
    return paths

def seed_from_path(p: str, base: int = 0) -> int:
    h = hashlib.md5(p.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff

def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# =========================
# preprocess for global views
# =========================
def resize_short_edge(img_rgb: np.ndarray, short=256):
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

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

def to_tensor_from_rgb(img_rgb_crop: np.ndarray, mean, std):
    x = img_rgb_crop.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))

# =========================
# Patch tiling
# =========================
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



def sample_windows_deterministic(windows: List[Tuple[int,int,int,int,int,int,int]], k: int, seed: int):
    """Deterministically sample k windows from list."""
    if len(windows) <= k:
        return windows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=k, replace=False)
    idx = np.sort(idx)
    return [windows[i] for i in idx.tolist()]

def gen_patch_windows(img_bgr: np.ndarray,
                      sizes=PATCH_SIZES,
                      stride_ratio=STRIDE_RATIO,
                      max_long=MAX_LONG,
                      max_patches=MAX_PATCHES_PER_IMAGE,
                      seed: int = 0):
    """
    Return:
      img_resized (bgr),
      windows: list of (x1,y1,x2,y2,win,patch_type,pos_int) on resized image coords

    patch_type:
      0 = square grid (old)
      1 = stripe sliding (new)
    pos_int:
      0..10000, for stripe; for square use 5000 as placeholder
    """
    img = resize_long_edge(img_bgr, max_long=max_long)
    H, W = img.shape[:2]

    # 判定条形（和 query 一致的思路）
    ar = max(W / (H + 1e-6), H / (W + 1e-6))
    is_stripe = (ar >= STRIPE_AR_THR)

    windows = []

    if is_stripe:
        # ---- 条形滑窗 ----
        # 这些参数建议与你 query 侧保持一致（你 get_query_patch_feats 用的）
        # 可以按数据再调，但先用这套稳态默认
        # unified stripe params (library side)
        win_h = 224  # length along stripe
        stride = 48  # dense sliding
        max_patches = 24  # stable cap

        win_w = int(np.clip(STRIPE_WIN_W_FRAC * W, STRIPE_WIN_W_MIN, STRIPE_WIN_W_MAX))
        windows = gen_stripe_windows(
            H, W,
            max_patches=STRIPE_MAX_PATCHES,
            seed=seed,
            win_w=win_w,
            win_h=STRIPE_WIN_H,
            stride=STRIPE_STRIDE,
            center_frac=STRIPE_CENTER_FRAC,
            jitter=STRIPE_JITTER,
        )

        # 如果窗太少，补一个中心大窗（兜底）
        if len(windows) < min(6, max_patches):
            ww = min(max(win_w, 160), W)
            hh = min(max(win_h, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            x2, y2 = x1 + ww, y1 + hh
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, 5000))

    else:
        # ---- 原来的方形网格 ----
        for win in sizes:
            if H < win or W < win:
                continue
            stride = max(1, int(round(win * stride_ratio)))
            for y1 in range(0, H - win + 1, stride):
                for x1 in range(0, W - win + 1, stride):
                    windows.append((x1, y1, x1 + win, y1 + win, win, 0, 5000))

        # 中心窗兜底
        for win in (512, 768, 384, 256):
            if H >= win and W >= win:
                cx1 = (W - win) // 2
                cy1 = (H - win) // 2
                windows.append((cx1, cy1, cx1 + win, cy1 + win, win, 0, 5000))
                break

        # 去重（含 patch_type/pos 的 tuple 一样才算重复）
        windows = list(dict.fromkeys(windows))

        # cap
        if max_patches is not None and max_patches > 0:
            windows = sample_windows_deterministic(windows, max_patches, seed=seed)

    return img, windows


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
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]),
                    dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]),
                    dtype=np.float32).reshape(1, 1, 3)
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

# =========================
# Global views -> embedding
# =========================
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
    agg = feats_view.max(dim=0).values
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

# =========================
# FAISS builders
# =========================
def build_ivfflat_ip(feats: np.ndarray, seed: int):
    feats = feats.astype("float32")
    N, D = feats.shape
    nlist = pick_nlist(N)
    train_size = pick_train_size(N, nlist)
    m = min(train_size, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=m, replace=False)
    train_x = feats[idx]

    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    print(f"[IVF-Flat] training on {m}, nlist={nlist}")
    index.train(train_x)
    index.add(feats)
    index.nprobe = min(64, index.nlist)
    return index

def build_ivfpq_ip(feats: np.ndarray, seed: int, m: int = PQ_M, nbits: int = PQ_NBITS):
    feats = feats.astype("float32")
    N, D = feats.shape
    if D % m != 0:
        raise ValueError(f"PQ_M={m} must divide D={D}. Please change PQ_M.")
    nlist = pick_nlist(N)
    train_size = pick_train_size(N, nlist)
    tr = min(train_size, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=tr, replace=False)
    train_x = feats[idx]

    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    print(f"[IVF-PQ] training on {tr}, nlist={nlist}, m={m}, nbits={nbits}")
    index.train(train_x)
    index.add(feats)
    index.nprobe = min(64, index.nlist)
    return index

# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)
    ensure_dir(GLOBAL_INDEX)
    ensure_dir(PATCH_STRIPE_INDEX)
    ensure_dir(PATCH_GRID_INDEX)

    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    paths = list_images_recursive(DATA_ROOT)
    print(f"[INFO] Found {len(paths)} images under {DATA_ROOT}")

    # ---------- Build GLOBAL feats ----------
    global_feats_chunks = []
    img_paths = []

    IMG_BATCH = max(1, VIEW_BATCH // max(1, VIEWS_PER_IMAGE))
    print(f"[GLOBAL] VIEWS_PER_IMAGE={VIEWS_PER_IMAGE}, VIEW_BATCH={VIEW_BATCH}, IMG_BATCH={IMG_BATCH}")

    img_batch_paths = []
    img_batch_views = []

    def flush_global_batch():
        nonlocal img_batch_paths, img_batch_views, global_feats_chunks, img_paths
        if not img_batch_paths:
            return
        flat_views = []
        offsets = [0]
        for vs in img_batch_views:
            flat_views.extend(vs)
            offsets.append(len(flat_views))

        bt = torch.stack(flat_views, dim=0).to(DEVICE)
        feats_v = extract_backbone_last(model, bt).cpu()  # (sumV,D)

        feats_img = []
        for k in range(len(img_batch_paths)):
            s, e = offsets[k], offsets[k+1]
            fv = feats_v[s:e]
            agg = aggregate_views_to_one(fv)
            feats_img.append(agg.unsqueeze(0))

        feats_img = torch.cat(feats_img, dim=0).numpy().astype("float32")
        global_feats_chunks.append(feats_img)
        img_paths.extend(img_batch_paths)
        img_batch_paths, img_batch_views = [], []

    # ---------- Build PATCH feats (streaming) ----------
    # We'll collect in chunks too; for huge sets you may want memmap, but this is a complete baseline.
    # stripe
    patch_buf_tensors_s = []
    patch_buf_meta_s = []
    patch_feats_chunks_s = []
    patch_meta_list_s = []

    # grid
    patch_buf_tensors_g = []
    patch_buf_meta_g = []
    patch_feats_chunks_g = []
    patch_meta_list_g = []

    # Patch streaming buffers
    # For patch encoding we use smaller batches to fit GPU
    PATCH_BATCH = 256

    # Re-scan with definitive img_id list to keep code simple and correct.
    img_paths_all = []
    valid_images = []
    for p in paths:
        img = imread_unicode(str(p))
        if img is None:
            continue
        img_id = len(img_paths_all)
        img_paths_all.append(str(p))
        valid_images.append((img_id, str(p)))

    print(f"[INFO] Valid images = {len(img_paths_all)}")

    # Now build global and patch features from valid_images
    # Reset global containers
    global_feats_chunks = []
    img_paths = []
    img_batch_paths = []
    img_batch_views = []

    def flush_patch_batch_stripe():
        nonlocal patch_buf_tensors_s, patch_buf_meta_s
        if not patch_buf_tensors_s:
            return
        bt = torch.stack(patch_buf_tensors_s).to(DEVICE)
        feats = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        patch_feats_chunks_s.append(feats)
        patch_meta_list_s.extend(patch_buf_meta_s)
        patch_buf_tensors_s.clear()
        patch_buf_meta_s.clear()

    def flush_patch_batch_grid():
        nonlocal patch_buf_tensors_g, patch_buf_meta_g
        if not patch_buf_tensors_g:
            return
        bt = torch.stack(patch_buf_tensors_g).to(DEVICE)
        feats = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        patch_feats_chunks_g.append(feats)
        patch_meta_list_g.extend(patch_buf_meta_g)
        patch_buf_tensors_g.clear()
        patch_buf_meta_g.clear()

    for idx, (img_id, path) in enumerate(valid_images, 1):
        img = imread_unicode(path)
        if img is None:
            continue

        # ---- GLOBAL ----
        # 统一做 seed 的源图
        seed_src = resize_long_edge(img, max_long=STRIPE_LONG_EDGE)
        random.seed(seed_from_image(seed_src, base=0))

        # views 用同一个 seed_src（或者至少同一份 resize 结果）
        views = make_views_for_global(seed_src, mean, std, to_rgb)

        if len(views) > 0:
            img_batch_paths.append(path)
            img_batch_views.append(views)
            if len(img_batch_paths) >= IMG_BATCH:
                flush_global_batch()

        # ---- PATCH ----
        # Deterministic seed per image (for window sampling)
        img_resized_tmp = resize_long_edge(img, max_long=MAX_LONG)
        s_patch = seed_from_image(img_resized_tmp, base=999)
        img_resized, windows, is_stripe = gen_patch_windows_unified(
            img,
            max_long=MAX_LONG,
            stripe_ar_thr=STRIPE_AR_THR,
            seed_base=999,
            stripe_max_patches=STRIPE_MAX_PATCHES,
            stripe_win_h=STRIPE_WIN_H,
            stripe_stride=STRIPE_STRIDE,
            stripe_center_frac=STRIPE_CENTER_FRAC,
            stripe_jitter=STRIPE_JITTER,
            grid_sizes=PATCH_SIZES,
            grid_stride_ratio=STRIDE_RATIO,
            max_patches=MAX_PATCHES_PER_IMAGE,
        )

        for (x1, y1, x2, y2, win, ptype, pos) in windows:
            patch = img_resized[y1:y2, x1:x2]
            t = patch_to_model_input(patch, mean, std, to_rgb, PATCH_ENC_SIZE)

            if ptype == 1:  # stripe
                patch_buf_tensors_s.append(t)
                patch_buf_meta_s.append((img_id, x1, y1, x2, y2, win, ptype, pos))
                if len(patch_buf_tensors_s) >= PATCH_BATCH:
                    flush_patch_batch_stripe()
            else:  # grid
                patch_buf_tensors_g.append(t)
                patch_buf_meta_g.append((img_id, x1, y1, x2, y2, win, ptype, pos))
                if len(patch_buf_tensors_g) >= PATCH_BATCH:
                    flush_patch_batch_grid()

        if idx % 500 == 0:
            ps = sum(x.shape[0] for x in patch_feats_chunks_s) + len(patch_buf_tensors_s)
            pg = sum(x.shape[0] for x in patch_feats_chunks_g) + len(patch_buf_tensors_g)
            print(
                f"[SCAN] {idx}/{len(valid_images)} global_kept={len(img_paths) + len(img_batch_paths)}  stripe={ps} grid={pg}")

    flush_global_batch()
    flush_patch_batch_stripe()
    flush_patch_batch_grid()

    # Finalize global feats
    if not global_feats_chunks:
        raise RuntimeError("No global feats extracted.")
    global_feats = np.concatenate(global_feats_chunks, axis=0).astype("float32")
    img_paths = np.array(img_paths, dtype=object)

    # IMPORTANT: img_paths here contains only images for which we extracted global views.
    # But we want a unified img_id mapping. We'll use img_paths_all as canonical.
    # If some images failed global extraction, you can either drop them from patch side or compute global too.
    # To keep consistent, we will rebuild global index over img_paths_all order using a mask.
    # For simplicity, we enforce: only keep images that appear in img_paths (global built).
    global_set = set(img_paths.tolist())
    keep_img_ids = []
    keep_paths = []
    for img_id, pth in valid_images:
        if pth in global_set:
            keep_img_ids.append(img_id)
            keep_paths.append(pth)

    # Map old global_feats rows (img_paths order) to canonical img_id
    # build mapping path->row
    path_to_row = {p: i for i, p in enumerate(img_paths.tolist())}
    global_feats2 = np.zeros((len(keep_paths), global_feats.shape[1]), dtype=np.float32)
    for new_i, pth in enumerate(keep_paths):
        global_feats2[new_i] = global_feats[path_to_row[pth]]

    # Build remap from original img_id (in img_paths_all) to new compact id (0..N-1)
    oldid_to_newid = {old: new for new, old in enumerate(keep_img_ids)}
    img_paths_compact = np.array(keep_paths, dtype=object)

    # Filter patch meta to only kept images, and remap img_id
    patch_feats_s = np.concatenate(patch_feats_chunks_s, axis=0) if patch_feats_chunks_s else np.zeros(
        (0, global_feats2.shape[1]), np.float32)
    patch_meta_s = np.array(patch_meta_list_s, dtype=np.int32) if patch_meta_list_s else np.zeros((0, 8), np.int32)

    patch_feats_g = np.concatenate(patch_feats_chunks_g, axis=0) if patch_feats_chunks_g else np.zeros(
        (0, global_feats2.shape[1]), np.float32)
    patch_meta_g = np.array(patch_meta_list_g, dtype=np.int32) if patch_meta_list_g else np.zeros((0, 8), np.int32)

    # stripe
    keep_mask_s = np.array([m[0] in oldid_to_newid for m in patch_meta_s], dtype=bool)
    patch_feats_s = patch_feats_s[keep_mask_s]
    patch_meta_s = patch_meta_s[keep_mask_s]
    patch_meta_s[:, 0] = np.array([oldid_to_newid[int(x)] for x in patch_meta_s[:, 0]], dtype=np.int32)

    # grid
    keep_mask_g = np.array([m[0] in oldid_to_newid for m in patch_meta_g], dtype=bool)
    patch_feats_g = patch_feats_g[keep_mask_g]
    patch_meta_g = patch_meta_g[keep_mask_g]
    patch_meta_g[:, 0] = np.array([oldid_to_newid[int(x)] for x in patch_meta_g[:, 0]], dtype=np.int32)

    print(f"[DONE] Global feats: {global_feats2.shape}, Patch stripe feats: {patch_feats_s.shape},Patch grid feats: {patch_feats_g.shape}")
    Nimg, D = global_feats2.shape
    D = int(global_feats2.shape[1])
    # 或者 assert patch_feats_s.shape[1] == D

    # ---------- Build GLOBAL index ----------
    if Nimg < GLOBAL_FLAT_THRESHOLD:
        print(f"[GLOBAL] Small set ({Nimg}), using FlatIP")
        g_index = faiss.IndexFlatIP(D)
        g_index.add(global_feats2)
    else:
        g_index = build_ivfflat_ip(global_feats2, seed=IVF_SEED)

    # ---------- Build PATCH index ----------
    # stripe index
    if len(patch_feats_s) < PATCH_FLAT_THRESHOLD_VECS:
        p_index_s = faiss.IndexFlatIP(D)
        p_index_s.add(patch_feats_s)
    else:
        p_index_s = build_ivfpq_ip(patch_feats_s, seed=IVF_SEED)

    # grid index
    if len(patch_feats_g) < PATCH_FLAT_THRESHOLD_VECS:
        p_index_g = faiss.IndexFlatIP(D)
        p_index_g.add(patch_feats_g)
    else:
        p_index_g = build_ivfpq_ip(patch_feats_g, seed=IVF_SEED)

    # Save
    faiss.write_index(g_index, GLOBAL_INDEX)
    np.save(GLOBAL_META, img_paths_compact, allow_pickle=True)

    faiss.write_index(p_index_s, PATCH_STRIPE_INDEX)
    np.save(PATCH_STRIPE_META, patch_meta_s, allow_pickle=True)

    faiss.write_index(p_index_g, PATCH_GRID_INDEX)
    np.save(PATCH_GRID_META, patch_meta_g, allow_pickle=True)


    print(f"[SAVE] Global index -> {GLOBAL_INDEX}")
    print(f"[SAVE] Global paths -> {GLOBAL_META}")
    print("[OK] Hybrid build finished.")

if __name__ == "__main__":
    main()

