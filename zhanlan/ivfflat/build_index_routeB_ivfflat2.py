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
- PATCH_IMG_PATHS: same as GLOBAL_META (share img_id->path)

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

PATCH_INDEX  = os.path.join(OUT_DIR, "patch.index")
PATCH_META   = os.path.join(OUT_DIR, "patch_meta.npy")
PATCH_IMG_PATHS = os.path.join(OUT_DIR, "patch_img_paths.npy")

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
# Patch tiling config (stable, not scene-specific)
# =========================
MAX_LONG = 1536
PATCH_SIZES = (256, 384, 512, 768)
STRIDE_RATIO = 0.5

# Cap patches per image to control explosion (still generic)
# If too many windows, we sample deterministically.
MAX_PATCHES_PER_IMAGE = 60

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
def resize_long_edge(img_bgr: np.ndarray, max_long: int):
    h, w = img_bgr.shape[:2]
    s = max_long / max(h, w)
    if s >= 1.0:
        return img_bgr
    nh, nw = int(round(h*s)), int(round(w*s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def sample_windows_deterministic(windows: List[Tuple[int,int,int,int,int]], k: int, seed: int):
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
      windows: list of (x1,y1,x2,y2,win) on resized image coordinates
    """
    img = resize_long_edge(img_bgr, max_long=max_long)
    H, W = img.shape[:2]

    windows = []
    for win in sizes:
        if H < win or W < win:
            continue
        stride = max(1, int(round(win * stride_ratio)))
        # Standard sliding windows
        for y1 in range(0, H - win + 1, stride):
            for x1 in range(0, W - win + 1, stride):
                windows.append((x1, y1, x1 + win, y1 + win, win))

    # Always include a central window at medium size if possible (robust, not scene-specific)
    for win in (512, 768, 384, 256):
        if H >= win and W >= win:
            cx1 = (W - win) // 2
            cy1 = (H - win) // 2
            windows.append((cx1, cy1, cx1 + win, cy1 + win, win))
            break

    # Deduplicate windows
    windows = list(dict.fromkeys(windows))

    # Cap per image
    if max_patches is not None and max_patches > 0:
        windows = sample_windows_deterministic(windows, max_patches, seed=seed)

    return img, windows

def patch_to_model_input(patch_bgr: np.ndarray, mean, std, to_rgb: bool, out_size=224):
    if to_rgb:
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    else:
        patch_rgb = patch_bgr[:, :, ::-1].copy()
    patch_rgb = cv2.resize(patch_rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return to_tensor_from_rgb(patch_rgb, mean, std)

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
    ensure_dir(PATCH_INDEX)

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
    patch_feats_chunks = []
    patch_meta_list = []  # list of (img_id,x1,y1,x2,y2,win) in resized coords
    total_patches = 0

    # For patch encoding we use smaller batches to fit GPU
    PATCH_BATCH = 256

    # We need img_id mapping
    img_id_map = {}  # path -> img_id

    for i, p in enumerate(paths, 1):
        img = imread_unicode(str(p))
        if img is None:
            continue

        # assign img_id
        img_id = len(img_paths) + len(img_batch_paths)  # current future index within img_paths
        # NOTE: Because img_paths is appended in flush, img_id based on this is tricky.
        # We'll instead assign img_id by a separate counter for correctness:
        # So let's use an independent counter:
        # (We'll fix by using a separate list and append immediately)
        # ----
        # We'll handle it by maintaining a definitive list img_paths_all.
        # ----

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

    # Patch streaming buffers
    patch_buf_tensors = []
    patch_buf_meta = []

    def flush_patch_batch():
        nonlocal patch_buf_tensors, patch_buf_meta, patch_feats_chunks, patch_meta_list, total_patches
        if not patch_buf_tensors:
            return
        bt = torch.stack(patch_buf_tensors, dim=0).to(DEVICE)
        feats = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        patch_feats_chunks.append(feats)
        patch_meta_list.extend(patch_buf_meta)
        total_patches += feats.shape[0]
        patch_buf_tensors = []
        patch_buf_meta = []

    for idx, (img_id, path) in enumerate(valid_images, 1):
        img = imread_unicode(path)
        if img is None:
            continue

        # ---- GLOBAL ----
        random.seed(seed_from_path(path, base=0))
        views = make_views_for_global(img, mean, std, to_rgb=to_rgb)
        if len(views) > 0:
            img_batch_paths.append(path)
            img_batch_views.append(views)
            if len(img_batch_paths) >= IMG_BATCH:
                flush_global_batch()

        # ---- PATCH ----
        # Deterministic seed per image (for window sampling)
        s_patch = seed_from_path(path, base=999)
        img_resized, windows = gen_patch_windows(
            img,
            sizes=PATCH_SIZES,
            stride_ratio=STRIDE_RATIO,
            max_long=MAX_LONG,
            max_patches=MAX_PATCHES_PER_IMAGE,
            seed=s_patch,
        )

        for (x1, y1, x2, y2, win) in windows:
            patch = img_resized[y1:y2, x1:x2]
            t = patch_to_model_input(patch, mean, std, to_rgb=to_rgb, out_size=PATCH_ENC_SIZE)
            patch_buf_tensors.append(t)
            patch_buf_meta.append((img_id, x1, y1, x2, y2, win))

            if len(patch_buf_tensors) >= PATCH_BATCH:
                flush_patch_batch()

        if idx % 500 == 0:
            print(f"[SCAN] {idx}/{len(valid_images)}  global_kept={len(img_paths)+len(img_batch_paths)}  patches={total_patches + len(patch_buf_tensors)}")

    flush_global_batch()
    flush_patch_batch()

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
    patch_feats = np.concatenate(patch_feats_chunks, axis=0).astype("float32")
    patch_meta = np.array(patch_meta_list, dtype=np.int32)  # shape (Npatch,6)

    keep_mask = np.array([m[0] in oldid_to_newid for m in patch_meta_list], dtype=bool)
    patch_feats = patch_feats[keep_mask]
    patch_meta = patch_meta[keep_mask]
    patch_meta[:, 0] = np.array([oldid_to_newid[int(x)] for x in patch_meta[:, 0]], dtype=np.int32)

    print(f"[DONE] Global feats: {global_feats2.shape}, Patch feats: {patch_feats.shape}")
    Nimg, D = global_feats2.shape
    Npatch = patch_feats.shape[0]

    # ---------- Build GLOBAL index ----------
    if Nimg < GLOBAL_FLAT_THRESHOLD:
        print(f"[GLOBAL] Small set ({Nimg}), using FlatIP")
        g_index = faiss.IndexFlatIP(D)
        g_index.add(global_feats2)
    else:
        g_index = build_ivfflat_ip(global_feats2, seed=IVF_SEED)

    # ---------- Build PATCH index ----------
    if Npatch < PATCH_FLAT_THRESHOLD_VECS:
        print(f"[PATCH] Small set ({Npatch}), using FlatIP")
        p_index = faiss.IndexFlatIP(D)
        p_index.add(patch_feats)
    else:
        # IVF-PQ is recommended for large patch sets
        p_index = build_ivfpq_ip(patch_feats, seed=IVF_SEED, m=PQ_M, nbits=PQ_NBITS)

    # Save
    faiss.write_index(g_index, GLOBAL_INDEX)
    np.save(GLOBAL_META, img_paths_compact, allow_pickle=True)

    faiss.write_index(p_index, PATCH_INDEX)
    np.save(PATCH_IMG_PATHS, img_paths_compact, allow_pickle=True)
    np.save(PATCH_META, patch_meta, allow_pickle=True)

    print(f"[SAVE] Global index -> {GLOBAL_INDEX}")
    print(f"[SAVE] Global paths -> {GLOBAL_META}")
    print(f"[SAVE] Patch index  -> {PATCH_INDEX}")
    print(f"[SAVE] Patch meta   -> {PATCH_META}")
    print(f"[SAVE] Patch paths  -> {PATCH_IMG_PATHS}")
    print("[OK] Hybrid build finished.")

if __name__ == "__main__":
    main()
