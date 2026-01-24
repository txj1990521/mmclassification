#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import hashlib
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
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

DATA_ROOT = r"D:\zhanlan\data"
OUT_INDEX = r"D:\zhanlan\faiss_database\faiss_ivf.index"
OUT_META  = r"D:\zhanlan\faiss_database\faiss_paths.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# Route-B Index Aug Config
# =========================
VIEWS_PER_IMAGE = 12
RESIZE_SHORT = 256
CROP_SIZE = 224

# view 计划（总计 12）: 与 search Route-B 对齐
VIEW_PLAN = [
    (0,   1, 5),
    (-15, 1, 1),
    (15,  1, 1),
    (-30, 1, 0),
    (30,  1, 0),
]

# 现在 batch 是 “views batch”
VIEW_BATCH = 256  # 4060Ti 通常 256~512 OK

# =========================
# IVF 参数（重点）
# =========================
# 10万级：4096；百万级：16384（建议你后期改成 16384）
NLIST = 4096
# 训练 IVF 用多少条向量（越多越稳，但越慢；10万级用 50k~100k 足够）
IVF_TRAIN_SIZE = 80000
IVF_TRAIN_SEED = 123

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

# =========================
# preprocess
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
# Views -> One embedding (Route B)
# =========================
@torch.no_grad()
def make_views_for_index(img_bgr: np.ndarray, mean, std, to_rgb: bool):
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
    # feats_view: (V,D) 已 L2
    agg = feats_view.max(dim=0).values
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

def pick_nlist(N: int) -> int:
    # 4*sqrt(N) 是非常稳的默认
    nlist = int(4 * np.sqrt(N))
    # clamp：避免太小/太大
    nlist = max(1024, nlist)
    nlist = min(65536, nlist)
    # 最后别超过 N/20（防止小库退化）
    nlist = min(nlist, max(1, N // 20))
    return max(1, nlist)

def pick_train_size(N: int, nlist: int) -> int:
    # 目标：至少 100*nlist，但别超过 N，也别太小
    t = min(N, max(nlist, 100 * nlist))
    # 上限再卡一下，避免训练过慢（你也可放大）
    t = min(t, 500_000)
    return int(t)

# =========================
# FAISS IVF
# =========================
def build_ivfflat_only(feats: np.ndarray, nlist: int, train_size: int, seed: int):
    feats = feats.astype("float32")
    N, D = feats.shape

    # --- nlist 保护 ---
    nlist_dyn = int(4 * np.sqrt(N))
    nlist = min(int(nlist), nlist_dyn)
    nlist = max(32, nlist)
    nlist = min(nlist, max(1, N // 20))

    # --- train size ---
    m = min(int(train_size), N)
    m = max(m, nlist)

    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)

    rng = np.random.default_rng(seed)
    train_idx = rng.choice(N, size=m, replace=False)
    train_x = feats[train_idx]

    print(f"[IVF] training on {m} vectors, nlist={nlist}")
    index.train(train_x)
    index.add(feats)
    assert index.is_trained
    assert index.ntotal == N

    print(f"[IVF] N={N} D={D} nlist={nlist} train_size={m}")
    print(f"[IVF] ntotal={index.ntotal}, nlist={index.nlist}")

    return index


# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)
    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    paths = list_images_recursive(DATA_ROOT)
    print(f"[INFO] Found {len(paths)} images under {DATA_ROOT}")

    all_feats = []
    kept_paths = []

    IMG_BATCH = max(1, VIEW_BATCH // max(1, VIEWS_PER_IMAGE))
    print(f"[INFO] VIEWS_PER_IMAGE={VIEWS_PER_IMAGE}, VIEW_BATCH={VIEW_BATCH}, IMG_BATCH={IMG_BATCH}")

    img_batch_paths = []
    img_batch_views = []

    def flush_one_batch():
        nonlocal img_batch_paths, img_batch_views, all_feats, kept_paths
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
        all_feats.append(feats_img)
        kept_paths.extend(img_batch_paths)

        img_batch_paths, img_batch_views = [], []

    for i, p in enumerate(paths, 1):
        img = imread_unicode(str(p))
        if img is None:
            continue

        # 每图固定但不同的随机性：稳定 + 多样
        random.seed(seed_from_path(str(p), base=0))

        views = make_views_for_index(img, mean, std, to_rgb=to_rgb)
        if len(views) == 0:
            continue

        img_batch_paths.append(str(p))
        img_batch_views.append(views)

        if len(img_batch_paths) >= IMG_BATCH:
            flush_one_batch()

        if i % 1000 == 0:
            print(f"[INDEX] scanned {i}/{len(paths)} kept={len(kept_paths)}")

    flush_one_batch()

    if not all_feats:
        raise RuntimeError("No valid images found or all failed to read.")

    feats = np.concatenate(all_feats, axis=0).astype("float32")
    kept_paths = np.array(kept_paths, dtype=object)
    print(f"[DONE] feats shape = {feats.shape}")

    N, D = feats.shape

    if N < 20000:
        print(f"[INDEX] Small dataset ({N}), using FlatIP")
        index = faiss.IndexFlatIP(D)
        index.add(feats)
    else:
        nlist = pick_nlist(N)
        train_size = pick_train_size(N, nlist)
        index = build_ivfflat_only(
            feats,
            nlist=nlist,
            train_size=train_size,
            seed=IVF_TRAIN_SEED
        )
    print(f"[INDEX] final: N={N}, D={D}, type={'Flat' if N < 20000 else 'IVF'}")
    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = min(64, index.nlist)

    faiss.write_index(index, OUT_INDEX)
    np.save(OUT_META, kept_paths, allow_pickle=True)
    np.save(OUT_META.replace(".npy", "_dim.npy"), np.array([D], np.int32))

    print(f"[SAVE] index -> {OUT_INDEX}")
    print(f"[SAVE] paths -> {OUT_META}")
    print("[OK] Build finished.")

if __name__ == "__main__":
    main()
