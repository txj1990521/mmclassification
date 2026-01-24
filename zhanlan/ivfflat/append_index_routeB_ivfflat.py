#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# =========================
# CONFIG: 只改这里
# =========================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

# 分批根目录（下面有很多批次子目录）
BATCH_ROOT = r"D:\zhanlan\data_batches"

# 现有索引/元数据
INDEX_PATH = r"D:\zhanlan\faiss_ivf.index"
META_PATH  = r"D:\zhanlan\faiss_paths.npy"

# 去重清单（增量用）
MANIFEST_PATH = r"D:\zhanlan\faiss_manifest.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# Route-B（与 search 对齐）
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

VIEW_BATCH = 256  # views batch

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

def load_manifest(path: str):
    if not os.path.exists(path):
        return {"seen_paths": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(path: str, manifest: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

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
    return F.normalize(feat, p=2, dim=1)

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
    agg = feats_view.max(dim=0).values
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index not found: {INDEX_PATH} (请先跑一次 base 建库并 train IVF)")

    index = faiss.read_index(INDEX_PATH)
    kept_paths = np.load(META_PATH, allow_pickle=True).tolist()
    print(f"[INFO] Loaded index ntotal={index.ntotal}, meta={len(kept_paths)}")

    manifest = load_manifest(MANIFEST_PATH)
    seen = set(manifest.get("seen_paths", []))
    print(f"[INFO] manifest seen={len(seen)}")

    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    all_paths = list_images_recursive(BATCH_ROOT)
    print(f"[INFO] scanned {len(all_paths)} images under {BATCH_ROOT}")

    # 过滤增量（按完整路径去重）
    new_paths = [str(p) for p in all_paths if str(p) not in seen]
    print(f"[INFO] new images = {len(new_paths)}")

    if not new_paths:
        print("[OK] nothing to add.")
        return

    IMG_BATCH = max(1, VIEW_BATCH // max(1, VIEWS_PER_IMAGE))
    print(f"[INFO] VIEWS_PER_IMAGE={VIEWS_PER_IMAGE}, VIEW_BATCH={VIEW_BATCH}, IMG_BATCH={IMG_BATCH}")

    img_batch_paths = []
    img_batch_views = []

    def flush_one_batch():
        nonlocal img_batch_paths, img_batch_views, kept_paths

        if not img_batch_paths:
            return 0

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
            agg = aggregate_views_to_one(feats_v[s:e])
            feats_img.append(agg.unsqueeze(0))

        feats_img = torch.cat(feats_img, dim=0).numpy().astype("float32")

        # IVF 增量 add（不需要 train）
        index.add(feats_img)
        kept_paths.extend(img_batch_paths)

        # 更新 manifest
        for p in img_batch_paths:
            seen.add(p)

        n_added = len(img_batch_paths)
        img_batch_paths.clear()
        img_batch_views.clear()
        return n_added

    added = 0
    for i, p in enumerate(new_paths, 1):
        img = imread_unicode(p)
        if img is None:
            continue

        # 每图稳定随机：同一路径每次生成的 views 一致（可重复构建/验证）
        random.seed(seed_from_path(p, base=0))

        views = make_views_for_index(img, mean, std, to_rgb=to_rgb)
        if not views:
            continue

        img_batch_paths.append(p)
        img_batch_views.append(views)

        if len(img_batch_paths) >= IMG_BATCH:
            added += flush_one_batch()
            if added % 1000 == 0:
                print(f"[ADD] added={added}  index_ntotal={index.ntotal}")

    added += flush_one_batch()

    # 保存 index / meta / manifest
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, np.array(kept_paths, dtype=object), allow_pickle=True)

    manifest["seen_paths"] = sorted(seen)
    save_manifest(MANIFEST_PATH, manifest)

    print(f"[SAVE] index -> {INDEX_PATH}")
    print(f"[SAVE] meta  -> {META_PATH}")
    print(f"[SAVE] manifest -> {MANIFEST_PATH}")
    print(f"[OK] appended images = {added}, index_ntotal={index.ntotal}")

if __name__ == "__main__":
    main()
