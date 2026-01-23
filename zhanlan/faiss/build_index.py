# build_index.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

DATA_ROOT = r"D:\zhanlan\data"      # 图库（递归扫描）
OUT_INDEX = r"D:\zhanlan\faiss.index"
OUT_META  = r"D:\zhanlan\faiss_paths.npy"

BATCH = 32  # 4060Ti 一般 16~64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 推理预处理（稳定版）
RESIZE_SHORT = 256
CROP_SIZE = 224


# =========================
# utils: 中文路径 imread
# =========================
def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images_recursive(root: str):
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    paths.sort()
    return paths


# =========================
# preprocess: BGR->RGB, resize short, center crop, normalize
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

def to_tensor_normalized(img_bgr: np.ndarray, mean, std, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_bgr[:, :, ::-1].copy()

    img = resize_short_edge(img, RESIZE_SHORT)
    img = center_crop(img, CROP_SIZE)

    x = img.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))  # HWC->CHW
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
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1,1,3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1,1,3)
    to_rgb = bool(dp.get("to_rgb", True))

    return model, mean, std, to_rgb

@torch.no_grad()
def extract_backbone_2048(model, batch_tensor: torch.Tensor):
    feat_map = model.backbone(batch_tensor)

    if isinstance(feat_map, dict):
        # 常见key: 'feat', 'features', 或取最后一个value
        if 'feat' in feat_map:
            feat_map = feat_map['feat']
        elif 'features' in feat_map:
            feat_map = feat_map['features']
        else:
            feat_map = list(feat_map.values())[-1]

    if isinstance(feat_map, (tuple, list)):
        feat_map = feat_map[-1]

    if feat_map.dim() == 2:
        # 有些backbone直接给 (N,C)，那就不用GAP
        feat = feat_map
    else:
        feat = feat_map.mean(dim=(2, 3))

    feat = F.normalize(feat, p=2, dim=1)
    return feat



# =========================
# FAISS
# =========================
def build_faiss_index(feats: np.ndarray):
    feats = feats.astype("float32")
    D = feats.shape[1]
    index = faiss.IndexFlatIP(D)  # cosine = normalized + inner product
    index.add(feats)
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

    batch_imgs = []
    batch_paths = []

    for i, p in enumerate(paths, 1):
        img = imread_unicode(str(p))
        if img is None:
            continue

        x = to_tensor_normalized(img, mean, std, to_rgb=to_rgb)
        batch_imgs.append(x)
        batch_paths.append(str(p))

        if len(batch_imgs) >= BATCH:
            bt = torch.stack(batch_imgs, dim=0).to(DEVICE)
            feat = extract_backbone_2048(model, bt).cpu().numpy()
            all_feats.append(feat)
            kept_paths.extend(batch_paths)

            batch_imgs, batch_paths = [], []

            if i % 500 == 0:
                print(f"[INDEX] processed {i}/{len(paths)}")

    if batch_imgs:
        bt = torch.stack(batch_imgs, dim=0).to(DEVICE)
        feat = extract_backbone_2048(model, bt).cpu().numpy()
        all_feats.append(feat)
        kept_paths.extend(batch_paths)

    if not all_feats:
        raise RuntimeError("No valid images found or all failed to read.")

    feats = np.concatenate(all_feats, axis=0).astype("float32")
    kept_paths = np.array(kept_paths, dtype=object)

    print(f"[DONE] feats shape = {feats.shape}")

    index = build_faiss_index(feats)
    faiss.write_index(index, OUT_INDEX)
    np.save(OUT_META, kept_paths, allow_pickle=True)

    print(f"[SAVE] index -> {OUT_INDEX}")
    print(f"[SAVE] paths -> {OUT_META}")
    print("[OK] Build finished.")


if __name__ == "__main__":
    main()
