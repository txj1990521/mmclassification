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
# CONFIG
# =========================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

BATCH_ROOT = r"D:\zhanlan\data_batches"

INDEX_PATH = r"D:\zhanlan\faiss_ivf_idmap.index"
META_PATH  = r"D:\zhanlan\faiss_paths.npy"
MANIFEST_PATH = r"D:\zhanlan\faiss_manifest.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# Route-B
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
BASE_SEED = 0     # 控制随机 crop 的可复现

# fingerprint 轻量 hash 参数
HASH_HEAD = 64 * 1024
HASH_TAIL = 64 * 1024

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
    return [str(p) for p in paths]

def seed_from_path(p: str, base: int = 0) -> int:
    h = hashlib.md5(p.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff

def file_sig_fast(path: str) -> str:
    """hash(head+tail) + size：快速且很稳"""
    try:
        st = os.stat(path)
        size = st.st_size
        with open(path, "rb") as f:
            head = f.read(HASH_HEAD)
            if size > HASH_TAIL:
                f.seek(max(0, size - HASH_TAIL), os.SEEK_SET)
                tail = f.read(HASH_TAIL)
            else:
                tail = b""
        m = hashlib.md5()
        m.update(str(size).encode("utf-8"))
        m.update(head)
        m.update(tail)
        return m.hexdigest()
    except Exception:
        return ""

def fingerprint(path: str):
    st = os.stat(path)
    size = int(st.st_size)
    mtime = int(st.st_mtime)  # 秒级就够了
    sig = file_sig_fast(path)
    return {"size": size, "mtime": mtime, "sig": sig}

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_atomic(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
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
def make_views(img_bgr: np.ndarray, mean, std, to_rgb: bool):
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
# faiss helpers
# =========================
def remove_ids(index, ids_np: np.ndarray):
    ids_np = ids_np.astype(np.int64)
    sel = faiss.IDSelectorBatch(ids_np.size, faiss.swig_ptr(ids_np))
    index.remove_ids(sel)

# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index not found: {INDEX_PATH}（请先 base 建库：IVFFlat + IDMap2 并 train）")

    index = faiss.read_index(INDEX_PATH)
    if not isinstance(index, faiss.IndexIDMap2):
        raise RuntimeError("你的 index 不是 IndexIDMap2。需要用 IndexIDMap2 包起来，才能 remove_ids。")

    kept_paths = np.load(META_PATH, allow_pickle=True).tolist()

    manifest = load_json(MANIFEST_PATH, default={"next_id": 0, "items": {}})
    items = manifest["items"]
    next_id = int(manifest.get("next_id", 0))

    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    paths = list_images_recursive(BATCH_ROOT)
    print(f"[INFO] scanned {len(paths)} images under {BATCH_ROOT}")

    # 统计：哪些需要新增 / 更新
    to_add = []
    to_update = []  # (path, old_id)

    for p in paths:
        try:
            fp = fingerprint(p)
        except Exception:
            continue

        rec = items.get(p)
        if rec is None:
            to_add.append((p, fp))
        else:
            # 判断是否变化：sig 优先，其次 size/mtime
            changed = (rec.get("sig") != fp.get("sig")) or (rec.get("size") != fp["size"]) or (rec.get("mtime") != fp["mtime"])
            if changed:
                to_update.append((p, fp, int(rec["id"])))

    print(f"[PLAN] add={len(to_add)} update={len(to_update)} index_ntotal={index.ntotal}")

    IMG_BATCH = max(1, VIEW_BATCH // max(1, VIEWS_PER_IMAGE))
    print(f"[INFO] IMG_BATCH={IMG_BATCH} (VIEW_BATCH={VIEW_BATCH}, VIEWS_PER_IMAGE={VIEWS_PER_IMAGE})")

    def process_batch(batch_list):
        """batch_list: list[(path, fp, id)] 统一提特征并 add_with_ids"""
        flat_views = []
        offsets = [0]
        ids = []
        paths_local = []
        fps_local = []

        for (p, fp, fid) in batch_list:
            img = imread_unicode(p)
            if img is None:
                continue

            random.seed(seed_from_path(p, base=BASE_SEED))
            vs = make_views(img, mean, std, to_rgb=to_rgb)
            if not vs:
                continue

            paths_local.append(p)
            fps_local.append(fp)
            ids.append(fid)

            flat_views.extend(vs)
            offsets.append(len(flat_views))

        if not ids:
            return 0

        bt = torch.stack(flat_views, dim=0).to(DEVICE)
        feats_v = extract_backbone_last(model, bt).cpu()

        feats_img = []
        for k in range(len(ids)):
            s, e = offsets[k], offsets[k+1]
            feats_img.append(aggregate_views_to_one(feats_v[s:e]).unsqueeze(0))

        x = torch.cat(feats_img, dim=0).numpy().astype("float32")
        ids_np = np.array(ids, dtype=np.int64)

        index.add_with_ids(x, ids_np)

        # meta / manifest 更新（按 id 写入）
        for p, fp, fid in zip(paths_local, fps_local, ids):
            items[p] = {"id": int(fid), "size": int(fp["size"]), "mtime": int(fp["mtime"]), "sig": fp.get("sig", "")}
        return len(ids)

    # 1) update：先 remove 再 add（复用旧 id）
    updated = 0
    if to_update:
        # 先批量 remove
        rm_ids = np.array([old_id for (_, _, old_id) in to_update], dtype=np.int64)
        print(f"[REMOVE] ids={rm_ids.size}")
        remove_ids(index, rm_ids)

        # 再分批 add_with_ids
        batch = []
        for (p, fp, old_id) in to_update:
            batch.append((p, fp, old_id))
            if len(batch) >= IMG_BATCH:
                updated += process_batch(batch)
                batch = []
        if batch:
            updated += process_batch(batch)

    # 2) add：分配新 id
    added = 0
    if to_add:
        batch = []
        for (p, fp) in to_add:
            fid = next_id
            next_id += 1
            batch.append((p, fp, fid))
            if len(batch) >= IMG_BATCH:
                added += process_batch(batch)
                batch = []
        if batch:
            added += process_batch(batch)

    # 3) 保存 index / meta / manifest
    # meta：按 id 不太好直接稠密数组（id 会越来越大），所以你两种选：
    # A) 继续用 kept_paths list（只做展示），不再依赖它做 id->path
    # B) 新建一个 id->path 的 dict（推荐）
    #
    # 这里我给你 B：保存一个 paths_map.json，search 时按 id 查
    PATHS_MAP = r"D:\zhanlan\faiss_id2path.json"
    id2path = load_json(PATHS_MAP, default={})
    for p, rec in items.items():
        id2path[str(rec["id"])] = p
    save_json_atomic(PATHS_MAP, id2path)

    # 旧的 kept_paths 仍然保存（可选）
    np.save(META_PATH, np.array(kept_paths, dtype=object), allow_pickle=True)

    manifest["items"] = items
    manifest["next_id"] = int(next_id)
    save_json_atomic(MANIFEST_PATH, manifest)

    faiss.write_index(index, INDEX_PATH)

    print(f"[SAVE] index -> {INDEX_PATH}")
    print(f"[SAVE] manifest -> {MANIFEST_PATH}")
    print(f"[SAVE] id2path -> {PATHS_MAP}")
    print(f"[OK] add={added} update={updated} index_ntotal={index.ntotal}")

if __name__ == "__main__":
    main()
