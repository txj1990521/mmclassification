#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST append new images to existing SimCLR FAISS index (Flat/IVF/IVF-PQ).

Fast path:
- update simclr_vec_to_imgid.npy incrementally
- rewrite images_meta.json (shared with CLIP)
- NO need to rebuild any jsonl
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS


# ============================================================
# ✅ CONFIG
# ============================================================
CONFIG: Dict[str, Any] = {
    # New images to append
    "NEW_IMG_ROOTS": [
        r"D:\zhanlan\new_data_add",
        r"D:\zhanlan\other_add",
    ],
    "IMG_EXTS": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
    "MAX_NEW_IMAGES": None,

    # Shared meta (same as CLIP side)
    "IMAGES_META_JSON": r".\outputs_hybrid_folder_big\images_meta.json",

    # SimCLR index folder/files
    "SIMCLR_OUT_DIR": r"D:\zhanlan\faiss_database_simclr_aligned",
    "SIMCLR_INDEX": "simclr_global.index",
    "SIMCLR_VEC2IMG": "simclr_vec_to_imgid.npy",

    "BAD_LOG": "bad_paths_append_simclr.log",

    # SimCLR model
    "CFG_PATH": r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py",
    "CKPT_PATH": r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth",

    "DEVICE": "auto",     # "auto" / "cpu" / "cuda"
    "BATCH": 256,
    "INPUT_SIZE": 224,

    # Dedup policy (same meaning as your CLIP append)
    "DEDUP_BY": "key",  # "key" | "abs" | "both"
}


# ============================================================
# Utils
# ============================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def pick_device(cfg: Dict[str, Any]) -> str:
    dv = str(cfg.get("DEVICE", "auto")).lower()
    if dv in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dv == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def norm_abs(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))

def list_images_multi_roots(roots: List[str], exts: List[str]) -> List[Tuple[str, str]]:
    exts_l = {e.lower() for e in exts}
    out: List[Tuple[str, str]] = []
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            raise FileNotFoundError(f"NEW_IMG_ROOT not found: {r}")
        for p in rp.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts_l:
                out.append((str(rp), str(p)))
    out.sort(key=lambda x: x[1])
    return out

def make_key(img_path: str, root: str) -> str:
    root_name = Path(root).name
    rel = os.path.relpath(img_path, root).replace("\\", "/")
    return f"{root_name}/{rel}"

def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def load_images_meta(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("images_meta.json must be a list")
    return data

def load_npy_if_exists(path: str, dtype=None):
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=False)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr

# ============================================================
# SimCLR model
# ============================================================
def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb

@torch.no_grad()
def extract_backbone_last(model, bt):
    feat = model.backbone(bt)
    if isinstance(feat, dict):
        feat = feat.get("feat", list(feat.values())[-1])
    if isinstance(feat, (tuple, list)):
        feat = feat[-1]
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))
    feat = F.normalize(feat, p=2, dim=1)
    return feat

def prep_tensor(img_bgr, mean, std, to_rgb, input_size: int):
    img = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) - mean) / std
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)

# ============================================================
# Main
# ============================================================
def main():
    cfg = CONFIG
    device = pick_device(cfg)

    # paths
    images_meta_path = cfg["IMAGES_META_JSON"]
    simclr_dir = cfg["SIMCLR_OUT_DIR"]
    simclr_index_path = os.path.join(simclr_dir, cfg["SIMCLR_INDEX"])
    simclr_vec2img_path = os.path.join(simclr_dir, cfg["SIMCLR_VEC2IMG"])
    bad_log_path = os.path.join(simclr_dir, cfg["BAD_LOG"])

    if not os.path.exists(simclr_index_path):
        raise FileNotFoundError(simclr_index_path)
    if not os.path.exists(images_meta_path):
        raise FileNotFoundError(images_meta_path)

    ensure_dir(simclr_dir)

    # list new images
    pairs = list_images_multi_roots(list(cfg["NEW_IMG_ROOTS"]), cfg["IMG_EXTS"])
    if isinstance(cfg.get("MAX_NEW_IMAGES"), int) and cfg["MAX_NEW_IMAGES"] > 0:
        pairs = pairs[: cfg["MAX_NEW_IMAGES"]]
    if not pairs:
        raise RuntimeError("No images found in NEW_IMG_ROOTS.")

    # load meta
    images_meta = load_images_meta(images_meta_path)
    existed_keys = set()
    existed_abs = set()
    for rec in images_meta:
        k = rec.get("key")
        p = rec.get("abs_path")
        if isinstance(k, str):
            existed_keys.add(k)
        if isinstance(p, str):
            existed_abs.add(norm_abs(p))

    # load simclr index + map
    index = faiss.read_index(simclr_index_path)
    if not index.is_trained:
        raise RuntimeError("SimCLR index is not trained. Build/train it first.")
    vec2img = load_npy_if_exists(simclr_vec2img_path, dtype=np.int32)
    if vec2img is None:
        raise FileNotFoundError(f"Missing {simclr_vec2img_path}. (建议先全量建库一次生成它)")

    if index.ntotal != len(vec2img):
        raise RuntimeError(f"[SIMCLR] index.ntotal={index.ntotal} != vec2img_len={len(vec2img)}")

    # build model
    model, mean, std, to_rgb = build_model(cfg["CFG_PATH"], cfg["CKPT_PATH"], device)
    batch_size = int(cfg.get("BATCH", 256))
    input_size = int(cfg.get("INPUT_SIZE", 224))

    # append buffers
    new_vec2img: List[int] = []
    bad_records: List[str] = []
    added_images = 0
    added_vecs = 0

    buf_t: List[torch.Tensor] = []
    buf_img_ids: List[int] = []

    def flush():
        nonlocal added_vecs, buf_t, buf_img_ids
        if not buf_t:
            return
        bt = torch.stack(buf_t, dim=0).to(device, non_blocking=True)
        fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        index.add(fv)
        added_vecs += int(fv.shape[0])
        new_vec2img.extend(buf_img_ids)
        buf_t, buf_img_ids = [], []

    dedup_by = str(cfg.get("DEDUP_BY", "key")).lower()

    for root, img_path in tqdm(pairs, desc="Append SimCLR (FAST)"):
        key = make_key(img_path, root)
        abs_norm = norm_abs(img_path)

        if dedup_by == "key":
            if key in existed_keys:
                continue
        elif dedup_by == "abs":
            if abs_norm in existed_abs:
                continue
        elif dedup_by == "both":
            if (key in existed_keys) or (abs_norm in existed_abs):
                continue

        img = imread_unicode(img_path)
        if img is None:
            bad_records.append(f"[BAD_IMAGE]\t{img_path}\tNone")
            continue

        h, w = img.shape[:2]
        img_id = len(images_meta)

        # append shared images_meta record
        images_meta.append({
            "img_id": img_id,
            "key": key,
            "abs_path": img_path,
            "src_wh": [w, h]
        })
        existed_keys.add(key)
        existed_abs.add(abs_norm)
        added_images += 1

        # SimCLR one-vector per image
        try:
            t = prep_tensor(img, mean, std, to_rgb, input_size)
        except Exception as e:
            bad_records.append(f"[PREP_FAIL]\t{img_path}\t{repr(e)}")
            continue

        buf_t.append(t)
        buf_img_ids.append(img_id)

        if len(buf_t) >= batch_size:
            flush()

    flush()

    # concat vec2img + sanity
    vec2img_new = np.concatenate([vec2img, np.asarray(new_vec2img, dtype=np.int32)], axis=0)
    if index.ntotal != len(vec2img_new):
        raise RuntimeError(f"[SIMCLR] after append: ntotal={index.ntotal} != map_len={len(vec2img_new)}")
    if added_vecs != len(new_vec2img):
        raise RuntimeError(f"[SIMCLR] mapping mismatch: new_map={len(new_vec2img)} vs added_vecs={added_vecs}")

    # write outputs
    with open(images_meta_path, "w", encoding="utf-8") as f:
        json.dump(images_meta, f, ensure_ascii=False)

    faiss.write_index(index, simclr_index_path)
    np.save(simclr_vec2img_path, vec2img_new.astype(np.int32))

    if bad_records:
        with open(bad_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(bad_records))

    print("\n===== APPEND DONE (SIMCLR FAST) =====")
    print(f"device={device}")
    print(f"added_images={added_images}")
    print(f"added_vecs={added_vecs} | SIMCLR ntotal={index.ntotal}")
    print("saved:", simclr_index_path)
    print("map  :", simclr_vec2img_path)
    if bad_records:
        print("bad log:", bad_log_path)


if __name__ == "__main__":
    main()