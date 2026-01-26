#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import shutil
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import faiss
import yaml

# ============================================================
# ✅ EDIT HERE
# ============================================================
MMPRETRAIN_ROOT = r"D:\zhanlanProject\mmpretrain"
DINOV2_VARIANT = "small"   # small/base/large/giant

DATA_ROOT = r"D:\zhanlan\data"
OUT_DIR = r"D:\zhanlan\cluster_out"

K = 10
NITER = 50
SEED = 123

BATCH_SIZE = 16            # img_size=518 比较吃显存，建议 8/16
MAX_IMAGES = -1            # -1=all
VIS_PER_CLUSTER = 25
COPY_MODE = True           # True=copy representative imgs into folders

FAISS_GPU = True           # 没装 faiss-gpu 会自动 fallback
# ============================================================


def add_repo_to_syspath(repo_root: str):
    # 让 “源码直接跑” 也能 import mmpretrain/mmengine
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


add_repo_to_syspath(MMPRETRAIN_ROOT)

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def list_images(data_root: str, recursive=True) -> List[str]:
    pats = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    out = []
    for pat in pats:
        if recursive:
            out.extend(glob.glob(os.path.join(data_root, "**", pat), recursive=True))
        else:
            out.extend(glob.glob(os.path.join(data_root, pat)))
    return sorted(list(set(out)))


def build_preprocess(mean, std, to_rgb: bool, img_size: int):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def preprocess_one(img_bgr: np.ndarray) -> torch.Tensor:
        if to_rgb:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img = img_bgr[:, :, ::-1].copy()
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))  # CHW
        return torch.from_numpy(x)

    return preprocess_one


@torch.no_grad()
def extract_embedding_batch(model, batch: torch.Tensor) -> torch.Tensor:
    """
    兼容 DINOv2 headless 的三种输出：
      - [B, D]                 (最常见：已是全局 embedding)
      - [B, N, C] tokens       (含 cls)
      - [B, C, H, W] feature map
    """
    feat = model.backbone(batch)

    # unwrap dict / list / tuple
    if isinstance(feat, dict):
        feat = list(feat.values())[-1]
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    if not torch.is_tensor(feat):
        raise RuntimeError(f"Backbone output is not a tensor: {type(feat)}")

    if feat.dim() == 2:          # [B, D]
        emb = feat
    elif feat.dim() == 3:        # [B, N, C]
        # cls token优先
        emb = feat[:, 0, :]
    elif feat.dim() == 4:        # [B, C, H, W]
        emb = feat.mean(dim=(2, 3))
    else:
        raise RuntimeError(f"Unsupported feat shape: {tuple(feat.shape)}")

    emb = F.normalize(emb, p=2, dim=1)
    return emb

def read_metafile_and_pick_weight(metafile_path: str, variant: str) -> Tuple[str, str]:
    """
    从 configs/dinov2/metafile.yml 读取对应 variant 的：
      - config path
      - weights url
    """
    with open(metafile_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    name_key = {
        "small": "vit-small-p14_dinov2-pre_3rdparty",
        "base":  "vit-base-p14_dinov2-pre_3rdparty",
        "large": "vit-large-p14_dinov2-pre_3rdparty",
        "giant": "vit-giant-p14_dinov2-pre_3rdparty",
    }[variant]

    models = meta.get("Models", [])
    for m in models:
        if m.get("Name") == name_key:
            cfg_rel = m.get("Config")
            w = m.get("Weights")
            return cfg_rel, w

    raise RuntimeError(f"Cannot find {name_key} in {metafile_path}")


def faiss_kmeans_cluster(
    x: np.ndarray, k: int, niter: int, seed: int, use_gpu: bool
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    n, d = x.shape
    if n < k:
        raise ValueError(f"N ({n}) < k ({k})")

    # 某些 faiss 版本没有 faiss.rand.seed；只用 Kmeans 的 seed 即可
    km = faiss.Kmeans(
        d, k,
        niter=niter,
        verbose=True,
        seed=seed,
        gpu=use_gpu,
        spherical=True  # 你已做 L2 normalize，用 spherical 更符合余弦空间
    )
    km.train(x)
    _, I = km.index.search(x, 1)
    cluster_ids = I.reshape(-1).astype(np.int32)
    centroids = km.centroids.astype(np.float32)
    return cluster_ids, centroids

def export_clusters(paths: List[str], cluster_ids: np.ndarray, out_dir: str) -> Dict[str, List[str]]:
    clusters: Dict[int, List[str]] = {}
    for p, cid in zip(paths, cluster_ids):
        clusters.setdefault(int(cid), []).append(p)

    clusters_json = {str(k): v for k, v in clusters.items()}
    with open(os.path.join(out_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters_json, f, ensure_ascii=False, indent=2)
    return clusters_json


def export_cluster_vis(clusters_json: Dict[str, List[str]], out_dir: str, vis_per_cluster: int, copy_mode: bool):
    vis_root = os.path.join(out_dir, "cluster_vis")
    os.makedirs(vis_root, exist_ok=True)

    for cid_str, imgs in tqdm(clusters_json.items(), desc="export cluster_vis"):
        cdir = os.path.join(vis_root, f"cluster_{cid_str.zfill(4)}")
        os.makedirs(cdir, exist_ok=True)
        rep = imgs[:vis_per_cluster]

        if copy_mode:
            for p in rep:
                name = os.path.basename(p)
                dst = os.path.join(cdir, name)
                if os.path.exists(dst):
                    continue
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    img = imread_unicode(p)
                    if img is None:
                        continue
                    cv2.imencode(".jpg", img)[1].tofile(dst)
        else:
            with open(os.path.join(cdir, "list.txt"), "w", encoding="utf-8") as f:
                for p in rep:
                    f.write(p + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    metafile = os.path.join(MMPRETRAIN_ROOT, r"configs\dinov2\metafile.yml")
    cfg_rel, ckpt_url = read_metafile_and_pick_weight(metafile, DINOV2_VARIANT)

    cfg_path = os.path.join(MMPRETRAIN_ROOT, cfg_rel)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print("[INFO] variant =", DINOV2_VARIANT)
    print("[INFO] cfg     =", cfg_path)
    print("[INFO] weights =", ckpt_url)

    # build model
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # load checkpoint (URL -> auto download)
    load_checkpoint(model, ckpt_url, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = dp.get("mean", [123.675, 116.28, 103.53])
    std = dp.get("std", [58.395, 57.12, 57.375])
    to_rgb = bool(dp.get("to_rgb", True))
    img_size = cfg.model["backbone"].get("img_size", 518)

    preprocess = build_preprocess(mean, std, to_rgb, img_size)

    paths = list_images(DATA_ROOT, recursive=True)
    if MAX_IMAGES and MAX_IMAGES > 0:
        paths = paths[:MAX_IMAGES]

    print(f"[INFO] images={len(paths)} img_size={img_size} batch={BATCH_SIZE} device={device}")
    if not paths:
        print("[ERROR] no images found.")
        return

    feats = []
    ok_paths = []
    bt_list, bp_list = [], []

    for p in tqdm(paths, desc="extract embedding"):
        img = imread_unicode(p)
        if img is None:
            continue
        bt_list.append(preprocess(img))
        bp_list.append(p)

        if len(bt_list) >= BATCH_SIZE:
            bt = torch.stack(bt_list, dim=0).to(device)
            emb = extract_embedding_batch(model, bt).cpu().numpy().astype("float32")
            feats.append(emb)
            ok_paths.extend(bp_list)
            bt_list, bp_list = [], []

    if bt_list:
        bt = torch.stack(bt_list, dim=0).to(device)
        emb = extract_embedding_batch(model, bt).cpu().numpy().astype("float32")
        feats.append(emb)
        ok_paths.extend(bp_list)

    embeddings = np.concatenate(feats, axis=0)
    print("[INFO] embeddings:", embeddings.shape)

    # cluster
    print(f"[INFO] clustering K={K} niter={NITER} faiss_gpu={FAISS_GPU}")
    try:
        cluster_ids, centroids = faiss_kmeans_cluster(embeddings, K, NITER, SEED, FAISS_GPU)
    except Exception as e:
        print("[WARN] faiss gpu failed -> cpu fallback:", repr(e))
        cluster_ids, centroids = faiss_kmeans_cluster(embeddings, K, NITER, SEED, False)

    # save
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(OUT_DIR, "cluster_ids.npy"), cluster_ids)
    np.save(os.path.join(OUT_DIR, "centroids.npy"), centroids)

    with open(os.path.join(OUT_DIR, "paths.txt"), "w", encoding="utf-8") as f:
        for p in ok_paths:
            f.write(p + "\n")

    meta = {
        "mmpretrain_root": MMPRETRAIN_ROOT,
        "cfg": cfg_path,
        "weights": ckpt_url,
        "variant": DINOV2_VARIANT,
        "data_root": DATA_ROOT,
        "num_images": len(ok_paths),
        "img_size": img_size,
        "batch_size": BATCH_SIZE,
        "k": K,
        "niter": NITER,
        "seed": SEED,
        "faiss_gpu": FAISS_GPU,
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    clusters_json = export_clusters(ok_paths, cluster_ids, OUT_DIR)
    export_cluster_vis(clusters_json, OUT_DIR, VIS_PER_CLUSTER, COPY_MODE)

    print("\n[DONE] out_dir =", OUT_DIR)
    print("  - clusters.json")
    print("  - cluster_vis/cluster_XXXX/")
    print("  - embeddings.npy / cluster_ids.npy / centroids.npy / paths.txt")


if __name__ == "__main__":
    main()
