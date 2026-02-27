#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# ====== CONFIG ======
IMAGES_META_JSON = r"D:\zhanlanProject\openai_search\outputs_hybrid_folder_big\images_meta.json"  # 你CLIP用的那份
CFG_PATH = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py"
CKPT_PATH = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

OUT_DIR = r"D:\zhanlan\faiss_database_simclr_aligned"
GLOBAL_INDEX_PATH = os.path.join(OUT_DIR, "simclr_global.index")

# 重要：为了“对齐可用”，输出一个向量id -> img_id 映射
VEC_TO_IMGID_PATH = os.path.join(OUT_DIR, "simclr_vec_to_imgid.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
INPUT_SIZE = 224

# ====== Dynamic strategy ======
# 你可以根据机器情况调整阈值
FLAT_MAX_N = 50_000          # <= 5w 用 FlatIP
IVF_MAX_N  = 500_000         # 5w~50w 用 IVF-Flat，>50w 用 IVF-PQ

# IVF params
NPROBE = 32                  # 建库时可设，查询时也用
IVF_NLIST_MIN = 1024
IVF_NLIST_MAX = 65536

# train sampling
TRAIN_MAX = 300_000          # 训练最多采样多少条向量
TRAIN_PER_LIST = 100         # train_size ~ nlist * TRAIN_PER_LIST

# PQ params (for IVFPQ)
PQ_M = 32                    # 2048 能被 32 整除 ✅
PQ_NBITS = 8

# ====== utils ======
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def build_model(cfg_path, ckpt_path):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(DEVICE)
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

def prep_tensor(img_bgr, mean, std, to_rgb):
    img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) - mean) / std
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)

def pick_nlist(N: int) -> int:
    # 常用经验：4*sqrt(N)
    nlist = int(4 * np.sqrt(max(N, 1)))
    nlist = max(IVF_NLIST_MIN, nlist)
    nlist = min(IVF_NLIST_MAX, nlist)
    # 不要让 nlist 太接近 N
    nlist = min(nlist, max(1, N // 20))
    return max(1, nlist)

def pick_train_size(N: int, nlist: int) -> int:
    # 至少 nlist，最好 nlist*TRAIN_PER_LIST，但最多 TRAIN_MAX
    t = max(nlist, nlist * TRAIN_PER_LIST)
    t = min(t, TRAIN_MAX)
    t = min(t, N)
    return int(t)

def make_index(strategy: str, D: int, nlist: int):
    """
    strategy: 'flat' | 'ivfflat' | 'ivfpq'
    """
    if strategy == "flat":
        return faiss.IndexFlatIP(D)

    quantizer = faiss.IndexFlatIP(D)
    if strategy == "ivfflat":
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        return index

    if strategy == "ivfpq":
        if D % PQ_M != 0:
            raise ValueError(f"PQ_M={PQ_M} must divide D={D}. Please change PQ_M.")
        index = faiss.IndexIVFPQ(quantizer, D, nlist, PQ_M, PQ_NBITS, faiss.METRIC_INNER_PRODUCT)
        return index

    raise ValueError(f"Unknown strategy: {strategy}")

def set_nprobe(index, nprobe: int):
    try:
        # 兼容包裹索引
        base = index
        while hasattr(base, "index"):
            base = base.index
        if hasattr(base, "nprobe") and hasattr(base, "nlist"):
            base.nprobe = min(int(nprobe), int(base.nlist))
            print(f"[FAISS] nprobe={base.nprobe}/{base.nlist}")
    except Exception:
        pass

# ====== feature iteration (streaming) ======
def iter_batches(images_meta, mean, std, to_rgb, model, batch_size: int):
    """
    yield: (batch_feats_np_float32, batch_img_ids_list, bad_list)
    """
    buf_t = []
    buf_ids = []
    bad = []

    for item in images_meta:
        img_id = int(item["img_id"])
        p = item["abs_path"]

        if not os.path.exists(p):
            bad.append((img_id, p))
            continue

        img = imread_unicode(p)
        if img is None:
            bad.append((img_id, p))
            continue

        buf_t.append(prep_tensor(img, mean, std, to_rgb))
        buf_ids.append(img_id)

        if len(buf_t) >= batch_size:
            bt = torch.stack(buf_t, dim=0).to(DEVICE, non_blocking=True)
            fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
            yield fv, buf_ids, bad
            buf_t, buf_ids, bad = [], [], []

    if buf_t:
        bt = torch.stack(buf_t, dim=0).to(DEVICE, non_blocking=True)
        fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        yield fv, buf_ids, bad

def main():
    ensure_dir(OUT_DIR)

    with open(IMAGES_META_JSON, "r", encoding="utf-8") as f:
        images_meta = json.load(f)

    # 保证按 img_id 顺序
    images_meta = sorted(images_meta, key=lambda d: int(d["img_id"]))
    N_total = len(images_meta)
    print(f"[INFO] images_meta total = {N_total}")

    model, mean, std, to_rgb = build_model(CFG_PATH, CKPT_PATH)

    # ---- probe D (取第一批) ----
    # 先跑一个很小的 batch 探测维度（避免你手填 D）
    probe_batch = []
    probe_ids = []
    probe_bad = []

    for item in images_meta:
        img_id = int(item["img_id"])
        p = item["abs_path"]
        if not os.path.exists(p):
            probe_bad.append((img_id, p))
            continue
        img = imread_unicode(p)
        if img is None:
            probe_bad.append((img_id, p))
            continue
        probe_batch.append(prep_tensor(img, mean, std, to_rgb))
        probe_ids.append(img_id)
        if len(probe_batch) >= min(8, BATCH):
            break

    if not probe_batch:
        raise RuntimeError("No valid images found in images_meta.json (all missing/unreadable).")

    bt = torch.stack(probe_batch, dim=0).to(DEVICE)
    fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
    D = int(fv.shape[1])
    print(f"[INFO] feature dim D = {D} (expected 2048 for ResNet50 stage4 GAP)")

    # ---- choose strategy ----
    if N_total <= FLAT_MAX_N:
        strategy = "flat"
    elif N_total <= IVF_MAX_N:
        strategy = "ivfflat"
    else:
        strategy = "ivfpq"

    nlist = pick_nlist(N_total) if strategy != "flat" else 0
    print(f"[STRATEGY] N={N_total} -> {strategy}  nlist={nlist if nlist else '-'}")

    # ---- build index ----
    index = make_index(strategy, D, nlist)

    # ---- pass-1: train (if IVF) ----
    bad_all = []
    vec_to_imgid = []  # 记录成功写入的向量对应 img_id（索引内部序号 -> img_id）

    if strategy != "flat":
        train_size = pick_train_size(N_total, nlist)
        print(f"[TRAIN] target train_size={train_size}")

        # 采样规则：均匀抽样（不需要随机，确保可复现）
        step = max(1, N_total // train_size)
        picked = images_meta[::step]
        # 由于 step 可能导致超过 train_size，这里裁一下
        picked = picked[:train_size]
        print(f"[TRAIN] picked={len(picked)} step={step}")

        train_feats_chunks = []
        seen = 0
        for feats, ids, bad in iter_batches(picked, mean, std, to_rgb, model, batch_size=BATCH):
            train_feats_chunks.append(feats)
            seen += feats.shape[0]
            bad_all.extend(bad)
            if seen >= train_size:
                break

        if not train_feats_chunks:
            raise RuntimeError("No train feats collected (check image reading).")

        train_x = np.concatenate(train_feats_chunks, axis=0).astype("float32")
        print(f"[TRAIN] actual train_x shape={train_x.shape}")

        print("[TRAIN] faiss index training ...")
        index.train(train_x)
        print("[TRAIN] done.")

    # ---- pass-2: add all (streaming) ----
    print("[ADD] streaming add ...")
    added = 0
    for feats, ids, bad in iter_batches(images_meta, mean, std, to_rgb, model, batch_size=BATCH):
        index.add(feats)
        vec_to_imgid.extend(ids)
        added += feats.shape[0]
        bad_all.extend(bad)

        if added % (BATCH * 50) == 0:
            print(f"[ADD] added={added}/{N_total}  index.ntotal={index.ntotal}")

    set_nprobe(index, NPROBE)

    # ---- save ----
    faiss.write_index(index, GLOBAL_INDEX_PATH)
    np.save(VEC_TO_IMGID_PATH, np.array(vec_to_imgid, dtype=np.int32))

    # bad log
    if bad_all:
        with open(os.path.join(OUT_DIR, "bad_images.txt"), "w", encoding="utf-8") as f:
            for img_id, p in bad_all:
                f.write(f"{img_id}\t{p}\n")

    print(f"[OK] saved index: {GLOBAL_INDEX_PATH}")
    print(f"[OK] saved vec->img_id map: {VEC_TO_IMGID_PATH} (len={len(vec_to_imgid)})")
    print(f"[OK] ntotal={index.ntotal}  D={D}  bad={len(bad_all)}")

if __name__ == "__main__":
    main()