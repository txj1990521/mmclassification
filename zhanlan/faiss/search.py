# search.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
OUT_VIZ = r"D:\zhanlan\topk_viz1.jpg"  # 输出拼图
TILE = 320  # 每张图显示尺寸（正方形，越大越清晰）

QUERY_IMG = r"D:\zhanlan\qurrey_data\003b.jpg"

OUT_INDEX = r"D:\zhanlan\faiss.index"
OUT_META  = r"D:\zhanlan\faiss_paths.npy"

TOPK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def to_tensor_normalized(img_bgr: np.ndarray, mean, std, to_rgb=True):
    if to_rgb:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = img_bgr[:, :, ::-1].copy()

    img = resize_short_edge(img, RESIZE_SHORT)
    img = center_crop(img, CROP_SIZE)

    x = img.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def _put_text(img, text, org=(8, 26), font_scale=0.8, thickness=2):
    """在图上画字（带黑边）"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _fit_square(img_bgr, tile=320):
    """等比缩放后居中放进 tile x tile 的黑底方块"""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((tile, tile, 3), dtype=np.uint8)

    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def make_topk_viz(query_path: str,
                  top_paths: list,
                  top_scores: list,
                  out_path: str,
                  tile: int = 320):
    """
    生成拼图：第一格是 Query，后面依次 TopK
    保存为 out_path（jpg/png 都行）
    """
    qimg = imread_unicode(query_path)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image for viz: {query_path}")

    # 1 + K 张
    imgs = []
    labels = []

    # Query
    imgs.append(_fit_square(qimg, tile))
    labels.append("QUERY")

    # TopK
    for i, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        img = imread_unicode(str(p))
        if img is None:
            img = np.zeros((tile, tile, 3), dtype=np.uint8)
            imgs.append(img)
            labels.append(f"#{i}  {float(s):.4f}\n(read fail)")
        else:
            imgs.append(_fit_square(img, tile))
            labels.append(f"#{i}  {float(s):.4f}")

    n = len(imgs)

    # 计算网格：尽量接近正方形
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    gap = 8  # 图块间距
    header = 40  # 每块上方留出写字空间

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols

        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)

        # 贴图
        canvas[y + header:y + header + tile, x:x + tile] = imgs[idx]

        # 写字（支持两行简单处理）
        lines = labels[idx].split("\n")
        for li, line in enumerate(lines):
            _put_text(canvas, line, org=(x + 8, y + 26 + li * 22), font_scale=0.7, thickness=2)

    # 保存（支持中文路径）
    ext = Path(out_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise ValueError("out_path extension should be an image format like .jpg/.png")

    ok, buf = cv2.imencode(ext, canvas)
    if not ok:
        raise RuntimeError("cv2.imencode failed when saving viz")
    buf.tofile(out_path)

    return out_path

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
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)

    # 1) load index + meta
    index = faiss.read_index(OUT_INDEX)
    kept_paths = np.load(OUT_META, allow_pickle=True)
    print(f"[INFO] Loaded index ntotal={index.ntotal}, meta size={len(kept_paths)}")

    if index.ntotal != len(kept_paths):
        print("[WARN] index.ntotal != len(meta). Meta/index may be mismatched!")

    # 2) load model (only for query feature extraction)
    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    # 3) read query
    qimg = imread_unicode(QUERY_IMG)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")

    qx = to_tensor_normalized(qimg, mean, std, to_rgb=to_rgb).unsqueeze(0).to(DEVICE)
    qfeat = extract_backbone_2048(model, qx).cpu().numpy().astype("float32")

    # 4) search
    scores, ids = index.search(qfeat, TOPK)
    scores = scores[0]
    ids = ids[0]

    print("\n===== TOPK =====")
    for r, (s, idx) in enumerate(zip(scores, ids), 1):
        if idx < 0:
            continue
        path = kept_paths[idx]
        print(f"{r:02d}  score={float(s):.4f}  {path}")
# 5) collect topk paths for viz
    top_paths = []
    top_scores = []
    for s, idx in zip(scores, ids):
        if idx < 0:
            continue
        top_paths.append(kept_paths[idx])
        top_scores.append(float(s))

    out = make_topk_viz(
        query_path=QUERY_IMG,
        top_paths=top_paths,
        top_scores=top_scores,
        out_path=OUT_VIZ,
        tile=TILE
    )
    print(f"\n[VIZ] saved -> {out}")

if __name__ == "__main__":
    main()
