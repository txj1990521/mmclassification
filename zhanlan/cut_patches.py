#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import json
import math
import numpy as np
import mmcv
from typing import List, Tuple, Optional, Dict

# =========================
# CONFIG (直接改这里)
# =========================

INPUT_PATH = r"D:/zhanlan/data"
OUTPUT_DIR = r"D:/zhanlan/output"

RECURSIVE = True
LIMIT_IMAGES = 0

MAX_LONG_SIDE = 2500  # 0 = 不缩放

PATCH_SIZES = [512, 256]
STRIDES = [160, 80]
TOPK_PER_SIZE = [25, 35]

TEXTURE_RATIO_MIN = 0.10
SHARPNESS_MIN_512 = 60.0
SHARPNESS_MIN_256 = 40.0

NMS_IOU = 0.5
DEDUP_CENTER_RATIO = 0.0

EDGE_PENALTY = True
MAKE_DEBUG = True
FOREGROUND_RATIO_MIN = 0.03   # 手机图常用 0.02~0.06

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# 工具函数
# =========================

def is_image_file(p):
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_images(path, recursive=True):
    if os.path.isfile(path):
        return [path] if is_image_file(path) else []
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if is_image_file(p):
                imgs.append(p)
        if not recursive:
            break
    return sorted(imgs)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path)
    return img

def resize_max(img, max_long):
    if max_long <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return img, 1.0
    scale = max_long / long_side
    new = cv2.resize(img, (int(w*scale), int(h*scale)))
    return new, scale

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    iw, ih = max(0, x2-x1), max(0, y2-y1)
    inter = iw * ih
    if inter == 0:
        return 0
    return inter / (aw*ah + bw*bh - inter)

def nms(items, iou_thr):
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    keep = []
    for it in items:
        if all(iou(it["xywh"], k["xywh"]) <= iou_thr for k in keep):
            keep.append(it)
    return keep

# =========================
# 核心评分
# =========================
def imwrite_unicode(path, img, ext=".jpg", quality=95):
    ensure_dir(os.path.dirname(path))
    if ext.lower() in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        params = []
    success, buf = cv2.imencode(ext, img, params)
    if not success:
        raise IOError(f"Failed to encode image for {path}")
    buf.tofile(path)

def score_patch(patch, sharp_min):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)

    texture = mag.mean()
    thr = np.percentile(mag, 70)
    texture_ratio = (mag > thr).mean() if thr > 1e-6 else 0

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = lap.var()

    illum_penalty = 0
    m, s = gray.mean(), gray.std()
    if m < 40 or m > 215:
        illum_penalty += 1
    if s < 18:
        illum_penalty += 1

    edge_penalty = 0
    if EDGE_PENALTY:
        h, w = gray.shape
        b = int(min(h, w) * 0.15)
        border = np.zeros_like(gray, dtype=bool)
        border[:b,:] = border[-b:,:] = border[:,:b] = border[:,-b:] = True
        if mag[border].mean() > mag[~border].mean() * 1.3:
            edge_penalty = 1
        # 前景覆盖率：梯度明显的像素占比（自适应阈值）
    t_fg = np.percentile(mag, 90)  # 更严格一点
    foreground_ratio = float((mag > t_fg).mean()) if t_fg > 1e-6 else 0.0
    # passed = texture_ratio >= TEXTURE_RATIO_MIN and sharpness >= sharp_min
    passed = (texture_ratio >= TEXTURE_RATIO_MIN) and (sharpness >= sharp_min) and (
                foreground_ratio >= FOREGROUND_RATIO_MIN)



    return {
        "texture": texture,
        "texture_ratio": texture_ratio,
        "sharpness": sharpness,
        "illum_penalty": illum_penalty,
        "edge_penalty": edge_penalty,
        "passed": passed,
        "foreground_ratio": foreground_ratio,
    }

# =========================
# 主逻辑
# =========================

def cut_one_image(img_path):
    img0 = imread(img_path)
    if img0 is None:
        return None

    img, scale = resize_max(img0, MAX_LONG_SIDE)
    h, w = img.shape[:2]

    results = []

    for ps, st, topk in zip(PATCH_SIZES, STRIDES, TOPK_PER_SIZE):
        sharp_min = SHARPNESS_MIN_512 if ps >= 512 else SHARPNESS_MIN_256
        candidates = []
        xs = list(range(0, w - ps + 1, st))
        ys = list(range(0, h - ps + 1, st))

        # 关键：补齐右边 / 下边的最后一格
        if xs and xs[-1] != (w - ps):
            xs.append(w - ps)
        if ys and ys[-1] != (h - ps):
            ys.append(h - ps)
        for y in ys:
            for x in xs:
                patch = img[y:y+ps, x:x+ps]
                sc = score_patch(patch, sharp_min)
                if not sc["passed"]:
                    continue
                score = (
                    0.45*sc["texture"] +
                    0.25*sc["texture_ratio"] +
                    0.25*sc["sharpness"] -
                    0.05*sc["illum_penalty"] -
                    0.05*sc["edge_penalty"]
                )
                # candidates.append({
                #     "xywh": [x, y, ps, ps],
                #     "score": float(score),
                #     **sc
                # })
                candidates.append({
                    "xywh": [int(x), int(y), int(ps), int(ps)],
                    "score": float(score),
                    "texture": float(sc["texture"]),
                    "texture_ratio": float(sc["texture_ratio"]),
                    "sharpness": float(sc["sharpness"]),
                    "illum_penalty": int(sc["illum_penalty"]),
                    "edge_penalty": int(sc["edge_penalty"]),
                })

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        selected = nms(candidates, NMS_IOU)[:topk]
        results.extend(selected)

    return img0, results, scale


def run():
    ensure_dir(OUTPUT_DIR)
    images = list_images(INPUT_PATH, RECURSIVE)
    if LIMIT_IMAGES > 0:
        images = images[:LIMIT_IMAGES]

    print(f"Found {len(images)} images")

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {os.path.basename(img_path)}")
        ret = cut_one_image(img_path)
        if ret is None:
            continue
        img0, patches, scale = ret


        name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(OUTPUT_DIR, name)
        ensure_dir(out_dir)
        orig_dst = os.path.join(out_dir, "original.jpg")
        if not os.path.exists(orig_dst):
            with open(img_path, "rb") as fsrc, open(orig_dst, "wb") as fdst:
                fdst.write(fsrc.read())
        patch_dir = os.path.join(out_dir, "patches")
        ensure_dir(patch_dir)

        meta = []
        H, W = img0.shape[:2]
        for i, p in enumerate(patches):
            x, y, w, h = p["xywh"]

            # map to original coords
            x0, y0, w0, h0 = x, y, w, h
            if scale != 1.0:
                x0 = int(x / scale)
                y0 = int(y / scale)
                w0 = int(w / scale)
                h0 = int(h / scale)

            # clamp to image bounds
            x0 = max(0, min(x0, W - 1))
            y0 = max(0, min(y0, H - 1))
            w0 = max(1, min(w0, W - x0))
            h0 = max(1, min(h0, H - y0))

            patch = img0[y0:y0 + h0, x0:x0 + w0]
            imwrite_unicode(os.path.join(patch_dir, f"{i:03d}.jpg"), patch, ".jpg")

            p2 = dict(p)
            p2["xywh"] = [x0, y0, w0, h0]  # store original coords in json
            meta.append(p2)
        meta_out = {
            "image": os.path.basename(img_path),
            "image_size": [H, W],
            "scale": scale,
            "patch_sizes": PATCH_SIZES,
            "patches": meta
        }
        # === 用入选 patch 合并出“图案大框” ===
        mask = np.zeros((H, W), dtype=np.uint8)

        for p in meta:
            x, y, w, h = p["xywh"]
            # 给每个 patch 在 mask 上涂白
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # 平滑 + 形态学，让碎片粘起来
        mask = cv2.GaussianBlur(mask, (0, 0), 7)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 找连通域框
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        motif_boxes = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 0.01 * H * W:  # 太小的噪声丢掉（可调）
                continue
            # 外扩一点边距
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(W - x, w + 2 * pad)
            h = min(H - y, h + 2 * pad)
            motif_boxes.append([int(x), int(y), int(w), int(h)])

        with open(os.path.join(out_dir, "patches.json"), "w", encoding="utf-8") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        if MAKE_DEBUG:
            dbg = img0.copy()
            for p in meta:
                x, y, w, h = p["xywh"]  # 已经是原图坐标
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 红色：合并后的整图案框
            for (x, y, w, h) in motif_boxes:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 4)
            imwrite_unicode(os.path.join(out_dir, "debug.jpg"), dbg, ".jpg")


if __name__ == "__main__":
    run()
