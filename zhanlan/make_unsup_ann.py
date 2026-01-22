#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path
import numpy as np
# =========================
# CONFIG（也可改成命令行）
# =========================

DATA_ROOT = r"D:\zhanlan\data"   # 数据根目录 图片目录（相对 DATA_ROOT）
OUT_FILE = "train.txt"             # 输出 annotation 文件

RECURSIVE = True
CHECK_IMAGE = True                 # 是否读取图片检查损坏
MIN_SIZE = 64                      # 过滤太小的图片（边长）

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# 工具
# =========================

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# =========================
# 主逻辑
# =========================

def main():
    data_root = Path(DATA_ROOT)
    img_root = data_root
    assert img_root.exists(), f"Image dir not found: {img_root}"

    out_path = data_root / OUT_FILE
    img_paths = []

    # 遍历图片
    if RECURSIVE:
        candidates = img_root.rglob("*")
    else:
        candidates = img_root.glob("*")

    print(f"[INFO] Scanning images in: {img_root}")

    for p in candidates:
        if not p.is_file():
            continue
        if not is_image(p):
            continue

        if CHECK_IMAGE:
            img = imread_unicode(p)
            if img is None:
                print(f"[WARN] Cannot read image, skip: {p}")
                continue
            h, w = img.shape[:2]
            if min(h, w) < MIN_SIZE:
                print(f"[WARN] Image too small, skip: {p}")
                continue

        # 关键：写成相对 data_root 的路径
        rel = p.relative_to(data_root).as_posix()
        img_paths.append(rel)

    img_paths.sort()

    # 写 annotation
    with open(out_path, "w", encoding="utf-8") as f:
        for p in img_paths:
            f.write(p + "\n")

    print("=" * 50)
    print(f"[DONE] {len(img_paths)} images written to:")
    print(out_path)
    print("=" * 50)

if __name__ == "__main__":
    main()
