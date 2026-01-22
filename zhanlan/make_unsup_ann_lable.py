#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path
import numpy as np

# =========================
# CONFIG（直接改这里）
# =========================

DATA_ROOT = r"D:\zhanlan\data"      # 数据根目录（图片都在这里或其子目录）
OUT_FILE = r"D:\zhanlan\ann\train_with_label.txt"  # 输出 annotation 文件（可绝对/相对）
DUMMY_LABEL = 0                    # 统一假标签（ImageNet 数据集要求第二列）

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
    """兼容中文路径读取"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data is None or data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# =========================
# 主逻辑
# =========================

def main():
    data_root = Path(DATA_ROOT)
    assert data_root.exists(), f"DATA_ROOT not found: {data_root}"

    out_path = Path(OUT_FILE)
    if not out_path.is_absolute():
        out_path = data_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 遍历图片
    candidates = data_root.rglob("*") if RECURSIVE else data_root.glob("*")

    print(f"[INFO] Scanning images in: {data_root}")

    lines = []
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
                print(f"[WARN] Image too small ({w}x{h}), skip: {p}")
                continue

        # 写成相对 DATA_ROOT 的 posix 路径（mmpretrain 更稳）
        rel = p.relative_to(data_root).as_posix()

        # 关键：两列 "path label"
        lines.append(f"{rel} {DUMMY_LABEL}")

    lines.sort()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print("=" * 60)
    print(f"[DONE] {len(lines)} images written to:")
    print(str(out_path))
    print(f"[FORMAT] <relative_path> <label>, label={DUMMY_LABEL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
