#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path
import numpy as np

# =========================
# CONFIG（直接改这里）
# =========================

DATA_ROOT = r"D:\zhanlan\ClassifyData"      # 数据根目录（图片在这里或其子目录）
OUT_FILE = r"D:\zhanlan\ClassifyData\ann\train_with_label.txt"  # 输出 annotation 文件（可绝对/相对）

# 模式开关：
# - "unsupervised": 无监督/自监督预训练常用：统一假标签
# - "supervised":   按文件夹名生成正常标签（适合分类训练/微调）
MODE = "supervised"

UNSUP_LABEL = 0                    # MODE=unsupervised 时使用的统一假标签

# 标签来源：用图片相对 DATA_ROOT 的第一级目录名作为类别名
# 例如 DATA_ROOT/grid/a.jpg -> class_name="grid"
LABEL_FROM_LEVEL = 1               # 1=第一级子目录；2=第二级…（一般 1 就够了）

RECURSIVE = True
CHECK_IMAGE = True                 # 是否读取图片检查损坏
MIN_SIZE = 64                      # 过滤太小的图片（边长）

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 可选：是否把类名->id 映射也写到同目录下一个文件里，方便你对照
SAVE_CLASS_MAP = True
CLASS_MAP_FILE = "classes.txt"     # 保存为：id<tab>class_name

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

def get_class_name_from_path(rel_path: Path, level: int) -> str:
    """
    从相对路径中取第 level 级目录名作为类名
    level=1 -> rel_path.parts[0]
    """
    parts = rel_path.parts
    if len(parts) <= level - 1:
        # 图片直接在根目录下，没有子目录，无法推断类名
        return ""
    return parts[level - 1]

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

    candidates = data_root.rglob("*") if RECURSIVE else data_root.glob("*")
    print(f"[INFO] Scanning images in: {data_root}")
    print(f"[INFO] MODE = {MODE}")

    lines = []

    # supervised 模式下才需要建立 class_name -> id
    class_to_id = {}
    next_id = 0

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

        rel = p.relative_to(data_root)          # Path
        rel_posix = rel.as_posix()              # mmpretrain 更稳

        if MODE.lower() == "unsupervised":
            label = UNSUP_LABEL

        elif MODE.lower() == "supervised":
            class_name = get_class_name_from_path(rel, LABEL_FROM_LEVEL)
            if not class_name:
                # 没有类目录，无法生成标签
                print(f"[WARN] No class folder found for: {rel_posix} (need level={LABEL_FROM_LEVEL}), skip.")
                continue

            if class_name not in class_to_id:
                class_to_id[class_name] = next_id
                next_id += 1

            label = class_to_id[class_name]

        else:
            raise ValueError(f"Unknown MODE: {MODE}. Use 'unsupervised' or 'supervised'.")

        lines.append(f"{rel_posix} {label}")

    lines.sort()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    # 可选保存类别映射（只对 supervised 有意义）
    if MODE.lower() == "supervised" and SAVE_CLASS_MAP:
        map_path = out_path.parent / CLASS_MAP_FILE
        # 按 id 排序写出
        items = sorted(class_to_id.items(), key=lambda kv: kv[1])
        with open(map_path, "w", encoding="utf-8") as f:
            for name, idx in items:
                f.write(f"{idx}\t{name}\n")
        print(f"[INFO] Class map saved to: {map_path}")

    print("=" * 60)
    print(f"[DONE] {len(lines)} images written to:")
    print(str(out_path))
    if MODE.lower() == "unsupervised":
        print(f"[FORMAT] <relative_path> <label>, label={UNSUP_LABEL} (all same)")
    else:
        print(f"[FORMAT] <relative_path> <label>, label from folder level={LABEL_FROM_LEVEL}")
        print(f"[CLASSES] {len(class_to_id)} classes: {sorted(class_to_id.keys())}")
    print("=" * 60)

if __name__ == "__main__":
    main()
