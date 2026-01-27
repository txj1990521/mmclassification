#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from pathlib import Path
import numpy as np
import random
import shutil
from collections import defaultdict

# =========================
# CONFIG（直接改这里）
# =========================

DATA_ROOT = r"D:\zhanlan\new_data"           # 原始数据根目录
OUT_DIR = r"D:\zhanlan\newData\ann"              # 输出标注目录（train.txt/test.txt/classes.txt）
SPLIT_OUT_ROOT = r"D:\zhanlan\newData\split_data"  # 拆分后复制到这里（train/ test 子目录）

MODE = "unsupervised"  # "unsupervised" 或 "supervised"

# 无监督模式参数
UNSUP_LABEL = 0
UNSUP_FILE = "unsup.txt"

# 有监督模式参数
LABEL_FROM_LEVEL = 1
TEST_RATIO = 0.2
SEED = 42
MIN_PER_CLASS_TEST = 1

# 复制拆分图片
COPY_SPLIT_IMAGES = True
OVERWRITE_EXISTING = False          # 目标文件存在时是否覆盖

RECURSIVE = True
CHECK_IMAGE = True
MIN_SIZE = 64

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
CLASS_MAP_FILE = "classes.txt"

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

def get_class_name_from_relpath(rel_path: Path, level: int) -> str:
    parts = rel_path.parts
    if len(parts) <= level - 1:
        return ""
    return parts[level - 1]

def scan_images(data_root: Path, recursive: bool = True):
    candidates = data_root.rglob("*") if recursive else data_root.glob("*")
    for p in candidates:
        if p.is_file() and is_image(p):
            yield p

def check_ok(p: Path) -> bool:
    if not CHECK_IMAGE:
        return True
    img = imread_unicode(p)
    if img is None:
        print(f"[WARN] Cannot read image, skip: {p}")
        return False
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIZE:
        print(f"[WARN] Image too small ({w}x{h}), skip: {p}")
        return False
    return True

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def copy_one(src: Path, dst: Path):
    ensure_parent(dst)
    if dst.exists() and not OVERWRITE_EXISTING:
        return
    shutil.copy2(src, dst)

def save_class_map(out_path: Path, class_to_id: dict):
    items = sorted(class_to_id.items(), key=lambda kv: kv[1])
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for name, idx in items:
            f.write(f"{idx}\t{name}\n")

def write_ann(out_path: Path, rel_paths, label_getter):
    lines = [f"{rp} {label_getter(rp)}" for rp in rel_paths]
    lines.sort()
    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

# =========================
# 主逻辑
# =========================

def main():
    data_root = Path(DATA_ROOT)
    assert data_root.exists(), f"DATA_ROOT not found: {data_root}"

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_out_root = Path(SPLIT_OUT_ROOT)
    if MODE.lower() == "supervised" and COPY_SPLIT_IMAGES:
        split_out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning images in: {data_root}")
    print(f"[INFO] MODE = {MODE}")

    all_imgs = list(scan_images(data_root, RECURSIVE))
    print(f"[INFO] Found {len(all_imgs)} image files (before check/filter)")

    # ---------------- unsupervised ----------------
    if MODE.lower() == "unsupervised":
        kept_rel = []
        for p in all_imgs:
            if not check_ok(p):
                continue
            rel = p.relative_to(data_root).as_posix()
            kept_rel.append(rel)

        out_path = out_dir / UNSUP_FILE
        write_ann(out_path, kept_rel, label_getter=lambda _: UNSUP_LABEL)

        print("=" * 60)
        print(f"[DONE] {len(kept_rel)} images written to: {out_path}")
        print(f"[FORMAT] <relative_path> <label>, label={UNSUP_LABEL} (all same)")
        print("=" * 60)
        return

    # ---------------- supervised ----------------
    if MODE.lower() != "supervised":
        raise ValueError(f"Unknown MODE: {MODE}. Use 'unsupervised' or 'supervised'.")

    # 1) 按类收集（并且先过滤掉坏图/小图）
    by_class = defaultdict(list)
    missing_class = 0
    checked_total = 0

    for p in all_imgs:
        rel_path = p.relative_to(data_root)
        cname = get_class_name_from_relpath(rel_path, LABEL_FROM_LEVEL)
        if not cname:
            missing_class += 1
            continue
        if not check_ok(p):
            continue
        checked_total += 1
        by_class[cname].append(p)

    if missing_class > 0:
        print(f"[WARN] {missing_class} images have no class folder (level={LABEL_FROM_LEVEL}), skipped.")

    class_names = sorted(by_class.keys())
    if not class_names:
        raise RuntimeError("No valid classes/images found. Check folder structure and LABEL_FROM_LEVEL.")

    # 2) 类名->id
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    save_class_map(out_dir / CLASS_MAP_FILE, class_to_id)
    print(f"[INFO] Valid images after check/filter: {checked_total}")
    print(f"[INFO] Classes: {class_to_id}")
    print(f"[INFO] Class map saved to: {out_dir / CLASS_MAP_FILE}")

    # 3) 分层拆分 train/test
    rng = random.Random(SEED)
    train_files = []
    test_files = []

    for cname in class_names:
        files = by_class[cname]
        rng.shuffle(files)
        n = len(files)

        if n == 1:
            train_files.extend(files)
            continue

        n_test = int(round(n * TEST_RATIO))
        n_test = max(n_test, MIN_PER_CLASS_TEST)
        n_test = min(n_test, n - 1)

        test_files.extend(files[:n_test])
        train_files.extend(files[n_test:])

    print(f"[INFO] Split done: train={len(train_files)}, test={len(test_files)}")

    # 4) 复制到新目录（保持 train/class_name/xxx.jpg 结构）
    #    复制后 ann 用“相对各自 split 根目录”的路径： class_name/xxx.jpg
    train_rel_for_ann = []
    test_rel_for_ann = []

    if COPY_SPLIT_IMAGES:
        train_root = split_out_root / "train"
        test_root = split_out_root / "test"
        train_root.mkdir(parents=True, exist_ok=True)
        test_root.mkdir(parents=True, exist_ok=True)

        def dst_path(split_root: Path, src: Path) -> Path:
            rel = src.relative_to(data_root)
            cname = get_class_name_from_relpath(rel, LABEL_FROM_LEVEL)
            # 复制时，取“从类目录开始”的相对路径：class_name/后面的层级/文件名
            # 例如 原始：grid/sub/a.jpg -> 复制为 train/grid/sub/a.jpg
            parts = rel.parts
            # 从 class folder 的 index 开始截取
            start_idx = LABEL_FROM_LEVEL - 1
            rel_from_class = Path(*parts[start_idx:])
            return split_root / rel_from_class

        # copy train
        for src in train_files:
            rel = src.relative_to(data_root)
            cname = get_class_name_from_relpath(rel, LABEL_FROM_LEVEL)
            dst = dst_path(train_root, src)
            copy_one(src, dst)
            # ann 中写相对于 train_root 的路径
            rel_ann = dst.relative_to(train_root).as_posix()
            train_rel_for_ann.append(rel_ann)

        # copy test
        for src in test_files:
            rel = src.relative_to(data_root)
            cname = get_class_name_from_relpath(rel, LABEL_FROM_LEVEL)
            dst = dst_path(test_root, src)
            copy_one(src, dst)
            rel_ann = dst.relative_to(test_root).as_posix()
            test_rel_for_ann.append(rel_ann)

        print(f"[INFO] Images copied to: {split_out_root}")
        print(f"       - train root: {train_root}")
        print(f"       - test  root: {test_root}")

        # label getter：ann 里的相对路径第一段就是类名
        def label_getter(rel_posix: str) -> int:
            relp = Path(rel_posix)
            cname = get_class_name_from_relpath(relp, 1)  # 因为现在 ann 相对 split_root，从第一级就是类名
            return class_to_id[cname]

        # 5) 写 ann（相对 split_root）
        train_out = out_dir / TRAIN_FILE
        test_out = out_dir / TEST_FILE
        write_ann(train_out, train_rel_for_ann, label_getter=label_getter)
        write_ann(test_out, test_rel_for_ann, label_getter=label_getter)

    else:
        # 不复制时：ann 写相对原 DATA_ROOT 的路径
        train_rel_for_ann = [p.relative_to(data_root).as_posix() for p in train_files]
        test_rel_for_ann = [p.relative_to(data_root).as_posix() for p in test_files]

        def label_getter(rel_posix: str) -> int:
            relp = Path(rel_posix)
            cname = get_class_name_from_relpath(relp, LABEL_FROM_LEVEL)
            return class_to_id[cname]

        train_out = out_dir / TRAIN_FILE
        test_out = out_dir / TEST_FILE
        write_ann(train_out, train_rel_for_ann, label_getter=label_getter)
        write_ann(test_out, test_rel_for_ann, label_getter=label_getter)

    print("=" * 60)
    print(f"[DONE] Train ann: {out_dir / TRAIN_FILE} ({len(train_rel_for_ann)} samples)")
    print(f"[DONE] Test  ann: {out_dir / TEST_FILE} ({len(test_rel_for_ann)} samples)")
    print(f"[DONE] Classes : {out_dir / CLASS_MAP_FILE} ({len(class_to_id)} classes)")
    print("=" * 60)

if __name__ == "__main__":
    main()
