#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

# =========================
# CONFIG
# =========================

INPUT_PATH = r"D:/zhanlan/test_data"
OUTPUT_DIR = r"D:/zhanlan/output_fg"

RECURSIVE = True
MAX_LONG_SIDE = 2500

# 前景阈值（经验值，后面告诉你怎么调）
COLOR_DIFF_THR = 25     # 颜色显著性
EDGE_THR = 30           # 梯度强度
MIN_AREA_RATIO = 0.0005  # 最小实例面积（占整图比例）
EXPAND_RATIO = 0.25     # 外扩比例

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# 工具
# =========================

def is_image(p):
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_images(path):
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            if is_image(os.path.join(root, f)):
                imgs.append(os.path.join(root, f))
        if not RECURSIVE:
            break
    return imgs

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)

def resize_max(img):
    h, w = img.shape[:2]
    long = max(h, w)
    if long <= MAX_LONG_SIDE:
        return img, 1.0
    scale = MAX_LONG_SIDE / long
    return cv2.resize(img, (int(w*scale), int(h*scale))), scale

# =========================
# 核心：前景分割
# =========================

def segment_foreground(resid_maskimg, out_debug_dir=None):
    """
    更稳的前景分割：
    1) 颜色分支：HSV 饱和度 + Lab(a,b) 偏离（适合彩色花）
    2) 浮雕/刺绣分支：灰度 - 大尺度模糊(背景) 的残差（适合同色刺绣）
    最后做形态学连通
    """
    H, W = img.shape[:2]

    # -----------------------
    # A) 颜色分支（彩色花很强）
    # -----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)

    # 饱和度阈值：对彩色花特别有效
    s_thr = 25  # 可调：30~60
    mask_sat = (S > s_thr).astype(np.uint8) * 255

    # Lab 的 a,b 偏离：补充一些低饱和彩色
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)
    med_a, med_b = np.median(A), np.median(B)
    ab_dist = np.sqrt((A - med_a) ** 2 + (B - med_b) ** 2)
    ab_thr = 18  # 可调：15~30
    mask_ab = (ab_dist > ab_thr).astype(np.uint8) * 255

    color_mask = cv2.bitwise_and(mask_sat, mask_ab)

    # -----------------------
    # B) 浮雕/刺绣分支（关键：去掉布底纹理）
    #    用“背景建模残差”而不是梯度
    # -----------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 大尺度模糊作为背景（sigma 越大越能抹掉斜纹底）
    # 手机布料纹理强：建议 sigma=25~45
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)

    # 残差：刺绣/花型会留下明显局部差异，布底斜纹会被抹掉
    resid = cv2.absdiff(gray, bg)

    # 归一化到 0~255，便于阈值/可视化
    resid_u8 = cv2.normalize(resid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 自适应阈值：用百分位比固定阈值更稳（适应曝光）
    # 97~99 越高越“只保留凸起/花”
    r_thr = np.percentile(resid_u8, 90)
    resid_mask = (resid_u8 >= r_thr).astype(np.uint8) * 255

    # -----------------------
    # C) 合并 + 形态学连通
    # -----------------------
    # ===== 自动选择：如果颜色前景已经足够，就不要 resid_mask（resid 会把布底纹理带进来）=====
    color_area = float(cv2.countNonZero(color_mask)) / (H * W)
    resid_area = np.count_nonzero(resid_mask) / (H * W)

    # 经验阈值（后面可调）
    COLOR_DOM_THR = 0.003  # 彩色花
    RESID_DOM_THR = 0.002  # 同色刺绣

    if color_area > COLOR_DOM_THR and color_area > resid_area * 1.5:
        # 明确彩色主导
        fg = color_mask
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 执行闭运算
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        # 去掉散点噪声：只保留面积较大的连通域（在 fg 还没 close 前做更有效）
        num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        clean = np.zeros_like(fg)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 0.01 * H * W:  # 可调：0.0001~0.001
                continue
            clean[lab == i] = 255
        fg = clean
    elif resid_area > RESID_DOM_THR and resid_area > color_area * 1.5:
        # 明确纹理主导（同色刺绣/提花）
        fg = resid_mask
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 执行闭运算
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        # 去掉散点噪声：只保留面积较大的连通域（在 fg 还没 close 前做更有效）
        num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        clean = np.zeros_like(fg)
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 0.0001 * H * W:  # 可调：0.0001~0.001
                continue
            clean[lab == i] = 255
        fg = clean
    else:
        # 两者都不强 or 混合情况
        fg = cv2.bitwise_or(color_mask, resid_mask)

    # # 先轻微膨胀：把细茎叶“加粗”，避免后续断掉
    # fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1)
    #
    # # 再用较大的 close：把同一枝花连起来
    # fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    #
    # # ⚠️ 不要 OPEN（会把细线结构开掉）
    # # 如确实有噪声，用很小的 open
    # fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # -----------------------
    # D) 可选：输出中间结果，方便你调参
    # -----------------------
    if out_debug_dir is not None:
        os.makedirs(out_debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_debug_dir, "mask_sat.jpg"), mask_sat)
        cv2.imwrite(os.path.join(out_debug_dir, "mask_ab.jpg"), mask_ab)
        cv2.imwrite(os.path.join(out_debug_dir, "color_mask.jpg"), color_mask)
        cv2.imwrite(os.path.join(out_debug_dir, "resid_u8.jpg"), resid_u8)
        cv2.imwrite(os.path.join(out_debug_dir, "resid_mask.jpg"), resid_mask)
        cv2.imwrite(os.path.join(out_debug_dir, "fg.jpg"), fg)

    return fg

# =========================
# 主流程
# =========================

def process_one(img_path):
    img0 = imread_unicode(img_path)
    if img0 is None:
        return

    img, scale = resize_max(img0)
    Hs, Ws = img.shape[:2]  # resized
    H0, W0 = img0.shape[:2] # original

    name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    fg = segment_foreground(img, out_debug_dir=os.path.join(out_dir, "_masks"))

    # 连通域
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)

    # 收集候选（跳过背景0）
    cand = []
    for cid in range(1, num):
        x, y, w, h, area = stats[cid]
        cand.append((area, x, y, w, h))

    cand.sort(reverse=True)  # 面积从大到小

    min_area = MIN_AREA_RATIO * Hs * Ws

    boxes = []
    max_keep = 5  # 最多保留几个实例框（可调）

    # 先按阈值筛选
    for area, x, y, w, h in cand:
        if area < min_area:
            continue

        # 外扩（在 resized 尺度上）
        pad_x = int(w * EXPAND_RATIO)
        pad_y = int(h * EXPAND_RATIO)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(Ws, x + w + pad_x)
        y2 = min(Hs, y + h + pad_y)

        # 映射回原图
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        # clamp
        x1 = max(0, min(x1, W0 - 1))
        y1 = max(0, min(y1, H0 - 1))
        x2 = max(1, min(x2, W0))
        y2 = max(1, min(y2, H0))

        boxes.append([x1, y1, x2 - x1, y2 - y1])

        if len(boxes) >= max_keep:
            break

    # 如果筛选后为空：保底取最大2个
    if len(boxes) == 0 and len(cand) > 0:
        for area, x, y, w, h in cand[:2]:
            pad_x = int(w * EXPAND_RATIO)
            pad_y = int(h * EXPAND_RATIO)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(Ws, x + w + pad_x)
            y2 = min(Hs, y + h + pad_y)
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            x1 = max(0, min(x1, W0 - 1))
            y1 = max(0, min(y1, H0 - 1))
            x2 = max(1, min(x2, W0))
            y2 = max(1, min(y2, H0))
            boxes.append([x1, y1, x2 - x1, y2 - y1])

    # debug
    dbg = img0.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
    imwrite_unicode(os.path.join(out_dir, "debug.jpg"), dbg)

    # 保存裁剪
    for i, (x, y, w, h) in enumerate(boxes):
        crop = img0[y:y + h, x:x + w]
        imwrite_unicode(os.path.join(out_dir, f"{i:03d}.jpg"), crop)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    imgs = list_images(INPUT_PATH)
    print(f"Found {len(imgs)} images")
    for i,p in enumerate(imgs,1):
        print(f"[{i}] {os.path.basename(p)}")
        process_one(p)
