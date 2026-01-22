#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

# =========================
# CONFIG（直接改这里）
# =========================
INPUT_PATH = r"D:/zhanlan/test_data"
OUTPUT_DIR = r"D:/zhanlan/output_dino"
RECURSIVE = True
LIMIT_IMAGES = 0

MAX_LONG_SIDE = 2500  # 0 不缩放

# --- ROI mask（刺绣/提花/部分印花）
ENABLE_COLOR_BRANCH = True
ENABLE_RESID_BRANCH = True

RESID_SIGMA = 55
RESID_PERCENTILE = 97.5

SAT_THR = 25
AB_THR = 16

MIN_COMPONENT_AREA_RATIO = 0.00015

# --- patch 采样（用于实例切割）
PATCH_SIZE = 224
STRIDE = 112
MIN_MASK_COVER = 0.08
MAX_PATCHES_PER_IMAGE = 2000

# --- DINO 模型（timm）
DINO_MODEL_NAME = "vit_small_patch16_224.dino"
BATCH_SIZE = 64
DEVICE = "cuda"  # 没 GPU 改 "cpu"

# --- 聚类（DBSCAN）(实例切割用)
DBSCAN_EPS = 0.18
DBSCAN_MIN_SAMPLES = 6
TOPK_CLUSTERS = 10
BBOX_EXPAND = 0.15
XY_ALPHA = 0.6  # 拼坐标权重

# --- 周期纹理检测（只做标注/分流，不过滤）
ENABLE_PERIODIC_ROUTING = True
PERIODIC_SHARPNESS_THR = 0.58
PERIODIC_MIN_EDGE_PIXELS = 500

# --- (新增) 检索特征输出：全图 + Dense Patches
# 全图 embedding：每张图必出 global_feat.npy
SAVE_GLOBAL_FEAT = True

# 局部检索用的 dense patch embedding：每张图必出 patch_feats.npz
SAVE_DENSE_PATCH_FEATS = True
DENSE_PATCH_STRIDE = 224          # 112/224 都可；112 更密更准但更慢
MAX_DENSE_PATCHES = 2500          # 防止超大图 patch 太多（会截断）
DENSE_PATCH_EDGE_KEEP = True      # 是否补齐右/下边界的 patch

# --- cluster 投票 mask + GrabCut（实例切割）
VOTE_THR_REL = 0.35
VOTE_MEDIAN_K = 5
VOTE_CLOSE_K = 5
GRABCUT_ITERS = 4

# debug/输出
MAKE_DEBUG = True
SAVE_MASKS = True
SAVE_BBOX_CROPS = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# IO（支持中文路径）
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_image(p):
    return os.path.splitext(p)[1].lower() in IMG_EXTS

def list_images(path):
    if os.path.isfile(path):
        return [path] if is_image(path) else []
    imgs = []
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if is_image(p):
                imgs.append(p)
        if not RECURSIVE:
            break
    return sorted(imgs)

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path, img, quality=95):
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"imencode failed: {path}")
    buf.tofile(path)

def resize_max(img, max_long):
    if max_long <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long:
        return img, 1.0
    s = max_long / long_side
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out, s

# =========================
# 周期纹理/格子检测（只做分流，不过滤入库）
# =========================
def is_strong_periodic_texture(img_bgr,
                               thr=PERIODIC_SHARPNESS_THR,
                               min_edge_pixels=PERIODIC_MIN_EDGE_PIXELS):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # 0~360

    m = mag > np.percentile(mag, 70)
    if np.count_nonzero(m) < min_edge_pixels:
        return False

    ang_sel = np.mod(ang[m], 180.0)
    hist, _ = np.histogram(ang_sel, bins=36, range=(0, 180))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)

    peak1 = float(hist.max())
    peak2 = float(np.partition(hist, -2)[-2])
    sharpness = peak1 + peak2
    return sharpness > thr

# =========================
# PATCH: Instance feasibility gate
# =========================
def should_skip_instance_cut(fg_mask_u8, patches, img_h, img_w):
    """
    返回 (skip: bool, reason: str, stats: dict)
    用于在实例切割前判断：这张图是否“适合做实例”
    """
    fg_ratio = float(np.count_nonzero(fg_mask_u8)) / float(fg_mask_u8.size + 1e-6)
    n_patches = len(patches)

    # ROI 的紧致程度：看前景 bbox 占整图比例
    ys, xs = np.where(fg_mask_u8 > 0)
    if len(xs) > 0:
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        roi_bbox_area = float((x2 - x1 + 1) * (y2 - y1 + 1))
        roi_bbox_ratio = roi_bbox_area / float(img_h * img_w + 1e-6)
    else:
        roi_bbox_ratio = 0.0

    stats = {
        "fg_ratio": fg_ratio,
        "roi_bbox_ratio": roi_bbox_ratio,
        "n_patches": n_patches,
    }

    # ---- 可调阈值（你后面根据数据再调）
    FG_RATIO_MAX = 0.35        # 前景占比过大，通常是“背景纹理被当成前景”
    ROI_BBOX_RATIO_MAX = 0.70  # 前景 bbox 覆盖太广，通常不是一个“局部花型”
    PATCHES_MAX = 800          # patch 太多，DBSCAN 很容易被背景簇主导（也很慢）
    PATCHES_MIN = 12           # patch 太少，没必要做实例切割

    if fg_ratio > FG_RATIO_MAX:
        return True, "skip: fg_ratio_too_large", stats
    if roi_bbox_ratio > ROI_BBOX_RATIO_MAX:
        return True, "skip: roi_bbox_too_large", stats
    if n_patches > PATCHES_MAX:
        return True, "skip: too_many_patches", stats
    if n_patches < PATCHES_MIN:
        return True, "skip: too_few_patches", stats

    return False, "ok", stats

# =========================
# PATCH: Anti-background cluster selection
# =========================
def clusters_to_bboxes_antibg(patches_xywh0, labels, img_shape_hw,
                             topk=TOPK_CLUSTERS,
                             ratio_max=0.30,
                             bbox_ratio_max=0.60):
    """
    反背景：过滤掉“占patch比例太大”的簇（背景簇）
    同时过滤 bbox 覆盖太大的簇（很可能是背景区域）
    """
    H, W = img_shape_hw
    n_total = len(patches_xywh0)
    uniq = [l for l in sorted(set(labels)) if l != -1]
    if not uniq:
        return []

    cand = []
    for l in uniq:
        idxs = np.where(labels == l)[0]
        n = int(len(idxs))
        if n <= 0:
            continue

        # 过滤：占比过大 -> 背景簇嫌疑很大
        ratio = n / float(n_total + 1e-6)
        if ratio > ratio_max:
            continue

        # 计算 bbox
        xs, ys, x2s, y2s = [], [], [], []
        for i in idxs:
            x, y, w, h = patches_xywh0[i]
            xs.append(x); ys.append(y)
            x2s.append(x + w); y2s.append(y + h)
        x1 = int(min(xs)); y1 = int(min(ys))
        x2 = int(max(x2s)); y2 = int(max(y2s))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        bbox_area = float(bw * bh)
        bbox_ratio = bbox_area / float(H * W + 1e-6)

        # 过滤：bbox 覆盖太大 -> 还是背景嫌疑
        if bbox_ratio > bbox_ratio_max:
            continue

        # 给一个“更像实例”的分数：
        # - patch 数越多越好（但已经过滤掉过大ratio）
        # - bbox 越紧致越好（bbox_ratio越小越好）
        score = (n ** 0.8) / (bbox_ratio + 1e-3)

        cand.append({
            "cluster": int(l),
            "n_patches": n,
            "ratio": float(ratio),
            "bbox_ratio": float(bbox_ratio),
            "score": float(score),
            "xywh_raw": [x1, y1, bw, bh],
        })

    if not cand:
        return []

    cand.sort(key=lambda d: d["score"], reverse=True)
    cand = cand[:topk]

    # bbox expand + clamp
    out = []
    for d in cand:
        x1, y1, bw, bh = d["xywh_raw"]
        pad_x = int(bw * BBOX_EXPAND)
        pad_y = int(bh * BBOX_EXPAND)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x1 + bw + 2 * pad_x)
        y2 = min(H, y1 + bh + 2 * pad_y)
        out.append({
            "cluster": int(d["cluster"]),
            "n_patches": int(d["n_patches"]),
            "xywh": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "ratio": float(d["ratio"]),
            "bbox_ratio": float(d["bbox_ratio"]),
            "score": float(d["score"]),
        })
    return out


# =========================
# 1) Foreground mask（ROI）
# =========================
def segment_foreground(img_bgr):
    H, W = img_bgr.shape[:2]
    fg = np.zeros((H, W), dtype=np.uint8)

    # A) color branch
    if ENABLE_COLOR_BRANCH:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1].astype(np.float32)
        mask_sat = (S > SAT_THR).astype(np.uint8) * 255

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        A = lab[:, :, 1].astype(np.float32)
        B = lab[:, :, 2].astype(np.float32)
        med_a, med_b = np.median(A), np.median(B)
        ab_dist = np.sqrt((A - med_a) ** 2 + (B - med_b) ** 2)
        mask_ab = (ab_dist > AB_THR).astype(np.uint8) * 255

        color_mask = cv2.bitwise_or(mask_sat, mask_ab)
        fg = cv2.bitwise_or(fg, color_mask)

    # B) residual branch
    if ENABLE_RESID_BRANCH:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=RESID_SIGMA, sigmaY=RESID_SIGMA)
        resid = cv2.absdiff(gray, bg)
        resid_u8 = cv2.normalize(resid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thr = np.percentile(resid_u8, RESID_PERCENTILE)
        resid_mask = (resid_u8 >= thr).astype(np.uint8) * 255
        fg = cv2.bitwise_or(fg, resid_mask)

    # C) 去小连通域
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    min_area = int(MIN_COMPONENT_AREA_RATIO * H * W)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        clean[lab == i] = 255
    return clean

# =========================
# 2) ROI patches（实例切割用）
# =========================
def sample_patches(img_bgr, fg_mask):
    H, W = img_bgr.shape[:2]
    ps = PATCH_SIZE
    st = STRIDE

    xs = list(range(0, max(1, W - ps + 1), st))
    ys = list(range(0, max(1, H - ps + 1), st))
    if xs and xs[-1] != W - ps:
        xs.append(W - ps)
    if ys and ys[-1] != H - ps:
        ys.append(H - ps)

    patches = []
    for y in ys:
        for x in xs:
            m = fg_mask[y:y+ps, x:x+ps]
            cover = float(np.count_nonzero(m)) / (ps * ps)
            if cover < MIN_MASK_COVER:
                continue
            patch = img_bgr[y:y+ps, x:x+ps]
            patches.append((x, y, ps, ps, cover, patch))

    if len(patches) > MAX_PATCHES_PER_IMAGE:
        patches.sort(key=lambda t: t[4], reverse=True)
        patches = patches[:MAX_PATCHES_PER_IMAGE]
    return patches

# =========================
# 2.5) Dense patches（检索用，所有图）
# =========================
def sample_dense_patches(img_bgr, stride=DENSE_PATCH_STRIDE, max_patches=MAX_DENSE_PATCHES):
    H, W = img_bgr.shape[:2]
    ps = PATCH_SIZE
    st = stride

    xs = list(range(0, max(1, W - ps + 1), st))
    ys = list(range(0, max(1, H - ps + 1), st))
    if DENSE_PATCH_EDGE_KEEP:
        if xs and xs[-1] != W - ps:
            xs.append(W - ps)
        if ys and ys[-1] != H - ps:
            ys.append(H - ps)

    coords = []
    patches = []
    for y in ys:
        for x in xs:
            patch = img_bgr[y:y+ps, x:x+ps]
            coords.append((x, y))
            patches.append((x, y, ps, ps, 1.0, patch))

    if len(patches) > max_patches:
        # 均匀抽样：避免只保留前面区域
        idx = np.linspace(0, len(patches)-1, max_patches).astype(int)
        patches = [patches[i] for i in idx]
        coords = [coords[i] for i in idx]

    return patches, np.array(coords, dtype=np.int32)

# =========================
# 3) DINO features (timm)
# =========================
def get_dino_model():
    import timm
    model = timm.create_model(DINO_MODEL_NAME, pretrained=True)
    model.eval()
    model.to(DEVICE)
    return model

def preprocess_patch_cv2(patch_bgr):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    img = patch_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))

def extract_features_dino(model, patches):
    import torch
    xs = [preprocess_patch_cv2(p[5]) for p in patches]
    x = torch.from_numpy(np.stack(xs, axis=0)).float().to(DEVICE)

    feats = []
    with torch.no_grad():
        for i in range(0, x.shape[0], BATCH_SIZE):
            xb = x[i:i+BATCH_SIZE]
            if hasattr(model, "forward_features"):
                fb = model.forward_features(xb)
                if isinstance(fb, (list, tuple)):
                    fb = fb[0]
                if fb.ndim == 3:
                    fb = fb[:, 0, :]
            else:
                fb = model(xb)
            fb = torch.nn.functional.normalize(fb, dim=1)
            feats.append(fb.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)

def extract_global_dino(model, img_bgr):
    import torch
    x = torch.from_numpy(preprocess_patch_cv2(cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA))[None]).float().to(DEVICE)
    with torch.no_grad():
        if hasattr(model, "forward_features"):
            fb = model.forward_features(x)
            if isinstance(fb, (list, tuple)):
                fb = fb[0]
            if fb.ndim == 3:
                fb = fb[:, 0, :]
        else:
            fb = model(x)
        fb = torch.nn.functional.normalize(fb, dim=1)
    return fb[0].detach().cpu().numpy()

# =========================
# 拼坐标特征（实例切割聚类用）
# =========================
def build_features_with_xy(feats, patches, img_w, img_h, alpha=XY_ALPHA):
    ps = PATCH_SIZE
    centers = []
    for (x, y, w, h, cover, _) in patches:
        cx = (x + ps * 0.5) / float(img_w)
        cy = (y + ps * 0.5) / float(img_h)
        centers.append([cx, cy])
    centers = np.array(centers, dtype=np.float32)
    return np.concatenate([feats, alpha * centers], axis=1)

# =========================
# 聚类（实例切割）
# =========================
def cluster_dbscan(feat2):
    from sklearn.cluster import DBSCAN
    cl = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="euclidean")
    return cl.fit_predict(feat2)

# =========================
# cluster -> bbox（原图坐标）
# =========================
def clusters_to_bboxes(patches_xywh, labels, img_shape_hw):
    H, W = img_shape_hw
    uniq = [l for l in sorted(set(labels)) if l != -1]
    if not uniq:
        return []

    clusters = []
    for l in uniq:
        idxs = np.where(labels == l)[0]
        if len(idxs) == 0:
            continue
        xs, ys, x2s, y2s = [], [], [], []
        for i in idxs:
            x, y, w, h = patches_xywh[i]
            xs.append(x); ys.append(y)
            x2s.append(x + w); y2s.append(y + h)
        x1 = int(min(xs)); y1 = int(min(ys))
        x2 = int(max(x2s)); y2 = int(max(y2s))
        area = (x2 - x1) * (y2 - y1)
        clusters.append((area, l, x1, y1, x2, y2, int(len(idxs))))

    clusters.sort(reverse=True)
    clusters = clusters[:TOPK_CLUSTERS]

    bboxes = []
    for area, l, x1, y1, x2, y2, npts in clusters:
        bw = x2 - x1
        bh = y2 - y1
        pad_x = int(bw * BBOX_EXPAND)
        pad_y = int(bh * BBOX_EXPAND)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)
        bboxes.append({
            "cluster": int(l),
            "n_patches": int(npts),
            "xywh": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        })
    return bboxes

# =========================
# 投票 mask / GrabCut / 透明 PNG
# =========================
def cluster_vote_mask(patches_xywh0, labels, img_h, img_w, cluster_id):
    vote = np.zeros((img_h, img_w), dtype=np.uint16)
    idxs = np.where(labels == cluster_id)[0]
    for i in idxs:
        x0, y0, w0, h0 = patches_xywh0[i]
        vote[y0:y0+h0, x0:x0+w0] += 1

    if vote.max() == 0:
        return None, vote

    thr = max(2, int(vote.max() * VOTE_THR_REL))
    mask = (vote >= thr).astype(np.uint8) * 255

    if VOTE_MEDIAN_K >= 3:
        mask = cv2.medianBlur(mask, VOTE_MEDIAN_K)
    if VOTE_CLOSE_K >= 3:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((VOTE_CLOSE_K, VOTE_CLOSE_K), np.uint8),
                                iterations=1)
    return mask, vote

def refine_with_grabcut(img_bgr, init_mask, bbox_xywh, iters=GRABCUT_ITERS):
    x, y, w, h = bbox_xywh
    H, W = img_bgr.shape[:2]
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W-x))
    h = max(1, min(h, H-y))

    gc_mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    gc_mask[y:y+h, x:x+w] = cv2.GC_PR_BGD
    gc_mask[init_mask == 255] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)

    cv2.grabCut(img_bgr, gc_mask, rect, bgdModel, fgdModel, iters, mode=cv2.GC_INIT_WITH_MASK)
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return out

def cutout_rgba(img_bgr, mask_u8, bbox_xywh):
    x, y, w, h = bbox_xywh
    crop = img_bgr[y:y+h, x:x+w].copy()
    m = mask_u8[y:y+h, x:x+w]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = m
    return rgba

# =========================
# Main per image
# =========================
def process_one(img_path, model):
    img0 = imread_unicode(img_path)
    if img0 is None:
        return

    name = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(out_dir)

    # 1) scene type（只做标注/分流）
    scene_type = "normal"
    if ENABLE_PERIODIC_ROUTING and is_strong_periodic_texture(img0):
        scene_type = "periodic_texture"  # 格子/条纹/强周期纹理

    # 2) 保存检索特征（所有图都做）
    meta = {
        "image": os.path.basename(img_path),
        "scene_type": scene_type,
        "instances": []
    }

    if SAVE_GLOBAL_FEAT:
        gfeat = extract_global_dino(model, img0)
        np.save(os.path.join(out_dir, "global_feat.npy"), gfeat)
        meta["global_feat"] = "global_feat.npy"

    # 为局部拍摄检索准备：dense patch embedding（所有图都做）
    if SAVE_DENSE_PATCH_FEATS:
        img_r, scale_r = resize_max(img0, MAX_LONG_SIDE)
        dense_patches, coords = sample_dense_patches(img_r, stride=DENSE_PATCH_STRIDE, max_patches=MAX_DENSE_PATCHES)
        dfeats = extract_features_dino(model, dense_patches)  # [N,C]
        # coords 是 resized 图坐标；保存 scale 便于回到原图
        np.savez_compressed(
            os.path.join(out_dir, "patch_feats.npz"),
            feats=dfeats.astype(np.float16),
            coords=coords.astype(np.int32),
            scale=np.array([scale_r], dtype=np.float32),
            patch_size=np.array([PATCH_SIZE], dtype=np.int32),
            stride=np.array([DENSE_PATCH_STRIDE], dtype=np.int32),
        )
        meta["patch_feats"] = "patch_feats.npz"

    # 3) 实例切割：对强周期纹理默认不做（但仍可检索）
    #    你后面如果要做“重复单元切割”，就在这里加 periodic 分支。
    if scene_type == "periodic_texture":
        if MAKE_DEBUG:
            dbg = img0.copy()
            cv2.putText(dbg, "scene_type: periodic_texture (no instance cut)", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            imwrite_unicode(os.path.join(out_dir, "debug.jpg"), dbg)
        with open(os.path.join(out_dir, "instances.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return

    # 4) 非周期：走花型实例切割链路
    img, scale = resize_max(img0, MAX_LONG_SIDE)
    Hs, Ws = img.shape[:2]
    H0, W0 = img0.shape[:2]

    fg = segment_foreground(img)
    if SAVE_MASKS:
        ensure_dir(os.path.join(out_dir, "_masks"))
        imwrite_unicode(os.path.join(out_dir, "_masks", "fg.jpg"), fg)

    patches = sample_patches(img, fg)
    if len(patches) == 0:
        with open(os.path.join(out_dir, "instances.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return
    # ---- PATCH: instance feasibility gate
    skip, reason, gate_stats = should_skip_instance_cut(
        fg_mask_u8=fg, patches=patches, img_h=Hs, img_w=Ws
    )
    meta["instance_gate"] = {"skip": bool(skip), "reason": reason, **gate_stats}

    if skip:
        if MAKE_DEBUG:
            dbg = img0.copy()
            cv2.putText(dbg, f"SKIP_INSTANCE: {reason}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            imwrite_unicode(os.path.join(out_dir, "debug.jpg"), dbg)
        with open(os.path.join(out_dir, "instances.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return



    # patch 坐标映射回原图
    patches_xywh0 = []
    for (x, y, w, h, cover, patch) in patches:
        if scale != 1.0:
            x0 = int(x / scale); y0 = int(y / scale)
            w0 = int(w / scale); h0 = int(h / scale)
        else:
            x0, y0, w0, h0 = x, y, w, h

        x0 = max(0, min(x0, W0 - 1))
        y0 = max(0, min(y0, H0 - 1))
        w0 = max(1, min(w0, W0 - x0))
        h0 = max(1, min(h0, H0 - y0))
        patches_xywh0.append((x0, y0, w0, h0))

    feats = extract_features_dino(model, patches)
    feat2 = build_features_with_xy(feats, patches, img_w=Ws, img_h=Hs, alpha=XY_ALPHA)
    labels = cluster_dbscan(feat2)

    bboxes = clusters_to_bboxes_antibg(
        patches_xywh0, labels, (H0, W0),
        topk=TOPK_CLUSTERS,
        ratio_max=0.30,  # 背景簇过滤强度：越小越严格
        bbox_ratio_max=0.60  # bbox 太大的簇过滤
    )

    inst_dir = os.path.join(out_dir, "instances")
    ensure_dir(inst_dir)
    ensure_dir(os.path.join(out_dir, "_masks"))

    dbg = img0.copy()
    out_instances = []

    for idx, inst in enumerate(bboxes):
        cid = inst["cluster"]
        x, y, w, h = inst["xywh"]

        vote_mask, vote = cluster_vote_mask(patches_xywh0, labels, H0, W0, cid)
        if vote_mask is None:
            continue

        refined = refine_with_grabcut(img0, vote_mask, (x, y, w, h), iters=GRABCUT_ITERS)
        rgba = cutout_rgba(img0, refined, (x, y, w, h))

        png_path = os.path.join(inst_dir, f"{idx:02d}_cluster{cid}.png")
        imwrite_unicode(png_path, rgba)

        if SAVE_BBOX_CROPS:
            crop = img0[y:y+h, x:x+w]
            imwrite_unicode(os.path.join(inst_dir, f"{idx:02d}_cluster{cid}.jpg"), crop)

        if SAVE_MASKS:
            imwrite_unicode(os.path.join(out_dir, "_masks", f"vote_{idx:02d}_c{cid}.jpg"), vote_mask)
            imwrite_unicode(os.path.join(out_dir, "_masks", f"refined_{idx:02d}_c{cid}.png"), refined)

        # debug
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(dbg, f"{idx}", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, cnts, -1, (0, 255, 0), 2)

        out_instances.append({
            "id": idx,
            "cluster": int(cid),
            "n_patches": int(inst["n_patches"]),
            "xywh": [int(x), int(y), int(w), int(h)],
            "area": int(np.count_nonzero(refined)),
            "png": os.path.relpath(png_path, out_dir).replace("\\", "/"),
            "ratio": float(inst.get("ratio", -1)),
            "bbox_ratio": float(inst.get("bbox_ratio", -1)),
            "score": float(inst.get("score", -1)),
        })

    meta["instances"] = out_instances
    meta["scale_instance"] = float(scale)

    if MAKE_DEBUG:
        cv2.putText(dbg, f"scene_type: {scene_type}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        imwrite_unicode(os.path.join(out_dir, "debug.jpg"), dbg)

    with open(os.path.join(out_dir, "instances.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# =========================
# RUN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    imgs = list_images(INPUT_PATH)
    if LIMIT_IMAGES > 0:
        imgs = imgs[:LIMIT_IMAGES]
    print(f"Found {len(imgs)} images")

    model = get_dino_model()

    for i, p in enumerate(imgs, 1):
        print(f"[{i}/{len(imgs)}] {os.path.basename(p)}")
        try:
            process_one(p, model)
        except Exception as e:
            print("ERROR:", p, e)

if __name__ == "__main__":
    main()
