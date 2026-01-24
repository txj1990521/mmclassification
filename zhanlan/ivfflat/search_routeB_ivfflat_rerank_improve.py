#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# =========================
# CONFIG: 只改这里
# =========================
CONFIG = r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan.py"
CKPT   = r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth"

QUERY_IMG = r"D:\zhanlan\qurrey_data\003b.jpg"

OUT_INDEX = r"D:\zhanlan\faiss_database\faiss_ivf.index"
OUT_META  = r"D:\zhanlan\faiss_database\faiss_paths.npy"
OUT_VIZ   = r"D:\zhanlan\topk_viz_rmac\topk_viz_rmac3.jpg"

# =========================
# Runtime
# =========================
TOPK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE = 320

# ---------- Coarse (Route-B aligned) ----------
COARSE_K = 800
# IVF 搜索的关键：nprobe 越大越准，越慢；10万常用 8~16；百万常用 16~64
NPROBE = 64

VIEWS_PER_QUERY = 24
RESIZE_SHORT = 256
CROP_SIZE = 224

VIEW_PLAN = [
    (0,   1, 5),
    (-15, 1, 1),
    (15,  1, 1),
    (-30, 1, 0),
    (30,  1, 0),
]
QUERY_RANDOM_SEED = 0  # 让同一 query 稳定（你喜欢稳定就保留）

# ---------- Rerank ----------
FEAT_LEVEL = -2
RMAC_INPUT_SHORT = 512
KEEP_PATCHES = 256
BORDER = 0.05

MARGIN = 0.015      # 0.01~0.02
MIN_KEEP = 6        # 6~10
BIN_SIZE = 4.0
TOPK_CORE = 64


TOPM = 6


# ---------- Fusion ----------
# 这里用你稳定版 gating fuse
GATE_T = 0.70
GAMMA = 0.85

# 周期纹理判别
PERIODIC_PEAK_THR = 0.28
PERIODIC_COVER_THR = 0.50

# coarse 保底
COARSE_TOPN = 60
MIN_KEEP_IN_TOPK = 6
# ---------- Texture gate ----------
TEX_ENABLE = True
TEX_SIZE = 256          # 计算纹理时缩放到 256 边长
TEX_CROP_FRAC = 0.75    # 只看中心区域，避免背景/边框干扰
TEX_THR_LOW = 0.85      # 低于这个认为纹理不像（强惩罚）
TEX_THR_HIGH = 0.92     # 高于这个认为纹理像（轻微加分）
TEX_PENALTY = 0.55   # 乘 0.35 才能明显压下去
TEX_BONUS   = 0.03
# =========================
# utils
# =========================
def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(out_path: str, img_bgr: np.ndarray):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ext = Path(out_path).suffix.lower()
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(out_path)

def _put_text(img, text, org=(8, 26), font_scale=0.8, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _fit_square(img_bgr, tile=320):
    h, w = img_bgr.shape[:2]
    scale = tile / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    y0 = (tile - nh) // 2
    x0 = (tile - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def make_topk_viz(query_path: str, top_paths: list, top_scores: list, out_path: str, tile: int = 320):
    qimg = imread_unicode(query_path)
    imgs = [_fit_square(qimg, tile)]
    labels = ["QUERY"]

    for i, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        img = imread_unicode(str(p))
        imgs.append(_fit_square(img, tile))
        labels.append(f"#{i}  {float(s):.4f}")

    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    gap = 8
    header = 40

    H = rows * (tile + header) + (rows + 1) * gap
    W = cols * tile + (cols + 1) * gap
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in range(n):
        r = idx // cols
        c = idx % cols
        x = gap + c * (tile + gap)
        y = gap + r * (tile + header + gap)
        canvas[y + header:y + header + tile, x:x + tile] = imgs[idx]
        _put_text(canvas, labels[idx], org=(x + 8, y + 26), font_scale=0.7, thickness=2)
    imwrite_unicode(out_path, canvas)
    return out_path

# =========================
# preprocess
# =========================
def grad_concentration_bgr(img_bgr, top_frac=0.05):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx*gx + gy*gy).reshape(-1)
    g.sort()
    k = int((1.0 - top_frac) * len(g))
    return float(g[k:].sum() / (g.sum() + 1e-6))
def find_roi_by_grad_energy(img_bgr, blur_ksize=31, thr_percentile=95, pad_ratio=0.20):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)

    grad_blur = cv2.GaussianBlur(grad, (blur_ksize, blur_ksize), 0)
    thr = np.percentile(grad_blur, thr_percentile)


    mask = (grad_blur > thr).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None, None  # 没找到

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    x, y, w, h, _ = stats[idx]

    pad = int(pad_ratio * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    roi = img_bgr[y1:y2, x1:x2].copy()
    bbox = (x1, y1, x2, y2)  # in original image coords
    return bbox, roi

def roi_is_valid(bbox, img_shape, min_area_frac=0.02, max_area_frac=0.40):
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    H, W = img_shape[:2]
    a = max(1, (x2 - x1)) * max(1, (y2 - y1))
    frac = a / float(H * W + 1e-9)
    return (frac >= min_area_frac) and (frac <= max_area_frac)

def _fft_texture_desc(img_bgr: np.ndarray, long_edge=256, crop_frac=0.75,
                      n_rings=16, n_angles=16, dc_frac=0.08) -> np.ndarray:
    """
    FFT texture descriptor: radial + angular energy on log-magnitude spectrum.
    Return L2-normalized 1D vector (n_rings + n_angles).
    """
    x = _center_crop(img_bgr, crop_frac)
    x = _resize_long_edge(x, long_edge)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Hann window reduce edge effect
    h, w = gray.shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    gray = gray * wy[:, None] * wx[None, :]

    # FFT
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F)).astype(np.float32)

    # remove DC (center low-freq square)
    cy, cx = h // 2, w // 2
    r0y = int(h * dc_frac)
    r0x = int(w * dc_frac)
    mag[cy - r0y:cy + r0y + 1, cx - r0x:cx + r0x + 1] = 0.0

    # polar bins
    yy, xx = np.mgrid[0:h, 0:w]
    yy = yy.astype(np.float32) - cy
    xx = xx.astype(np.float32) - cx
    rr = np.sqrt(xx * xx + yy * yy)
    rr /= (rr.max() + 1e-6)

    ang = np.arctan2(yy, xx)  # [-pi, pi]
    ang = (ang + np.pi) / (2 * np.pi)  # [0,1)

    # ring energy
    ring_hist = np.zeros((n_rings,), np.float32)
    for i in range(n_rings):
        r1 = i / n_rings
        r2 = (i + 1) / n_rings
        m = (rr >= r1) & (rr < r2)
        if np.any(m):
            ring_hist[i] = float(mag[m].mean())

    # angle energy (ignore very center already)
    ang_hist = np.zeros((n_angles,), np.float32)
    for i in range(n_angles):
        a1 = i / n_angles
        a2 = (i + 1) / n_angles
        m = (ang >= a1) & (ang < a2)
        if np.any(m):
            ang_hist[i] = float(mag[m].mean())

    feat = np.concatenate([ring_hist, ang_hist], axis=0).astype(np.float32)

    # normalize
    feat -= feat.mean()
    feat /= (feat.std() + 1e-6)
    feat /= (np.linalg.norm(feat) + 1e-9)
    return feat


def fft_tex_sim(q_img_bgr, c_img_bgr, long_edge=256, crop_frac=0.75) -> float:
    q = _fft_texture_desc(q_img_bgr, long_edge=long_edge, crop_frac=crop_frac)
    c = _fft_texture_desc(c_img_bgr, long_edge=long_edge, crop_frac=crop_frac)
    return float(np.clip(np.dot(q, c), 0.0, 1.0))

def _center_crop(img, frac=0.75):
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y1 = (h - ch) // 2
    x1 = (w - cw) // 2
    return img[y1:y1+ch, x1:x1+cw]

def _resize_long_edge(img, long_edge=256):
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    nh, nw = max(1, int(round(h*s))), max(1, int(round(w*s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def lbp_hist_gray(gray: np.ndarray) -> np.ndarray:
    """
    8-neighbor LBP, radius=1, output 256-bin hist (L2 normalized).
    gray: uint8
    """
    g = gray.astype(np.uint8)
    c = g

    # 8 neighbors via roll (wrap), then we will ignore border by later cropping
    n0 = np.roll(g, -1, axis=0)      # up
    n1 = np.roll(g, (-1, 1), axis=(0, 1))  # up-right
    n2 = np.roll(g, 1, axis=1)       # right
    n3 = np.roll(g, (1, 1), axis=(0, 1))   # down-right
    n4 = np.roll(g, 1, axis=0)       # down
    n5 = np.roll(g, (1, -1), axis=(0, 1))  # down-left
    n6 = np.roll(g, -1, axis=1)      # left
    n7 = np.roll(g, (-1, -1), axis=(0, 1)) # up-left

    code = ((n0 >= c).astype(np.uint8) << 0) | \
           ((n1 >= c).astype(np.uint8) << 1) | \
           ((n2 >= c).astype(np.uint8) << 2) | \
           ((n3 >= c).astype(np.uint8) << 3) | \
           ((n4 >= c).astype(np.uint8) << 4) | \
           ((n5 >= c).astype(np.uint8) << 5) | \
           ((n6 >= c).astype(np.uint8) << 6) | \
           ((n7 >= c).astype(np.uint8) << 7)

    # 去掉边缘一圈（roll 会环绕，边缘不可信）
    code = code[1:-1, 1:-1]

    hist = np.bincount(code.reshape(-1), minlength=256).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    # L2 normalize for cosine
    hist /= (np.linalg.norm(hist) + 1e-9)
    return hist

def texture_desc_lbp(img_bgr: np.ndarray, long_edge=256, crop_frac=0.75) -> np.ndarray:
    x = _center_crop(img_bgr, crop_frac)
    x = _resize_long_edge(x, long_edge)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return lbp_hist_gray(gray)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def resize_short_edge(img_rgb: np.ndarray, short=256):
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

def rotate_bound(img_rgb: np.ndarray, deg: float):
    if deg == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img_rgb, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def crop_center(img_rgb: np.ndarray, size=224):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img_rgb[y1:y1+size, x1:x1+size]

def random_crop(img_rgb: np.ndarray, size=224):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img_rgb[y:y+size, x:x+size]

def pad_to_square(img_rgb: np.ndarray):
    h, w = img_rgb.shape[:2]
    if h == w:
        return img_rgb
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_REFLECT101)

def to_tensor_from_rgb_crop(img_rgb_crop: np.ndarray, mean, std):
    x = img_rgb_crop.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))

@torch.no_grad()
def aggregate_views_to_one(feats_view: torch.Tensor):
    agg = feats_view.max(dim=0).values
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

@torch.no_grad()
def make_query_views_routeB(img_bgr: np.ndarray, mean, std, to_rgb: bool):
    random.seed(QUERY_RANDOM_SEED)

    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = resize_short_edge(img_rgb, RESIZE_SHORT)

    views = []
    for deg, n_center, n_rand in VIEW_PLAN:
        rot = rotate_bound(img_rgb, deg)
        for _ in range(n_center):
            views.append(to_tensor_from_rgb_crop(crop_center(rot, CROP_SIZE), mean, std))
        for _ in range(n_rand):
            views.append(to_tensor_from_rgb_crop(random_crop(rot, CROP_SIZE), mean, std))

    return views[:VIEWS_PER_QUERY]

def random_resized_crop(img_rgb, out_size=224, scale=(0.25, 1.0), ratio=(0.75, 1.33)):
    h, w = img_rgb.shape[:2]
    area = h * w
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(*ratio)
        nh = int(round(np.sqrt(target_area / aspect)))
        nw = int(round(np.sqrt(target_area * aspect)))
        if nh <= h and nw <= w:
            y = random.randint(0, h - nh)
            x = random.randint(0, w - nw)
            crop = img_rgb[y:y+nh, x:x+nw]
            return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    # fallback center
    return crop_center(img_rgb, out_size)

@torch.no_grad()
def make_views_from_img(img_bgr, mean, std, to_rgb: bool, resize_short=256, crop_size=224,
                        n_rrc=12, rrc_scale=(0.10, 1.0)):
    random.seed(QUERY_RANDOM_SEED)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if to_rgb else img_bgr[:, :, ::-1].copy()
    img_rgb = resize_short_edge(img_rgb, resize_short)

    views = []
    for deg, n_center, n_rand in VIEW_PLAN:
        rot = rotate_bound(img_rgb, deg)
        for _ in range(n_center):
            views.append(to_tensor_from_rgb_crop(crop_center(rot, crop_size), mean, std))
        for _ in range(n_rand):
            views.append(to_tensor_from_rgb_crop(random_crop(rot, crop_size), mean, std))

    for _ in range(n_rrc):
        rrc = random_resized_crop(img_rgb, out_size=crop_size, scale=rrc_scale)
        views.append(to_tensor_from_rgb_crop(rrc, mean, std))

    return views
@torch.no_grad()
def make_query_views_more_robust(img_bgr, mean, std, to_rgb: bool, roi_bgr=None, roi_weight=0.6):
    # 1) 全图 views
    v_full = make_views_from_img(
        img_bgr, mean, std, to_rgb,
        resize_short=RESIZE_SHORT, crop_size=CROP_SIZE,
        n_rrc=12, rrc_scale=(0.10, 1.0)
    )

    if roi_bgr is None:
        return v_full[:VIEWS_PER_QUERY]

    # 2) ROI views：更偏“局部”，scale 下限更小更聚焦
    v_roi = make_views_from_img(
        roi_bgr, mean, std, to_rgb,
        resize_short=RESIZE_SHORT, crop_size=CROP_SIZE,
        n_rrc=16, rrc_scale=(0.40, 1.0)
    )

    # 3) 按比例拼起来（固定总数）
    n_roi = int(round(VIEWS_PER_QUERY * roi_weight))
    n_full = VIEWS_PER_QUERY - n_roi

    # 为了稳定：各自截取前面（你也可以 shuffle）
    out = v_roi[:n_roi] + v_full[:n_full]
    return out[:VIEWS_PER_QUERY]


@torch.no_grad()
def make_single_tensor_for_rerank(img_bgr: np.ndarray, mean, std, to_rgb: bool):
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = pad_to_square(img_rgb)
    img_rgb = cv2.resize(img_rgb, (RMAC_INPUT_SHORT, RMAC_INPUT_SHORT), interpolation=cv2.INTER_LINEAR)

    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)

def map_bbox_resize_to_feat(bbox512, in_size, Hf, Wf):
    x1,y1,x2,y2 = bbox512
    sx = Wf / float(in_size)
    sy = Hf / float(in_size)
    fx1 = int(np.floor(x1 * sx)); fx2 = int(np.ceil(x2 * sx))
    fy1 = int(np.floor(y1 * sy)); fy2 = int(np.ceil(y2 * sy))
    fx1 = max(0, min(Wf-1, fx1)); fx2 = max(1, min(Wf, fx2))
    fy1 = max(0, min(Hf-1, fy1)); fy2 = max(1, min(Hf, fy2))
    return (fx1, fy1, fx2, fy2)

def map_bbox_to_square_resized(bbox, H, W, out_size=512):
    """
    bbox: (x1,y1,x2,y2) on original
    return bbox on padded-square then resized to out_size
    """
    x1,y1,x2,y2 = bbox
    S = max(H, W)

    # pad offsets (reflect pad in your pad_to_square)
    pad_x = (S - W) // 2
    pad_y = (S - H) // 2

    # bbox in padded square coords
    x1s = x1 + pad_x; x2s = x2 + pad_x
    y1s = y1 + pad_y; y2s = y2 + pad_y

    # scale to out_size
    scale = out_size / float(S)
    x1r = int(round(x1s * scale)); x2r = int(round(x2s * scale))
    y1r = int(round(y1s * scale)); y2r = int(round(y2s * scale))

    x1r = max(0, min(out_size-1, x1r))
    y1r = max(0, min(out_size-1, y1r))
    x2r = max(1, min(out_size,   x2r))
    y2r = max(1, min(out_size,   y2r))
    return (x1r, y1r, x2r, y2r)

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
def extract_global_feat_for_coarse(model, batch_tensor: torch.Tensor, tau: float = 0.7, alpha: float = 0.5):
    fm = model.backbone(batch_tensor)
    if isinstance(fm, (tuple, list)):
        fm = fm[-1]

    gap = fm.mean(dim=(2,3))

    e = (fm * fm).sum(dim=1)  # (B,H,W)
    w = torch.softmax((e / (e.mean(dim=(1,2), keepdim=True) + 1e-6)) / tau, dim=-1)
    w = w.view(w.shape[0], 1, w.shape[1], w.shape[2])
    att = (fm * w).sum(dim=(2,3))

    feat = alpha * att + (1 - alpha) * gap
    feat = F.normalize(feat, p=2, dim=1)
    return feat



@torch.no_grad()
def extract_featmap(model, batch_tensor: torch.Tensor, prefer_level: int):
    out = model.backbone(batch_tensor)

    if isinstance(out, dict):
        if 'feat' in out:
            out = out['feat']
        elif 'features' in out:
            out = out['features']
        else:
            out = list(out.values())[-1]

    if isinstance(out, (tuple, list)):
        n = len(out)
        lvl = prefer_level
        if lvl < -n: lvl = -n
        if lvl > n - 1: lvl = n - 1
        feat = out[lvl]

        if not hasattr(extract_featmap, "_printed"):
            print(f"[DEBUG] backbone returns {type(out).__name__} of len={n}")
            for i, t in enumerate(out):
                print(f"  [{i}] shape={tuple(t.shape)}")
            print(f"[DEBUG] selected level={lvl} (prefer {prefer_level})")
            extract_featmap._printed = True

        return feat

    if isinstance(out, torch.Tensor):
        return out

    raise TypeError(f"Unsupported backbone output type: {type(out)}")

# =========================
# rerank: top patches + xy
# =========================
@torch.no_grad()
def select_top_patches_with_xy(feat_map, keep=512, border=0.15, roi_fbox=None):
    fm = feat_map[0]
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)

    # 默认 border mask
    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))
    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask[y1b:y2b, x1b:x2b] = True

    # 如果提供 ROI：mask 再与 ROI 相交
    if roi_fbox is not None:
        rx1, ry1, rx2, ry2 = roi_fbox
        roi_mask = torch.zeros_like(mask)
        roi_mask[ry1:ry2, rx1:rx2] = True
        mask = mask & roi_mask

    idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        return None, None  # 兜底：外面处理回退

    k = min(keep, idx_all.numel())
    vals = energy.flatten()[idx_all]
    top_local = torch.topk(vals, k=k, largest=True).indices
    idx = idx_all[top_local]

    patches = fm.flatten(1).t()[idx]
    patches = F.normalize(patches, p=2, dim=1)

    ys = (idx // W).float()
    xs = (idx %  W).float()
    xy = torch.stack([xs, ys], dim=1)
    return patches, xy

def energy_roi_box(fm, frac=0.35):
    # fm: (C,H,W)
    e = fm.pow(2).sum(dim=0)  # (H,W)
    flat = e.flatten()
    k = max(1, int(flat.numel() * frac))
    thr = torch.topk(flat, k=k, largest=True).values.min()
    m = (e >= thr)
    ys, xs = torch.where(m)
    if ys.numel() < 10:
        return None
    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1
    return (x1, y1, x2, y2)

@torch.no_grad()
def select_top_patches_with_xy_batch(feat_map, keep=512, border=0.15, roi_fboxes=None, fallback_to_border=True):
    """
    feat_map: (B, C, H, W)
    roi_fboxes: None or list/tuple length B, each is (rx1, ry1, rx2, ry2) in feature-map coords
               coords follow python slicing: [ry1:ry2, rx1:rx2]
    return: list of (patches, xy) for each item, patches shape (K,C), xy shape (K,2) with (x,y)
    """
    B, C, H, W = feat_map.shape
    out = []

    # border mask range
    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))

    for b in range(B):
        fm = feat_map[b]
        energy = fm.pow(2).sum(dim=0)  # (H,W)

        # base border mask
        mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
        mask[y1b:y2b, x1b:x2b] = True

        # roi intersect (if provided)
        if roi_fboxes is not None:
            rb = roi_fboxes[b]
            if rb is not None:
                rx1, ry1, rx2, ry2 = rb
                # clamp
                rx1 = max(0, min(W-1, int(rx1)))
                ry1 = max(0, min(H-1, int(ry1)))
                rx2 = max(1, min(W,   int(rx2)))
                ry2 = max(1, min(H,   int(ry2)))
                if rx2 > rx1 and ry2 > ry1:
                    roi_mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
                    roi_mask[ry1:ry2, rx1:rx2] = True
                    mask2 = mask & roi_mask

                    # 如果 ROI 太小/无效导致空，则回退
                    if fallback_to_border and (mask2.sum().item() == 0):
                        mask2 = mask
                    mask = mask2

        idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
        if idx_all.numel() == 0:
            # 极端情况：连 border 也为空（基本不会），直接全图
            idx_all = torch.arange(H*W, device=energy.device)

        k = min(int(keep), int(idx_all.numel()))
        vals = energy.flatten()[idx_all]
        top_local = torch.topk(vals, k=k, largest=True).indices
        idx = idx_all[top_local]

        patches = fm.flatten(1).t()[idx]          # (k, C)
        patches = F.normalize(patches, p=2, dim=1)

        ys = (idx // W).float()
        xs = (idx %  W).float()
        xy = torch.stack([xs, ys], dim=1)         # (k, 2)

        out.append((patches, xy))

    return out

@torch.no_grad()
def geom_score_adaptive(q_desc, q_xy, c_desc, c_xy,
                        margin=0.02, min_keep=8,
                        bin_size=4.0, topM=6, topk_core=64,
                        periodic_peak_thr=0.22, periodic_cover_thr=0.55):

    sim = q_desc @ c_desc.t()                     # (Nq, Nc)

    # 1) q->c best and 2nd best
    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv  = topv[:, 1]

    # 2) c->q best (for mutual)
    c_best = torch.argmax(sim, dim=0)             # (Nc,)

    # 3) mutual NN mask
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    # 4) ratio / margin test
    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng = int(good.sum().item())
    if Ng < min_keep:
        return 0.0
    # good matches indices
    # 先拿到匹配到的 candidate 索引（必须先定义 mi）
    good_idx = torch.nonzero(good, as_tuple=False).squeeze(1)
    # 按 best 相似度排序，取 topK
    K = min(topk_core, good_idx.numel())
    sel = torch.topk(q_bestv[good], k=K, largest=True).indices
    good_idx = good_idx[sel]

    mi = q_best[good_idx]
    qg = q_xy[good_idx]
    cg = c_xy[mi]
    Ng = int(good_idx.numel())

    # ---- shape consistency: pairwise distance ratio ----
    P = min(64, Ng)
    qg2 = qg[:P]
    cg2 = cg[:P]

    ratios = []
    for i in range(P):
        j = (i * 7 + 13) % P
        dq = torch.norm(qg2[i] - qg2[j]) + 1e-6
        if dq < 3.0:
            continue
        dc = torch.norm(cg2[i] - cg2[j]) + 1e-6
        rr = (dc / dq).clamp(0.25, 4.0)
        ratios.append(torch.log(rr))  # ★用 log ratio
    if len(ratios) < 8:
        return 0.0

    ratio_std = float(torch.stack(ratios).std().item())

    # 不要硬切 0
    # if ratio_std > 0.35:
    #     return 0.0

    # 改成 soft gate：>0.35 开始强压，>0.55 极强压
    # ratio_std 已经算出来以后
    # ratio_std: float

    # 1) gate 一定要先给默认值
    shape_gate = 1.0
    if ratio_std > 1.2:
        shape_gate = 0.40
    elif ratio_std > 0.8:
        shape_gate = 0.70
    else:
        shape_gate = 1.0

    # 2) 再算 shape_scale（一次性）
    shape_scale = float(np.exp(-ratio_std / 0.55)) * float(shape_gate)

    # 位移统计（用 cg / qg 或者原写法都行）
    d = cg - qg                                      # (Ng,2)


    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(Ng)
    m = min(topM, cnt.numel())
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(Ng)

    k = min(topk_core, Ng)
    core = float(torch.topk(q_bestv[good], k=k, largest=True).values.mean().item())

    # 关键：Ng 越少越不可信，强行压分（24~40 可调）
    ng_scale = float(min(1.0, Ng / 32.0))  # Ng=32 才满分，Ng=8 只有 0.25

    is_periodic = (peak_ratio < periodic_peak_thr) and (cover_topM > periodic_cover_thr)

    score = core * peak_ratio * ng_scale * shape_scale

    if is_periodic:
        score *= (0.25 + 0.75 * peak_ratio)  # 周期纹理再压
    # print(f"[DBG] Ng={Ng} peak_ratio={peak_ratio:.3f} ratio_std={ratio_std:.3f} core={core:.3f} score={score:.3f}")

    return float(score)


# =========================
# fusion helpers
# =========================


def _minmax(x: np.ndarray):
    x = x.astype(np.float32)
    mn = float(x.min()); mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-9)

def fuse_scores(coarse01, rerank01, T=0.65, gamma=0.85):
    r = float(rerank01) ** gamma
    c = float(coarse01)

    # 低 coarse：强依赖 rerank
    if c < T:
        return 0.15 * c + 0.85 * r

    # 高 coarse
    return 0.50 * c + 0.50 * r


def enforce_coarse_floor(order_all, coarse_raw, final_scores, topk=10, coarse_topN=60, min_keep=6):
    top = list(order_all[:topk])
    topN = int(min(coarse_topN, len(coarse_raw)))
    coarse_top_set = set(np.argsort(-coarse_raw)[:topN].tolist())
    keep_cnt = sum(1 for i in top if i in coarse_top_set)
    if keep_cnt >= min_keep:
        return np.array(top, dtype=np.int64)

    need = min_keep - keep_cnt
    coarse_rank = np.argsort(-coarse_raw)[:topN]
    fillers = [i for i in coarse_rank if i not in top][:need]
    if not fillers:
        return np.array(top, dtype=np.int64)

    def kick_key(i):
        in_set = (i in coarse_top_set)
        return (in_set, coarse_raw[i])  # 不在set优先踢；其次coarse小优先踢

    top_sorted_to_kick = sorted(top, key=kick_key)
    for f in fillers:
        for j in range(len(top_sorted_to_kick)):
            kick = top_sorted_to_kick[j]
            if kick == f:
                continue
            top.remove(kick)
            top.append(f)
            top_sorted_to_kick.pop(j)
            break

    top = sorted(top, key=lambda i: final_scores[i], reverse=True)
    return np.array(top, dtype=np.int64)

# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)

    index = faiss.read_index(OUT_INDEX)
    print("[INFO] FAISS index type:", type(index), "is_trained:", index.is_trained, "ntotal:", index.ntotal)

    base = index
    while hasattr(base, "index"):
        base = base.index
    print("[DBG] base index:", type(base))

    if hasattr(base, "nprobe") and hasattr(base, "nlist"):
        base.nprobe = min(int(NPROBE), int(base.nlist))
        print(f"[INFO] IVF detected, set nprobe={base.nprobe}/{base.nlist}")
    else:
        print("[INFO] Flat index detected, skip nprobe")

    kept_paths = np.load(OUT_META, allow_pickle=True)
    print(f"[INFO] Loaded meta={len(kept_paths)}")
    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    qimg = imread_unicode(QUERY_IMG)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")
    # ---- ROI gating ----
    conc = grad_concentration_bgr(qimg, top_frac=0.05)

    roi_bbox, roi_img = find_roi_by_grad_energy(qimg, blur_ksize=31, thr_percentile=95, pad_ratio=0.20)
    bbox_ok = roi_is_valid(roi_bbox, qimg.shape, min_area_frac=0.01, max_area_frac=0.45)

    ROI_ON = (conc > 0.12) and bbox_ok
    if not ROI_ON:
        roi_bbox, roi_img = None, None

    print(f"[DBG] conc={conc:.4f} ROI_ON={ROI_ON} bbox={roi_bbox}")


    if TEX_ENABLE:
        q_tex = _fft_texture_desc(qimg, long_edge=TEX_SIZE, crop_frac=TEX_CROP_FRAC)

    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")

    # ---------- Stage-1: coarse (Route-B aligned) ----------
    # views = make_query_views_routeB(qimg, mean, std, to_rgb=to_rgb)
    roi_weight = 0.70 if ROI_ON else 0.0  # ROI_ON就0.70，不开就0.35(可再改)

    views = make_query_views_more_robust(qimg, mean, std, to_rgb=to_rgb, roi_bgr=roi_img, roi_weight=roi_weight)
    print(f"[INFO] coarse views mixed = {len(views)}  roi_weight={roi_weight:.2f}")

    print(f"[INFO] coarse views(RouteB) = {len(views)}")

    bt = torch.stack(views, dim=0).to(DEVICE)
    feats_v = extract_global_feat_for_coarse(model, bt)   # (V,D) L2
    qvec = aggregate_views_to_one(feats_v).unsqueeze(0).cpu().numpy().astype("float32")

    COARSE_K_EFF = min(int(COARSE_K), int(index.ntotal))
    print("[DBG] COARSE_K_EFF =", COARSE_K_EFF)
    scores, ids = index.search(qvec, COARSE_K_EFF)

    print("[DBG] faiss returned:", len(ids[0]), "neg ids:", int(np.sum(ids[0] < 0)))
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    coarse_ids = [i for i in ids if i >= 0]
    coarse_raw = np.array([float(s) for i, s in zip(ids, scores) if i >= 0], dtype=np.float32)

    coarse_paths = [kept_paths[i] for i in coarse_ids]

    # ==== DEBUG: check positives in coarse ====
    pos_names = {"IMG_4833.jpg", "IMG_4834.jpg", "IMG_4835.jpg", "IMG_4836.jpg", "IMG_4837.jpg"}

    hits = []
    for rank, p in enumerate(coarse_paths):
        name = os.path.basename(str(p))
        if name in pos_names:
            hits.append((rank, name, str(p), float(coarse_raw[rank])))

    print("[DBG] positives in coarse:", len(hits))
    for r, name, path, s in hits:
        print(f"  rank={r:4d}  coarse={s:.4f}  {name}  {path}")

    print(f"[INFO] coarse candidates = {len(coarse_paths)}")

    MAX_RERANK = 400
    KEEP_PCTL = 80

    if len(coarse_paths) > MAX_RERANK:
        thr = np.percentile(coarse_raw, KEEP_PCTL)
        keep_idx = [i for i in range(len(coarse_raw))
                    if (i < MAX_RERANK) or (coarse_raw[i] >= thr)]
        coarse_paths = [coarse_paths[i] for i in keep_idx]
        coarse_raw = coarse_raw[keep_idx]
        print(f"[INFO] rerank cut: keep {len(coarse_paths)} (top {MAX_RERANK} + coarse>=p{KEEP_PCTL})")

    # ---------- Stage-2: rerank ----------
    qx = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    q_fm = extract_featmap(model, qx, FEAT_LEVEL)

    # 1) 给 query 找一个能量 ROI（别太小）
    q_roi = energy_roi_box(q_fm[0], frac=0.18)  # 建议 0.15~0.25

    # 2) ROI + 全图混合选 patch：避免 ROI 太小导致 Ng 不够 -> rerank=0
    k_roi = KEEP_PATCHES // 2
    k_full = KEEP_PATCHES - k_roi

    q_desc_list = []
    q_xy_list = []

    # ROI patches（如果 ROI 有效）
    if q_roi is not None:
        q_desc_roi, q_xy_roi = select_top_patches_with_xy(
            q_fm, keep=k_roi, border=BORDER, roi_fbox=q_roi
        )
        if (q_desc_roi is not None) and (q_xy_roi is not None) and (q_desc_roi.shape[0] >= 4):
            q_desc_list.append(q_desc_roi)
            q_xy_list.append(q_xy_roi)

    # Full-image patches（兜底，始终取）
    q_desc_full, q_xy_full = select_top_patches_with_xy(
        q_fm, keep=k_full, border=BORDER, roi_fbox=None
    )
    if q_desc_full is None or q_xy_full is None:
        raise RuntimeError("Query patch selection failed.")

    q_desc_list.append(q_desc_full)
    q_xy_list.append(q_xy_full)

    q_desc = torch.cat(q_desc_list, dim=0)  # (K,C)
    q_xy = torch.cat(q_xy_list, dim=0)  # (K,2)

    cand_tensors = []
    valid_paths = []
    valid_coarse_raw = []

    valid_tex = []
    for p, cs in zip(coarse_paths, coarse_raw.tolist()):
        img = imread_unicode(str(p))
        if img is None:
            continue
        cand_tensors.append(make_single_tensor_for_rerank(img, mean, std, to_rgb=to_rgb))
        valid_paths.append(str(p))
        valid_coarse_raw.append(float(cs))
        # texture
        try:
            if TEX_ENABLE:
                c_tex = _fft_texture_desc(img, long_edge=TEX_SIZE, crop_frac=TEX_CROP_FRAC)
                valid_tex.append(float(np.clip(np.dot(q_tex, c_tex), 0.0, 1.0)))
            else:
                valid_tex.append(1.0)
        except Exception:
            valid_tex.append(0.0)  # 纹理算失败：当成不相似，后面会被惩罚



    if not cand_tensors:
        raise RuntimeError("All coarse candidate images failed to read.")

    rerank_raw = []
    bs2 = 8
    for st in range(0, len(cand_tensors), bs2):
        bt2 = torch.cat(cand_tensors[st:st + bs2], dim=0).to(DEVICE)
        fm = extract_featmap(model, bt2, FEAT_LEVEL)
        _, _, Hf, Wf = fm.shape
        roi_fboxes = [energy_roi_box(fm[i]) for i in range(fm.shape[0])]
        c_list = select_top_patches_with_xy_batch(fm, keep=KEEP_PATCHES, border=BORDER, roi_fboxes=roi_fboxes)

        for c_desc, c_xy in c_list:
            s = geom_score_adaptive(
                q_desc, q_xy, c_desc, c_xy,
                margin=MARGIN, min_keep=MIN_KEEP,
                bin_size=BIN_SIZE, topM=TOPM, topk_core=TOPK_CORE,
                periodic_peak_thr=PERIODIC_PEAK_THR,
                periodic_cover_thr=PERIODIC_COVER_THR
            )
            rerank_raw.append(float(s))

    coarse_raw2 = np.array(valid_coarse_raw, dtype=np.float32)
    rerank_raw2 = np.array(rerank_raw, dtype=np.float32)


    def norm_percentile(x: np.ndarray, lo=50, hi=95, eps=1e-9):
        x = x.astype(np.float32)
        p_lo = np.percentile(x, lo)
        p_hi = np.percentile(x, hi)
        y = (x - p_lo) / (p_hi - p_lo + eps)
        return np.clip(y, 0.0, 1.0)

    cs01 = norm_percentile(coarse_raw2, lo=50, hi=95)
    # 用 raw 分数的分位数做中心与尺度，然后 sigmoid 映射
    r = rerank_raw2.astype(np.float32)
    r[r < 1e-8] = 0.0

    nz = r[r > 0]
    if nz.size < 50:
        # 非零太少：直接用你之前的 percentile norm（更稳）
        rs01 = norm_percentile(r, lo=60, hi=98)
        rs01[r <= 1e-8] = 0.0
    else:
        p50 = np.percentile(nz, 50)
        p90 = np.percentile(nz, 90)
        scale = (p90 - p50) + 1e-6
        rs01 = 1.0 / (1.0 + np.exp(-(r - p50) / (0.40 * scale)))
        rs01[r <= 1e-8] = 0.0

    rs_adj = rs01
    # ---- tex gate ----
    tex = np.array(valid_tex, dtype=np.float32)

    low_mask = tex < 0.88
    high_mask = tex > 0.94

    print("[DBG] tex stats:", float(tex.min()), float(tex.mean()), float(tex.max()))
    print("[DBG] tex p10/p50/p90:", np.quantile(tex, 0.1), np.quantile(tex, 0.5), np.quantile(tex, 0.9))

    if TEX_ENABLE:
        # t = np.clip((tex - TEX_THR_LOW) / (TEX_THR_HIGH - TEX_THR_LOW + 1e-9), 0.0, 1.0)
        # tex_scale = 0.40 + 0.65 * t  # [0.40, 1.05]
        # rs_adj = np.clip(rs_adj * tex_scale, 0.0, 1.0)
        rs_adj = rs01.copy()
        rs_adj[low_mask] *= 0.45  # 强压
        rs_adj[high_mask] = np.clip(rs_adj[high_mask] + 0.02, 0.0, 1.0)

    # ---- fuse ----
    GATE_T2 = 0.65

    final = np.array([fuse_scores(cs01[i], rs_adj[i], T=GATE_T2, gamma=GAMMA)
                      for i in range(len(cs01))], dtype=np.float32)
    # 只打击：coarse 很高但 rerank 很低（coarse 假阳性）
    HI_C_T = 0.88
    LOW_R_T = 0.45
    PENALTY = 0.55

    mask = (cs01 > HI_C_T) & (rs01 < LOW_R_T)
    final[mask] *= PENALTY

    # ---- high coarse tiny bonus (optional) ----
    c_hi = np.clip((cs01 - 0.92) / 0.08, 0.0, 1.0)
    # final = np.clip(final + 0.05 * c_hi, 0.0, 1.0)

    # ---- debug pos ----
    pos_names = {"IMG_4833.jpg", "IMG_4834.jpg", "IMG_4835.jpg", "IMG_4836.jpg", "IMG_4837.jpg"}
    for i, p in enumerate(valid_paths):
        if os.path.basename(p) in pos_names:
            print(f"[POS] {p} coarse01={cs01[i]:.4f} rerank01={rs01[i]:.4f} "
                  f"tex={tex[i]:.4f} rs_adj={rs_adj[i]:.4f} final={final[i]:.4f}")

    # ---- enforce coarse floor (MUST last) ----
    order_all = np.argsort(-final)
    order_top = enforce_coarse_floor(
        order_all=order_all,
        coarse_raw=coarse_raw2,
        final_scores=final,
        topk=TOPK,
        coarse_topN=60,
        min_keep=6
    )

    top_paths = [valid_paths[i] for i in order_top]
    top_scores = [float(final[i]) for i in order_top]

    print("\n===== TOPK (FINAL fused) =====")
    for r, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        print(f"{r:02d}  score={s:.4f}  {p}")

    out = make_topk_viz(QUERY_IMG, top_paths, top_scores, OUT_VIZ, tile=TILE)
    print(f"\n[VIZ] saved -> {out}")

    for i in order_top:
        p = valid_paths[i]
        print(
            f"[DBG] {p}\n"
            f"  coarse_raw={coarse_raw2[i]:.4f} coarse01={cs01[i]:.4f}\n"
            f"  rerank01={rs01[i]:.4f} rs_adj={rs_adj[i]:.4f}\n"
            f"  tex={tex[i]:.4f} final={final[i]:.4f}"
        )


if __name__ == "__main__":
    main()
