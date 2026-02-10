# zhanlan/ivfflat/zhanlan_retrieval/geom_utils.py
import cv2
import numpy as np
import random


def rotate_bgr(img, deg):
    if deg == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


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


def safe_pad_crop(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(1, min(W, int(x2)))
    y2 = max(1, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def largest_cc(mask_u8: np.ndarray):
    num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_i = 1 + int(np.argmax(areas))
    return (labels_cc == best_i).astype(np.uint8) * 255


def expand_bbox_to_limit_ar(x1, y1, x2, y2, H, W, max_ar=3.0):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    ar = bh / bw
    if ar <= max_ar:
        return x1, y1, x2, y2

    target_bw = int(np.ceil(bh / max_ar))
    extra = target_bw - bw
    pad_l = extra // 2
    pad_r = extra - pad_l

    nx1 = max(0, x1 - pad_l)
    nx2 = min(W, x2 + pad_r)

    cur_bw = nx2 - nx1
    if cur_bw < target_bw:
        lack = target_bw - cur_bw
        nx1 = max(0, nx1 - lack // 2)
        nx2 = min(W, nx2 + (lack - lack // 2))

    return nx1, y1, nx2, y2


def resize_long_edge(img_bgr: np.ndarray, max_long: int):
    h, w = img_bgr.shape[:2]
    s = max_long / max(h, w)
    if s >= 1.0:
        return img_bgr
    nh, nw = int(round(h*s)), int(round(w*s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


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
    return cv2.warpAffine(img_rgb, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)
