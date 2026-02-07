#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for Zhanlan Hybrid retrieval (build + query)
Goal: keep window generation / randomness / resize logic identical on both sides.

This file contains:
- STRIPE constants (must match build & query)
- image resize / crop / rotate helpers
- deterministic seeds
- stripe windows generator (sliding along long edge)
- grid windows generator (multi-scale + center fallback + deterministic sampling)
- unified patch windows entry (stripe vs grid)
"""

import hashlib
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


# ============================================================
# 1) Shared constants (STRIPE)
# ============================================================
STRIPE_AR_THR = 2.7
STRIPE_LONG_EDGE = 1536

STRIPE_WIN_H = 224
STRIPE_STRIDE = 48
STRIPE_MAX_PATCHES = 24
STRIPE_CENTER_FRAC = 0.92
STRIPE_JITTER = 8

STRIPE_WIN_W_FRAC = 0.85
STRIPE_WIN_W_MIN = 160
STRIPE_WIN_W_MAX = 256


# ============================================================
# 2) Shared constants (GRID)
# ============================================================
MAX_LONG = 1536
PATCH_SIZES = (256, 384, 512, 768)
STRIDE_RATIO = 0.5
MAX_PATCHES_PER_IMAGE = 60

PATCH_ENC_SIZE = 224


# ============================================================
# 3) Shared types
# ============================================================
# window tuple: (x1,y1,x2,y2,win,ptype,pos_int)
# ptype: 0=grid, 1=stripe
# pos_int: stripe: 0..10000 (pos along long axis), grid: 5000
Window = Tuple[int, int, int, int, int, int, int]


# ============================================================
# 4) IO / image basics
# ============================================================
def imread_unicode(p: str):
    """Windows unicode path safe imread."""
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def resize_long_edge(img_bgr: np.ndarray, max_long: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = max_long / max(h, w)
    if s >= 1.0:
        return img_bgr
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def resize_short_edge(img_rgb: np.ndarray, short: int = 256) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def center_crop(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img_rgb[y1:y1 + size, x1:x1 + size]


def random_crop(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y = random.randint(0, h - size)
    x = random.randint(0, w - size)
    return img_rgb[y:y + size, x:x + size]


def rotate_bound(img_rgb: np.ndarray, deg: float) -> np.ndarray:
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
    return cv2.warpAffine(
        img_rgb, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )


# ============================================================
# 5) Seeds (deterministic)
# ============================================================
def seed_from_image(img_bgr: np.ndarray, base: int = 999) -> int:
    """
    Deterministic seed from image bytes.
    IMPORTANT: build & query must feed the SAME pre-resized image to this function.
    """
    if img_bgr is None or img_bgr.size == 0:
        return base & 0x7fffffff
    h = hashlib.md5(img_bgr.tobytes()).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff


def seed_from_path(p: str, base: int = 0) -> int:
    h = hashlib.md5(str(p).encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff


# ============================================================
# 6) Tensor conversion (shared)
# ============================================================
def to_tensor_from_rgb(img_rgb: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def patch_to_model_input(
    patch_bgr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool,
    out_size: int = PATCH_ENC_SIZE
) -> torch.Tensor:
    if to_rgb:
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    else:
        patch_rgb = patch_bgr[:, :, ::-1].copy()
    patch_rgb = cv2.resize(patch_rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return to_tensor_from_rgb(patch_rgb, mean, std)


# ============================================================
# 7) Stripe windows
# ============================================================
def gen_stripe_windows(
    H: int, W: int,
    max_patches: int,
    seed: int,
    win_w: int,
    win_h: int = STRIPE_WIN_H,
    stride: Optional[int] = STRIPE_STRIDE,
    center_frac: float = STRIPE_CENTER_FRAC,
    jitter: int = STRIPE_JITTER,
) -> List[Window]:
    """
    Return windows: list of (x1,y1,x2,y2,win,ptype,pos_int)
    ptype=1 (stripe), pos_int in [0..10000] along long axis
    """
    rng = np.random.default_rng(seed)
    vertical = (H >= W)

    windows: List[Window] = []
    if stride is None:
        stride = max(32, win_h // 4)

    if vertical:
        Wu = int(round(W * center_frac))
        x0 = max(0, (W - Wu) // 2)

        x_base = x0 + max(0, (Wu - min(win_w, Wu)) // 2)
        ww = min(win_w, Wu)
        hh = min(win_h, H)
        if ww < 16 or hh < 16:
            return []

        y = 0
        while y + hh <= H and len(windows) < max_patches:
            dx = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            x1 = int(np.clip(x_base + dx, x0, x0 + Wu - ww))
            y1 = int(y)
            x2 = x1 + ww
            y2 = y1 + hh
            center = (y1 + y2) * 0.5
            pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            y += stride

        # tail coverage
        if windows:
            last_y2 = windows[-1][3]
            if last_y2 < H:
                y1 = max(0, H - hh)
                x1 = windows[-1][0]
                x2 = x1 + ww
                y2 = y1 + hh
                center = (y1 + y2) * 0.5
                pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
                windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    else:
        Hu = int(round(H * center_frac))
        y0 = max(0, (H - Hu) // 2)

        hh = min(win_w, Hu)  # thickness
        ww = min(win_h, W)   # length
        if hh < 16 or ww < 16:
            return []

        y_base = y0 + max(0, (Hu - hh) // 2)

        x = 0
        while x + ww <= W and len(windows) < max_patches:
            dy = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            y1 = int(np.clip(y_base + dy, y0, y0 + Hu - hh))
            x1 = int(x)
            x2 = x1 + ww
            y2 = y1 + hh
            center = (x1 + x2) * 0.5
            pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            x += stride

        # tail coverage
        if windows:
            last_x2 = windows[-1][2]
            if last_x2 < W:
                x1 = max(0, W - ww)
                y1 = windows[-1][1]
                x2 = x1 + ww
                y2 = y1 + hh
                center = (x1 + x2) * 0.5
                pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
                windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    return windows


# ============================================================
# 8) Grid windows (match build logic)
# ============================================================
def sample_windows_deterministic(windows: List[Window], k: int, seed: int) -> List[Window]:
    """Deterministically sample k windows from list."""
    if len(windows) <= k:
        return windows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=k, replace=False)
    idx = np.sort(idx)
    return [windows[i] for i in idx.tolist()]


def gen_grid_windows_buildstyle(
    H: int, W: int,
    sizes: Tuple[int, ...] = PATCH_SIZES,
    stride_ratio: float = STRIDE_RATIO,
    max_patches: int = MAX_PATCHES_PER_IMAGE,
    seed: int = 0,
) -> List[Window]:
    """
    EXACTLY match your build-side grid window enumeration:
    - multi-scale grid
    - center fallback window
    - dedupe via dict.fromkeys
    - deterministic sampling if too many
    """
    windows: List[Window] = []

    for win in sizes:
        if H < win or W < win:
            continue
        stride = max(1, int(round(win * stride_ratio)))
        for y1 in range(0, H - win + 1, stride):
            for x1 in range(0, W - win + 1, stride):
                windows.append((x1, y1, x1 + win, y1 + win, win, 0, 5000))

    # center fallback
    for win in (512, 768, 384, 256):
        if H >= win and W >= win:
            cx1 = (W - win) // 2
            cy1 = (H - win) // 2
            windows.append((cx1, cy1, cx1 + win, cy1 + win, win, 0, 5000))
            break

    # dedupe
    windows = list(dict.fromkeys(windows))

    # cap (deterministic)
    if max_patches is not None and max_patches > 0:
        windows = sample_windows_deterministic(windows, max_patches, seed=seed)

    return windows


# ============================================================
# 9) Unified patch windows (stripe vs grid) + resize
# ============================================================
def gen_patch_windows_unified(
    img_bgr: np.ndarray,
    max_long: int = MAX_LONG,
    stripe_ar_thr: float = STRIPE_AR_THR,
    seed_base: int = 999,
    # stripe params:
    stripe_max_patches: int = STRIPE_MAX_PATCHES,
    stripe_win_h: int = STRIPE_WIN_H,
    stripe_stride: int = STRIPE_STRIDE,
    stripe_center_frac: float = STRIPE_CENTER_FRAC,
    stripe_jitter: int = STRIPE_JITTER,
    # grid params:
    grid_sizes: Tuple[int, ...] = PATCH_SIZES,
    grid_stride_ratio: float = STRIDE_RATIO,
    max_patches: int = MAX_PATCHES_PER_IMAGE,
) -> Tuple[np.ndarray, List[Window], bool]:
    """
    Return:
      img_resized (bgr),
      windows on resized coords,
      is_stripe
    """
    img = resize_long_edge(img_bgr, max_long=max_long)
    H, W = img.shape[:2]

    ar = max(W / (H + 1e-6), H / (W + 1e-6))
    is_stripe = (ar >= stripe_ar_thr)

    s = seed_from_image(img, base=seed_base)

    if is_stripe:
        win_w = int(np.clip(STRIPE_WIN_W_FRAC * W, STRIPE_WIN_W_MIN, STRIPE_WIN_W_MAX))
        windows = gen_stripe_windows(
            H, W,
            max_patches=stripe_max_patches,
            seed=s,
            win_w=win_w,
            win_h=stripe_win_h,
            stride=stripe_stride,
            center_frac=stripe_center_frac,
            jitter=stripe_jitter,
        )
        # fallback big center window
        if len(windows) < min(6, stripe_max_patches):
            ww = min(max(win_w, 160), W)
            hh = min(max(stripe_win_h, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            windows.append((x1, y1, x1 + ww, y1 + hh, int(max(ww, hh)), 1, 5000))
        return img, windows, True

    windows = gen_grid_windows_buildstyle(
        H, W,
        sizes=grid_sizes,
        stride_ratio=grid_stride_ratio,
        max_patches=max_patches,
        seed=s
    )
    return img, windows, False
