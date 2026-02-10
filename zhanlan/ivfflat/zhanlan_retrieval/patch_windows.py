# zhanlan/ivfflat/zhanlan_retrieval/patch_windows.py
import hashlib
import numpy as np
import cv2


def seed_from_image(img_bgr: np.ndarray, base: int = 999) -> int:
    if img_bgr is None or img_bgr.size == 0:
        return base & 0x7fffffff
    h = hashlib.md5(img_bgr.tobytes()).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff


def resize_long_edge_local(img_bgr, long_edge=1536):
    h, w = img_bgr.shape[:2]
    s = long_edge / float(max(h, w))
    if s >= 1.0:
        return img_bgr
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def gen_stripe_windows(
        H: int, W: int,
        max_patches: int,
        seed: int,
        win_w: int = 192,
        win_h: int = 384,
        stride: int = None,
        center_frac: float = 0.92,
        jitter: int = 8,
):
    rng = np.random.default_rng(seed)
    vertical = (H >= W)

    windows = []
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

        if len(windows) > 0:
            last = windows[-1]
            if last[1] != max(0, H - hh):
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

        hh = min(win_w, Hu)
        ww = min(win_h, W)
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

        if len(windows) > 0:
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


def extract_patches_grid(img_bgr, patch_sizes=(256, 384, 512, 768), stride_ratio=0.5,
                         max_patches=64, border_frac=0.02, roi_xyxy=None,
                         return_xyxy=False):
    H, W = img_bgr.shape[:2]
    if roi_xyxy is not None:
        rx1, ry1, rx2, ry2 = roi_xyxy
        rx1 = max(0, int(rx1)); ry1 = max(0, int(ry1))
        rx2 = min(W, int(rx2)); ry2 = min(H, int(ry2))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, W, H

    crop = img_bgr[ry1:ry2, rx1:rx2]
    HH, WW = crop.shape[:2]

    patches = []
    for ps in patch_sizes:
        if min(HH, WW) < ps:
            continue
        stride = max(1, int(ps * stride_ratio))
        y0 = int(HH * border_frac); x0 = int(WW * border_frac)
        y1m = max(y0, HH - int(HH * border_frac) - ps)
        x1m = max(x0, WW - int(WW * border_frac) - ps)

        ys = list(range(y0, y1m + 1, stride)) if y1m >= y0 else [max(0, (HH - ps)//2)]
        xs = list(range(x0, x1m + 1, stride)) if x1m >= x0 else [max(0, (WW - ps)//2)]

        for yy in ys:
            for xx in xs:
                patch = crop[yy:yy+ps, xx:xx+ps]
                if patch.shape[0] == ps and patch.shape[1] == ps:
                    if return_xyxy:
                        x1 = rx1 + xx; y1 = ry1 + yy
                        x2 = x1 + ps; y2 = y1 + ps
                        patches.append((patch, (x1, y1, x2, y2)))
                    else:
                        patches.append(patch)
                if len(patches) >= max_patches:
                    return patches

    if not patches:
        if return_xyxy:
            patches.append((crop, (rx1, ry1, rx2, ry2)))
        else:
            patches.append(crop)
    return patches
