# zhanlan/ivfflat/zhanlan_retrieval/head_features.py
import cv2
import numpy as np


def make_head_view(img_bgr: np.ndarray, prefer_gray=True):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    H, W = img_bgr.shape[:2]
    y1 = int(H * 0.15); y2 = int(H * 0.85)
    x1 = int(W * 0.10); x2 = int(W * 0.90)
    return img_bgr[y1:y2, x1:x2].copy()


def fft_peak_mass(gray, resize=512, topk_frac=0.002):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32); g -= g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    H, W = mag.shape; cy, cx = H//2, W//2
    r0 = int(min(H, W) * 0.03)
    mag[cy-r0:cy+r0+1, cx-r0:cx+r0+1] = 0

    flat = mag.reshape(-1)
    K = int(topk_frac * flat.size)
    K = max(50, min(K, 4000))
    topk = np.partition(flat, -K)[-K:]
    return float(topk.sum() / (flat.sum() + 1e-6))


def fft_peak_radius(gray, resize=512):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32); gray -= gray.mean()
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(F)
    H, W = mag.shape
    cy, cx = H//2, W//2
    r0 = int(min(H, W) * 0.03)
    mag[cy-r0:cy+r0+1, cx-r0:cx+r0+1] = 0
    y, x = np.unravel_index(np.argmax(mag), mag.shape)
    r = np.sqrt((y - cy)**2 + (x - cx)**2)
    r_norm = r / (0.5 * min(H, W) + 1e-6)
    return float(r_norm)


def fft_stripe_grid_head(gray, resize=512):
    h, w = gray.shape[:2]
    s = resize / float(max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    gray = gray.astype(np.float32)
    gray -= gray.mean()
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))

    H, W = mag.shape
    cy, cx = H//2, W//2
    r = int(min(H, W) * 0.03)
    mag[cy-r:cy+r+1, cx-r:cx+r+1] = 0

    band = int(min(H, W) * 0.04)
    horiz = mag[cy-band:cy+band+1, :].mean()
    vert  = mag[:, cx-band:cx+band+1].mean()
    allm  = mag.mean() + 1e-6

    stripe = float(max(horiz, vert) / allm)
    grid   = float(min(horiz, vert) / allm)

    stripe_score = float(np.clip((stripe - 1.05) / 0.6, 0, 1))
    grid_score   = float(np.clip((grid   - 1.02) / 0.6, 0, 1))
    return {"stripe_score": stripe_score, "grid_score": grid_score}


def stripe_grid_head_v21(img_bgr, resize_long=512, nbins=36):
    if img_bgr is None or img_bgr.size == 0:
        return {"stripe_score": 0.0, "grid_score": 0.0, "ori_peakedness": 0.0}

    h, w = img_bgr.shape[:2]
    s = resize_long / float(max(h, w))
    img = cv2.resize(img_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA) if s < 1.0 else img_bgr

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)

    thr = np.percentile(mag, 50)
    mask = mag > thr
    if mask.sum() < 200:
        return {"stripe_score": 0.0, "grid_score": 0.0, "ori_peakedness": 0.0}

    mag2 = mag[mask]
    ang2 = ang[mask]

    hist, _ = np.histogram(ang2, bins=nbins, range=(0, np.pi), weights=mag2)
    hist = hist.astype(np.float32)
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (1, 5), 0).ravel()

    eps = 1e-6
    p = hist / (hist.sum() + eps)

    bin_angles = (np.arange(nbins) + 0.5) / nbins * np.pi
    d0 = np.minimum(np.abs(bin_angles - 0), np.pi - np.abs(bin_angles - 0))
    d90 = np.abs(bin_angles - (np.pi / 2))
    d = np.minimum(d0, d90)
    w = np.exp(-(d ** 2) / (2 * (0.18 ** 2))).astype(np.float32)
    axis_align = float((p * w).sum())

    peaked = float(p.max() / (p.mean() + eps))

    k1 = int(np.argmax(p))
    ban = max(1, nbins // 18)
    p2 = p.copy()
    p2[max(0, k1-ban):min(nbins, k1+ban+1)] = 0
    k2 = int(np.argmax(p2))

    peak1 = float(p[k1])
    peak2 = float(p[k2])

    a1 = k1 / nbins * np.pi
    a2 = k2 / nbins * np.pi
    dd = abs(a1 - a2)
    dd = min(dd, np.pi - dd)

    ortho = np.exp(-((dd - (np.pi/2))**2) / (2*(0.40**2)))

    stripe_score = peak1 * (peaked / 6.0)
    stripe_score = float(np.clip(stripe_score, 0.0, 1.0))

    grid_score = (0.65*peak1 + 0.35*peak2) * ortho * (peaked / 6.0)
    grid_score = float(np.clip(grid_score, 0.0, 1.0))

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r_peak = fft_peak_radius(gray2, resize=512)
    fft_h = fft_stripe_grid_head(gray2)
    peak_mass = fft_peak_mass(gray2, resize=512)

    stripe_score = max(stripe_score, fft_h["stripe_score"])
    grid_score = max(grid_score, fft_h["grid_score"])

    mag_full = np.sqrt(gx * gx + gy * gy)
    thr2 = np.percentile(mag_full, 80)
    edge_density = float((mag_full > thr2).mean())
    if edge_density < 0.035:
        stripe_score *= 0.3
        grid_score *= 0.3

    return {
        "stripe_score": stripe_score,
        "grid_score": grid_score,
        "ori_peakedness": peaked,
        "axis_align": axis_align,
        "r_peak": r_peak,
        "peak1": peak1,
        "peak2": peak2,
        "ortho": float(ortho),
        "peak_mass": peak_mass
    }


def is_grid_like(h: dict):
    g  = h.get("grid_score", 0.0)
    ax = h.get("axis_align", 0.0)
    pk = h.get("ori_peakedness", 0.0)
    ort= h.get("ortho", 0.0)
    p2 = h.get("peak2", 0.0)
    pm = h.get("peak_mass", 0.0)

    if g >= 0.22 and pm > 0.010:
        return True
    if g < 0.10:
        return False
    if ort < 0.25 or p2 < 0.015:
        return False
    if pk < 2.5:
        return (ax > 0.22) and (pm > 0.008)
    return True
