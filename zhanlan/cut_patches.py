#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_patches.py
--------------
Fabric pattern patch cutter (v1) for mobile photos.

What it does:
- Sliding-window generate candidate patches at multiple scales
- Score each patch by texture/texture_ratio/sharpness + illumination penalty
- Select Top-K with spatial de-dup (NMS / distance)
- Export patches + patches.json + optional debug overlay image

Dependencies:
  pip install opencv-python numpy

Example:
  python cut_patches.py \
    --input /path/to/images \
    --out   /path/to/out \
    --patch-sizes 512 256 \
    --stride 160 80 \
    --topk 25 35 \
    --make-debug

Tips:
- If your images are huge and slow, add: --max-long-side 2500
- If patches often hit cloth borders, enable: --edge-penalty
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in IMG_EXTS


def list_images(input_path: str, recursive: bool = True) -> List[str]:
    paths = []
    if os.path.isfile(input_path):
        return [input_path] if is_image_file(input_path) else []
    for root, _, files in os.walk(input_path):
        for fn in files:
            p = os.path.join(root, fn)
            if is_image_file(p):
                paths.append(p)
        if not recursive:
            break
    paths.sort()
    return paths


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_imread(path: str) -> Optional[np.ndarray]:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def resize_max_long_side(img: np.ndarray, max_long_side: int) -> Tuple[np.ndarray, float]:
    if max_long_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return img, 1.0
    scale = max_long_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return (x, y, x + w, y + h)


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter) / float(area_a + area_b - inter + 1e-9)


def nms_xywh(items: List[dict], iou_thr: float) -> List[dict]:
    """items must include 'xywh' and 'score'."""
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: d["score"], reverse=True)
    keep = []
    for it in items_sorted:
        ok = True
        for k in keep:
            if iou_xywh(it["xywh"], k["xywh"]) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(it)
    return keep


def center_distance(a_xywh, b_xywh) -> float:
    ax, ay, aw, ah = a_xywh
    bx, by, bw, bh = b_xywh
    acx, acy = ax + aw / 2.0, ay + ah / 2.0
    bcx, bcy = bx + bw / 2.0, by + bh / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def dedup_by_center_distance(items: List[dict], min_dist: float) -> List[dict]:
    """Greedy keep by score; drop if center too close."""
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: d["score"], reverse=True)
    keep = []
    for it in items_sorted:
        if all(center_distance(it["xywh"], k["xywh"]) >= min_dist for k in keep):
            keep.append(it)
    return keep


@dataclass
class PatchScore:
    texture: float
    texture_ratio: float
    sharpness: float
    illum_penalty: float
    edge_penalty: float
    score: float


def compute_patch_score(
    patch_bgr: np.ndarray,
    texture_ratio_min: float = 0.10,
    sharpness_min: float = 40.0,
    use_edge_penalty: bool = False,
    edge_band_ratio: float = 0.15,
) -> Tuple[PatchScore, bool]:
    """
    Returns (PatchScore, passed_basic_filters).
    Basic filters: texture_ratio >= texture_ratio_min and sharpness >= sharpness_min
    """
    # Gray
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

    # Gradient magnitude (texture)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    texture = float(np.mean(mag))

    # texture_ratio (adaptive threshold by percentile)
    # If mag is near-zero everywhere, percentile is 0; handle safely.
    t = float(np.percentile(mag, 70))
    if t <= 1e-6:
        texture_ratio = 0.0
    else:
        texture_ratio = float(np.mean(mag > t))

    # Sharpness: var of Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness = float(np.var(lap))

    # Illumination penalty: too dark / too bright / too flat
    m = float(np.mean(gray))
    s = float(np.std(gray))
    illum_penalty = 0.0
    if m < 40:
        illum_penalty += (40 - m) / 40.0
    elif m > 215:
        illum_penalty += (m - 215) / 40.0
    if s < 18:
        illum_penalty += (18 - s) / 18.0

    # Optional edge concentration penalty
    edge_penalty = 0.0
    if use_edge_penalty:
        h, w = gray.shape[:2]
        band = int(round(min(h, w) * edge_band_ratio))
        band = max(1, min(band, min(h, w) // 3))

        # border mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:band, :] = 1
        mask[-band:, :] = 1
        mask[:, :band] = 1
        mask[:, -band:] = 1

        border_energy = float(np.mean(mag[mask == 1])) if np.any(mask == 1) else 0.0
        center_energy = float(np.mean(mag[mask == 0])) if np.any(mask == 0) else 0.0
        # penalize when border dominates
        if center_energy > 1e-6:
            ratio = border_energy / center_energy
            if ratio > 1.3:
                edge_penalty = min(2.0, (ratio - 1.3))  # capped
        else:
            # if center has no energy but border does, heavy penalty
            if border_energy > 1e-6:
                edge_penalty = 2.0

    # Basic filters
    passed = (texture_ratio >= texture_ratio_min) and (sharpness >= sharpness_min)

    # score weights (mobile-photo tuned)
    # Normalize later across candidates of same scale; here keep raw components.
    # We'll compute final score after normalization. For now store raw and a preliminary score.
    score = 0.0  # placeholder; will be overwritten after normalization

    return PatchScore(
        texture=texture,
        texture_ratio=texture_ratio,
        sharpness=sharpness,
        illum_penalty=illum_penalty,
        edge_penalty=edge_penalty,
        score=score,
    ), passed


def normalize_list(vals: List[float]) -> List[float]:
    if not vals:
        return []
    v = np.array(vals, dtype=np.float32)
    # robust normalization by percentiles
    lo = float(np.percentile(v, 5))
    hi = float(np.percentile(v, 95))
    if hi - lo < 1e-9:
        return [0.0 for _ in vals]
    v = (v - lo) / (hi - lo)
    v = np.clip(v, 0.0, 1.0)
    return v.tolist()


def generate_windows(w: int, h: int, patch: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) within image bounds."""
    if patch > w or patch > h:
        return []
    xs = list(range(0, w - patch + 1, stride))
    ys = list(range(0, h - patch + 1, stride))

    # Ensure last coverage near the end
    if xs and xs[-1] != (w - patch):
        xs.append(w - patch)
    if ys and ys[-1] != (h - patch):
        ys.append(h - patch)

    boxes = [(x, y, patch, patch) for y in ys for x in xs]
    return boxes


def draw_debug(img_bgr: np.ndarray, selected: List[dict], color=(0, 255, 0)) -> np.ndarray:
    out = img_bgr.copy()
    for it in selected:
        x, y, w, h = it["xywh"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        txt = f'{it["score"]:.2f}'
        cv2.putText(out, txt, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def save_image(path: str, img_bgr: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    else:
        params = []
    # handle unicode paths on windows
    success, buf = cv2.imencode(ext if ext else ".jpg", img_bgr, params)
    if success:
        buf.tofile(path)
    else:
        cv2.imwrite(path, img_bgr)


def cut_one_image(
    image_path: str,
    out_dir: str,
    patch_sizes: List[int],
    strides: List[int],
    topk_per_size: List[int],
    texture_ratio_min: float,
    sharpness_min_512: float,
    sharpness_min_256: float,
    nms_iou: float,
    dedup_center_ratio: float,
    max_long_side: int,
    make_debug: bool,
    edge_penalty: bool,
) -> Dict:
    img0 = safe_imread(image_path)
    if img0 is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    img, scale = resize_max_long_side(img0, max_long_side)
    h, w = img.shape[:2]

    # Prepare per-size results
    all_selected = []
    summary_by_size = []

    for ps, st, topk in zip(patch_sizes, strides, topk_per_size):
        boxes = generate_windows(w, h, ps, st)
        candidates = []
        # Evaluate each candidate
        sharp_min = sharpness_min_512 if ps >= 512 else sharpness_min_256

        for (x, y, bw, bh) in boxes:
            patch = img[y : y + bh, x : x + bw]
            sc, passed = compute_patch_score(
                patch,
                texture_ratio_min=texture_ratio_min,
                sharpness_min=sharp_min,
                use_edge_penalty=edge_penalty,
            )
            if not passed:
                continue
            candidates.append(
                {
                    "xywh": [int(x), int(y), int(bw), int(bh)],
                    "patch_size": int(ps),
                    "texture": sc.texture,
                    "texture_ratio": sc.texture_ratio,
                    "sharpness": sc.sharpness,
                    "illum_penalty": sc.illum_penalty,
                    "edge_penalty": sc.edge_penalty,
                    "score": 0.0,  # will fill after normalization
                }
            )

        # If too few candidates, relax a bit by only filtering texture_ratio (keep sharpness filter)
        if len(candidates) < max(10, topk // 2):
            relaxed = []
            for (x, y, bw, bh) in boxes:
                patch = img[y : y + bh, x : x + bw]
                sc, _ = compute_patch_score(
                    patch,
                    texture_ratio_min=max(0.06, texture_ratio_min * 0.7),
                    sharpness_min=sharp_min,
                    use_edge_penalty=edge_penalty,
                )
                # accept relaxed
                if sc.sharpness >= sharp_min and sc.texture_ratio >= max(0.06, texture_ratio_min * 0.7):
                    relaxed.append(
                        {
                            "xywh": [int(x), int(y), int(bw), int(bh)],
                            "patch_size": int(ps),
                            "texture": sc.texture,
                            "texture_ratio": sc.texture_ratio,
                            "sharpness": sc.sharpness,
                            "illum_penalty": sc.illum_penalty,
                            "edge_penalty": sc.edge_penalty,
                            "score": 0.0,
                        }
                    )
            candidates = relaxed

        # Normalize per-size
        tex_n = normalize_list([c["texture"] for c in candidates])
        tr_n = normalize_list([c["texture_ratio"] for c in candidates])
        sh_n = normalize_list([c["sharpness"] for c in candidates])
        ip_n = normalize_list([c["illum_penalty"] for c in candidates])
        ep_n = normalize_list([c["edge_penalty"] for c in candidates])

        # Final score weights (mobile tuned)
        # score = 0.45*texture + 0.25*texture_ratio + 0.25*sharpness - 0.05*illum - 0.05*edge
        for i, c in enumerate(candidates):
            score = (
                0.45 * tex_n[i]
                + 0.25 * tr_n[i]
                + 0.25 * sh_n[i]
                - 0.05 * ip_n[i]
                - 0.05 * ep_n[i]
            )
            c["score"] = float(score)

        candidates.sort(key=lambda d: d["score"], reverse=True)
        candidates = candidates[: max(topk * 6, 200)]  # limit before dedup

        # De-dup: NMS then optional center distance
        after_nms = nms_xywh(candidates, nms_iou)

        # Further enforce diversity by center distance (optional)
        min_center_dist = ps * dedup_center_ratio
        if dedup_center_ratio > 0:
            after_nms = dedup_by_center_distance(after_nms, min_center_dist)

        selected = after_nms[:topk]

        all_selected.extend(selected)
        summary_by_size.append(
            {
                "patch_size": ps,
                "stride": st,
                "num_windows": len(boxes),
                "num_candidates_after_filter": len(candidates),
                "num_selected": len(selected),
            }
        )

    # If we resized, map xywh back to original coordinates for metadata saving
    def map_back_xywh(xywh):
        if scale == 1.0:
            return xywh
        x, y, ww, hh = xywh
        x0 = int(round(x / scale))
        y0 = int(round(y / scale))
        w0 = int(round(ww / scale))
        h0 = int(round(hh / scale))
        return [x0, y0, w0, h0]

    # Prepare output dirs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_item_dir = os.path.join(out_dir, base_name)
    patches_dir = os.path.join(out_item_dir, "patches")
    ensure_dir(patches_dir)

    # Save patches
    patches_meta = []
    for idx, it in enumerate(all_selected):
        x, y, pw, ph = it["xywh"]
        patch = img[y : y + ph, x : x + pw]
        # map back for metadata (original image coordinates)
        xywh_orig = map_back_xywh(it["xywh"])
        patch_path = os.path.join(patches_dir, f"{idx:03d}.jpg")
        save_image(patch_path, patch)

        patches_meta.append(
            {
                "id": idx,
                "file": f"patches/{idx:03d}.jpg",
                "xywh": xywh_orig,
                "patch_size": int(it["patch_size"]),
                "score": float(it["score"]),
                "texture": float(it["texture"]),
                "texture_ratio": float(it["texture_ratio"]),
                "sharpness": float(it["sharpness"]),
                "illum_penalty": float(it["illum_penalty"]),
                "edge_penalty": float(it["edge_penalty"]),
            }
        )

    # Debug overlay on original resolution
    debug_path = None
    if make_debug:
        # draw on resized then scale back? Better draw directly on original using mapped boxes.
        dbg = img0.copy()
        for pm in patches_meta:
            x, y, ww, hh = pm["xywh"]
            cv2.rectangle(dbg, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.putText(
                dbg,
                f'{pm["score"]:.2f}',
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        debug_path = os.path.join(out_item_dir, "debug.jpg")
        save_image(debug_path, dbg)

    # Save JSON
    meta = {
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "original_hw": [int(img0.shape[0]), int(img0.shape[1])],
        "processed_hw": [int(h), int(w)],
        "resize_scale": float(scale),
        "patch_sizes": [int(x) for x in patch_sizes],
        "strides": [int(x) for x in strides],
        "topk_per_size": [int(x) for x in topk_per_size],
        "filters": {
            "texture_ratio_min": float(texture_ratio_min),
            "sharpness_min_512": float(sharpness_min_512),
            "sharpness_min_256": float(sharpness_min_256),
            "edge_penalty": bool(edge_penalty),
        },
        "dedup": {
            "nms_iou": float(nms_iou),
            "dedup_center_ratio": float(dedup_center_ratio),
        },
        "summary_by_size": summary_by_size,
        "num_selected_total": len(patches_meta),
        "patches": patches_meta,
    }
    json_path = os.path.join(out_item_dir, "patches.json")
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def parse_args():
    p = argparse.ArgumentParser(description="Fabric pattern patch cutter (v1) for mobile photos")
    p.add_argument("--input", required=True, help="Image file or folder")
    p.add_argument("--out", required=True, help="Output folder")

    p.add_argument("--recursive", action="store_true", help="Recursively scan input folder")
    p.add_argument("--max-long-side", type=int, default=0,
                   help="Resize so max(H,W) <= this value; 0 disables")

    p.add_argument("--patch-sizes", type=int, nargs="+", default=[512, 256],
                   help="Patch sizes, e.g. 512 256")
    p.add_argument("--stride", type=int, nargs="+", default=[160, 80],
                   help="Stride for each patch size, must match count of patch-sizes")
    p.add_argument("--topk", type=int, nargs="+", default=[25, 35],
                   help="TopK for each patch size, must match count of patch-sizes")

    p.add_argument("--texture-ratio-min", type=float, default=0.10,
                   help="Minimum texture_ratio filter (mobile tuned)")
    p.add_argument("--sharpness-min-512", type=float, default=60.0,
                   help="Min sharpness for >=512 patches")
    p.add_argument("--sharpness-min-256", type=float, default=40.0,
                   help="Min sharpness for <512 patches")

    p.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--dedup-center-ratio", type=float, default=0.0,
                   help="Additional dedup by center distance: min_dist = patch_size * ratio; 0 disables")

    p.add_argument("--make-debug", action="store_true", help="Save debug overlay image")
    p.add_argument("--edge-penalty", action="store_true",
                   help="Penalize patches where gradient energy concentrates on borders (cloth edges)")

    p.add_argument("--limit", type=int, default=0, help="Limit number of images processed (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.patch_sizes) != len(args.stride) or len(args.patch_sizes) != len(args.topk):
        print("ERROR: --patch-sizes, --stride, --topk must have the same number of values.", file=sys.stderr)
        sys.exit(2)

    images = list_images(args.input, recursive=args.recursive)
    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        images = images[: args.limit]

    ensure_dir(args.out)

    print(f"Found {len(images)} images.")
    ok = 0
    for i, path in enumerate(images, 1):
        try:
            meta = cut_one_image(
                image_path=path,
                out_dir=args.out,
                patch_sizes=args.patch_sizes,
                strides=args.stride,
                topk_per_size=args.topk,
                texture_ratio_min=args.texture_ratio_min,
                sharpness_min_512=args.sharpness_min_512,
                sharpness_min_256=args.sharpness_min_256,
                nms_iou=args.nms_iou,
                dedup_center_ratio=args.dedup_center_ratio,
                max_long_side=args.max_long_side,
                make_debug=args.make_debug,
                edge_penalty=args.edge_penalty,
            )
            ok += 1
            print(f"[{i}/{len(images)}] OK: {os.path.basename(path)} -> {meta['num_selected_total']} patches")
        except Exception as e:
            print(f"[{i}/{len(images)}] FAIL: {path}\n  {e}", file=sys.stderr)

    print(f"Done. Success: {ok}/{len(images)}. Output: {args.out}")


if __name__ == "__main__":
    main()
