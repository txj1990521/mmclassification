# zhanlan/ivfflat/zhanlan_retrieval/cropper.py
import os
import time
import cv2
import numpy as np

from .geom_utils import safe_pad_crop, largest_cc, expand_bbox_to_limit_ar
from .yolo_seg import yolo_extract_mask_u8, YoloSegProvider


def fill_background(crop_bgr, crop_mask_u8, mode="mean"):
    if crop_bgr is None or crop_mask_u8 is None:
        return crop_bgr
    m = crop_mask_u8.astype(bool)
    if m.sum() < 10:
        return crop_bgr

    out = crop_bgr.copy()
    if mode == "white":
        out[~m] = (255, 255, 255)
        return out
    if mode == "edge":
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out[~m] = blur[~m]
        return out

    mean_color = out[m].mean(axis=0)
    out[~m] = mean_color
    return out


def crop_by_yolo_mask_final(
        yolo_provider: YoloSegProvider,
        img_bgr: np.ndarray,
        pad: int = 10,
        min_area_frac: float = 0.06,
        score_thr: float = 0.6,
        use_classes=None,
        merge_all: bool = True,
        do_rectify: bool = True,
        rectify_pad: int = 10,
        warp_border: str = "reflect",   # "reflect" | "replicate"
        bg_mode: str = "mean",          # "mean" | "white" | "edge"
        debug_dir: str = None,
        yolo_imgsz: int = 640,
        yolo_iou: float = 0.5,
        yolo_retina_masks: bool = True,
        yolo_device=0,
        yolo_max_det: int = 100,
):
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, None, None

    H, W = img_bgr.shape[:2]
    default_mask = np.ones((H, W), dtype=np.uint8) * 255
    default_raw = img_bgr

    model = yolo_provider.model()
    results = model.predict(
        source=img_bgr,
        conf=float(score_thr),
        iou=float(yolo_iou),
        imgsz=int(yolo_imgsz),
        device=yolo_device,
        max_det=int(yolo_max_det),
        retina_masks=bool(yolo_retina_masks),
        classes=use_classes,
        verbose=False,
        stream=False,
        save=False,
        show=False
    )
    if results is None or len(results) == 0:
        return img_bgr, default_mask, default_raw

    r0 = results[0]
    mask_u8 = yolo_extract_mask_u8(
        r0, H, W,
        conf_thr=float(score_thr),
        use_classes=use_classes,
        merge_all=merge_all
    )
    if mask_u8 is None:
        return img_bgr, default_mask, default_raw

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, ker, iterations=1)

    mask_u8 = largest_cc(mask_u8)
    if mask_u8 is None:
        return img_bgr, default_mask, default_raw

    area = float((mask_u8 > 0).sum())
    if area < float(min_area_frac) * (H * W):
        return img_bgr, default_mask, default_raw

    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 20:
        return img_bgr, default_mask, default_raw

    if do_rectify:
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), ang = rect
        if rw < rh:
            ang = ang + 90.0

        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        borderMode = cv2.BORDER_REFLECT101 if warp_border == "reflect" else cv2.BORDER_REPLICATE

        rot_img = cv2.warpAffine(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=borderMode)
        rot_msk = cv2.warpAffine(mask_u8, M, (W, H), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        ys2, xs2 = np.where(rot_msk > 0)
        if ys2.size < 20:
            return img_bgr, default_mask, default_raw

        x1, x2 = xs2.min() - rectify_pad, xs2.max() + 1 + rectify_pad
        y1, y2 = ys2.min() - rectify_pad, ys2.max() + 1 + rectify_pad
        crop_img = safe_pad_crop(rot_img, x1, y1, x2, y2)
        crop_msk = safe_pad_crop(rot_msk, x1, y1, x2, y2)
    else:
        x1, x2 = xs.min() - pad, xs.max() + 1 + pad
        y1, y2 = ys.min() - pad, ys.max() + 1 + pad
        bw = (x2 - x1)
        bh = (y2 - y1)
        ar = max(bh / (bw + 1e-9), bw / (bh + 1e-9))
        if ar > 8.0:
            x1, y1, x2, y2 = expand_bbox_to_limit_ar(x1, y1, x2, y2, H, W, max_ar=8.0)

        crop_img = safe_pad_crop(img_bgr, x1, y1, x2, y2)
        crop_msk = safe_pad_crop(mask_u8, x1, y1, x2, y2)

    if crop_img is None or crop_msk is None or crop_img.size == 0:
        return img_bgr, default_mask, default_raw

    crop_msk_u8 = (crop_msk > 0).astype(np.uint8) * 255
    crop_msk_bool = crop_msk_u8.astype(bool)

    filled = fill_background(crop_img, crop_msk_u8, mode=bg_mode)
    out = filled.copy()
    out[crop_msk_bool] = crop_img[crop_msk_bool]

    tag = f"q_{int(time.time())}"
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_mask.png"), mask_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_mask.png"), crop_msk_u8)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_raw.png"), crop_img)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_crop_out.png"), out)

    return out, crop_msk_u8, crop_img
