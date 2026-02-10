# zhanlan/ivfflat/zhanlan_retrieval/patch_query.py
import numpy as np
import cv2
import torch

from hybrid_shared import gen_patch_windows_unified

from .patch_windows import (
    seed_from_image,
    resize_long_edge_local,
    gen_stripe_windows,
    extract_patches_grid,
)
from .model_build import extract_backbone_last


@torch.no_grad()
def get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg_bgr,
        qmask=None,
        device: str = "cuda",
        long_edge=1536,
        stripe_ar_thr=2.7,
        stripe_win_h=224,
        stripe_stride=48,
        stripe_max_patches=24,
        stripe_center_frac=0.92,
        stripe_jitter=8,
        stripe_win_w_frac=0.85,
        stripe_win_w_min=160,
        stripe_win_w_max=256,
        min_mask_cover=0.0,
        batch_size=64,
):
    qimg = resize_long_edge_local(qimg_bgr, long_edge=long_edge)
    H, W = qimg.shape[:2]

    qmask_rs = None
    if qmask is not None:
        qmask_rs = cv2.resize(qmask, (W, H), interpolation=cv2.INTER_NEAREST)

    q_seed = seed_from_image(qimg, base=999)

    # unified windows first (kept as in your code)
    qimg_resized, windows_unified, is_stripe = gen_patch_windows_unified(
        qimg_bgr,
        max_long=long_edge,
        stripe_ar_thr=stripe_ar_thr,
        seed_base=999,
        stripe_max_patches=stripe_max_patches,
        stripe_win_h=stripe_win_h,
        stripe_stride=stripe_stride,
        stripe_center_frac=stripe_center_frac,
        stripe_jitter=stripe_jitter,
        grid_sizes=(256, 384, 512, 768),
        grid_stride_ratio=0.5,
        max_patches=60,
    )

    patches = []
    for (x1, y1, x2, y2, win, ptype, pos) in windows_unified:
        patch = qimg_resized[y1:y2, x1:x2]
        patches.append(patch)

    if is_stripe:
        win_w = int(np.clip(stripe_win_w_frac * W, stripe_win_w_min, stripe_win_w_max))
        windows = gen_stripe_windows(
            H, W,
            max_patches=stripe_max_patches,
            seed=q_seed,
            win_w=win_w,
            win_h=stripe_win_h,
            stride=stripe_stride,
            center_frac=stripe_center_frac,
            jitter=stripe_jitter,
        )

        if len(windows) < min(6, stripe_max_patches):
            ww = min(max(win_w, 160), W)
            hh = min(max(stripe_win_h, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            windows.append((x1, y1, x1 + ww, y1 + hh, int(max(ww, hh)), 1, 5000))

        if qmask_rs is not None and min_mask_cover > 0:
            filtered = []
            for (x1, y1, x2, y2, win, ptype, pos) in windows:
                cover = float(qmask_rs[y1:y2, x1:x2].mean()) / 255.0
                if cover >= min_mask_cover:
                    filtered.append((x1, y1, x2, y2, win, ptype, pos))
            if len(filtered) >= 16:
                windows = filtered

        for (x1, y1, x2, y2, *_rest) in windows:
            patches.append(qimg[y1:y2, x1:x2])

    else:
        patches_with_xy = extract_patches_grid(
            qimg,
            patch_sizes=(256, 384, 512, 768),
            stride_ratio=0.5,
            max_patches=60,
            roi_xyxy=None,
            return_xyxy=True
        )
        for patch, (x1, y1, x2, y2) in patches_with_xy:
            if qmask_rs is not None and min_mask_cover > 0:
                cover = float(qmask_rs[y1:y2, x1:x2].mean()) / 255.0
                if cover < min_mask_cover:
                    continue
            patches.append(patch)

    if len(patches) == 0:
        patches = [qimg]

    tensors = []
    for p in patches:
        p224 = cv2.resize(p, (224, 224), interpolation=cv2.INTER_LINEAR)
        if to_rgb:
            p224 = cv2.cvtColor(p224, cv2.COLOR_BGR2RGB)
        x = (p224.astype(np.float32) - mean) / std
        tensors.append(torch.from_numpy(x.transpose(2, 0, 1)))

    feats_chunks = []
    for st in range(0, len(tensors), batch_size):
        bt = torch.stack(tensors[st:st + batch_size]).to(device)
        fv = extract_backbone_last(model, bt)
        feats_chunks.append(fv.detach().cpu())

    feats = torch.cat(feats_chunks, dim=0).numpy().astype("float32")
    return feats, len(patches), bool(is_stripe), (H, W)
