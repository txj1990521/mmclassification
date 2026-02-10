# zhanlan/ivfflat/zhanlan_retrieval/featmap_patches.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .geom_utils import pad_to_square
from .model_build import extract_featmap
from .io_utils import imread_unicode


@torch.no_grad()
def make_single_tensor_for_rerank(img_bgr: np.ndarray, mean, std, to_rgb: bool, rmac_input_size: int):
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()

    img_rgb = pad_to_square(img_rgb)
    img_rgb = cv2.resize(img_rgb, (rmac_input_size, rmac_input_size), interpolation=cv2.INTER_LINEAR)

    x = img_rgb.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


def energy_roi_box(fm_1chw: torch.Tensor, frac=0.18):
    e = fm_1chw.pow(2).sum(dim=0)
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
def select_top_patches_with_xy(feat_map_1bchw: torch.Tensor, keep=256, border=0.05, roi_fbox=None):
    fm = feat_map_1bchw[0]
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)

    y1b = int(H * border); y2b = int(H * (1 - border))
    x1b = int(W * border); x2b = int(W * (1 - border))
    mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
    mask[y1b:y2b, x1b:x2b] = True

    if roi_fbox is not None:
        rx1, ry1, rx2, ry2 = roi_fbox
        rx1 = max(0, min(W-1, int(rx1))); ry1 = max(0, min(H-1, int(ry1)))
        rx2 = max(1, min(W,   int(rx2))); ry2 = max(1, min(H,   int(ry2)))
        if rx2 > rx1 and ry2 > ry1:
            roi_mask = torch.zeros((H, W), device=energy.device, dtype=torch.bool)
            roi_mask[ry1:ry2, rx1:rx2] = True
            mask = mask & roi_mask

    idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    if idx_all.numel() == 0:
        idx_all = torch.arange(H * W, device=energy.device)

    k = min(int(keep), int(idx_all.numel()))
    vals = energy.flatten()[idx_all]
    top_local = torch.topk(vals, k=k, largest=True).indices
    idx = idx_all[top_local]

    patches = fm.flatten(1).t()[idx]
    patches = F.normalize(patches, p=2, dim=1)

    ys = (idx // W).float()
    xs = (idx % W).float()
    xs = xs / max(1.0, float(W - 1))
    ys = ys / max(1.0, float(H - 1))
    xy = torch.stack([xs, ys], dim=1)
    return patches, xy


@torch.no_grad()
def select_query_patches(q_fm_1bchw: torch.Tensor,
                         keep=256, border=0.05,
                         roi_frac=0.18, roi_ratio=0.50):
    q_roi = energy_roi_box(q_fm_1bchw[0], frac=roi_frac)
    k_roi = int(round(keep * roi_ratio))
    k_full = max(1, keep - k_roi)

    desc_list, xy_list = [], []
    if q_roi is not None and k_roi >= 4:
        d1, x1 = select_top_patches_with_xy(q_fm_1bchw, keep=k_roi, border=border, roi_fbox=q_roi)
        if d1 is not None and d1.shape[0] >= 4:
            desc_list.append(d1); xy_list.append(x1)

    d2, x2 = select_top_patches_with_xy(q_fm_1bchw, keep=k_full, border=border, roi_fbox=None)
    desc_list.append(d2); xy_list.append(x2)

    return torch.cat(desc_list, dim=0), torch.cat(xy_list, dim=0)


@torch.no_grad()
def select_candidate_patches(c_fm_1bchw: torch.Tensor,
                             keep=256, border=0.05,
                             roi_frac=0.18):
    c_roi = energy_roi_box(c_fm_1bchw[0], frac=roi_frac)
    d, x = select_top_patches_with_xy(c_fm_1bchw, keep=keep, border=border, roi_fbox=c_roi)
    return d, x


@torch.no_grad()
def batch_candidate_desc_xy(model, img_ids, img_cache, img_paths, mean, std, to_rgb,
                            device: str, batch_size=32, feat_level=-2, rmac_input_size=512):
    out = {}
    buf_ids = []
    buf_tensors = []

    for img_id in img_ids:
        cimg = img_cache.get(img_id, None)
        if cimg is None:
            cimg = imread_unicode(img_paths[img_id])
            if cimg is None:
                continue
            img_cache[img_id] = cimg

        t = make_single_tensor_for_rerank(cimg, mean, std, to_rgb, rmac_input_size=rmac_input_size)
        buf_ids.append(img_id)
        buf_tensors.append(t)

        if len(buf_ids) >= batch_size:
            bt = torch.cat(buf_tensors, dim=0).to(device, non_blocking=True)
            with autocast(enabled=str(device).startswith("cuda")):
                fm = extract_featmap(model, bt, feat_level)
            for i, _id in enumerate(buf_ids):
                d, x = select_candidate_patches(fm[i:i + 1])
                out[_id] = (d.float(), x.float())
            buf_ids, buf_tensors = [], []

    if buf_ids:
        bt = torch.cat(buf_tensors, dim=0).to(device, non_blocking=True)
        with autocast(enabled=str(device).startswith("cuda")):
            fm = extract_featmap(model, bt, feat_level)
        for i, _id in enumerate(buf_ids):
            d, x = select_candidate_patches(fm[i:i + 1])
            out[_id] = (d.float(), x.float())

    return out
