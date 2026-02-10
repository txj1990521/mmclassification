# zhanlan/ivfflat/zhanlan_retrieval/global_views.py
import hashlib
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .geom_utils import resize_long_edge, resize_short_edge, rotate_bound, center_crop, random_crop
from .model_build import extract_backbone_last


def seed_from_image(img_bgr: np.ndarray, base: int = 999) -> int:
    if img_bgr is None or img_bgr.size == 0:
        return base & 0x7fffffff
    h = hashlib.md5(img_bgr.tobytes()).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff


def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))


def to_tensor_from_rgb(img_rgb_crop: np.ndarray, mean, std):
    x = img_rgb_crop.astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


@torch.no_grad()
def make_views_for_global(img_bgr: np.ndarray, mean, std, to_rgb: bool,
                          resize_short: int, crop_size: int, view_plan, views_per_image: int):
    if to_rgb:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_bgr[:, :, ::-1].copy()
    img_rgb = resize_short_edge(img_rgb, resize_short)

    views = []
    for deg, n_center, n_rand in view_plan:
        rot = rotate_bound(img_rgb, deg)
        for _ in range(n_center):
            views.append(to_tensor_from_rgb(center_crop(rot, crop_size), mean, std))
        for _ in range(n_rand):
            views.append(to_tensor_from_rgb(random_crop(rot, crop_size), mean, std))
    return views[:views_per_image]


@torch.no_grad()
def aggregate_views_to_one(feats_view: torch.Tensor):
    agg = feats_view.mean(dim=0)
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg


@torch.no_grad()
def get_query_global_feat(model, mean, std, to_rgb, img_bgr,
                          device: str,
                          stripe_long_edge: int,
                          resize_short: int,
                          crop_size: int,
                          view_plan,
                          views_per_image: int):
    seed_src = resize_long_edge(img_bgr, max_long=stripe_long_edge)
    random.seed(seed_from_image(seed_src, base=0))

    views = make_views_for_global(seed_src, mean, std, to_rgb,
                                  resize_short=resize_short,
                                  crop_size=crop_size,
                                  view_plan=view_plan,
                                  views_per_image=views_per_image)
    if len(views) == 0:
        raise RuntimeError("No global views generated for query.")

    bt = torch.stack(views, dim=0).to(device)
    feats_v = extract_backbone_last(model, bt)
    q = aggregate_views_to_one(feats_v)
    return q.unsqueeze(0).detach().cpu().numpy().astype("float32")
