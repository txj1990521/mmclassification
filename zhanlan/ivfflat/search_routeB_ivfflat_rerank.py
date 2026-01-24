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
OUT_VIZ   = r"D:\zhanlan\topk_viz_rmac\topk_viz_rmac23.jpg"

# =========================
# Runtime
# =========================
TOPK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE = 320

# ---------- Coarse (Route-B aligned) ----------
COARSE_K = 300
# IVF 搜索的关键：nprobe 越大越准，越慢；10万常用 8~16；百万常用 16~64
NPROBE = 16

VIEWS_PER_QUERY = 12
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

KEEP_PATCHES = 512
BORDER = 0.15

MARGIN = 0.02
MIN_KEEP = 8

BIN_SIZE = 4.0
TOPM = 6
TOPK_CORE = 64

# ---------- Fusion ----------
# 这里用你稳定版 gating fuse
GATE_T = 0.70
GAMMA = 0.85

# 周期纹理判别
PERIODIC_PEAK_THR = 0.22
PERIODIC_COVER_THR = 0.55

# coarse 保底
COARSE_TOPN = 60
MIN_KEEP_IN_TOPK = 6

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
def extract_global_feat_for_coarse(model, batch_tensor: torch.Tensor):
    feat_map = model.backbone(batch_tensor)
    if isinstance(feat_map, (tuple, list)):
        feat_map = feat_map[-1]
    feat = feat_map.mean(dim=(2,3))
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
def select_top_patches_with_xy(feat_map, keep=512, border=0.15):
    fm = feat_map[0]
    C, H, W = fm.shape
    energy = fm.pow(2).sum(dim=0)

    y1 = int(H * border); y2 = int(H * (1 - border))
    x1 = int(W * border); x2 = int(W * (1 - border))

    mask = torch.zeros_like(energy, dtype=torch.bool)
    mask[y1:y2, x1:x2] = True

    idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
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

@torch.no_grad()
def select_top_patches_with_xy_batch(feat_map, keep=512, border=0.15):
    B, C, H, W = feat_map.shape
    out = []

    y1 = int(H * border); y2 = int(H * (1 - border))
    x1 = int(W * border); x2 = int(W * (1 - border))

    for b in range(B):
        fm = feat_map[b]
        energy = fm.pow(2).sum(dim=0)

        mask = torch.zeros_like(energy, dtype=torch.bool)
        mask[y1:y2, x1:x2] = True

        idx_all = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
        k = min(keep, idx_all.numel())

        vals = energy.flatten()[idx_all]
        top_local = torch.topk(vals, k=k, largest=True).indices
        idx = idx_all[top_local]

        patches = fm.flatten(1).t()[idx]
        patches = F.normalize(patches, p=2, dim=1)

        ys = (idx // W).float()
        xs = (idx %  W).float()
        xy = torch.stack([xs, ys], dim=1)

        out.append((patches, xy))

    return out

@torch.no_grad()
def geom_score_adaptive(q_desc, q_xy, c_desc, c_xy,
                        margin=0.02, min_keep=8,
                        bin_size=4.0, topM=6, topk_core=64,
                        periodic_peak_thr=0.22, periodic_cover_thr=0.55):
    sim = q_desc @ c_desc.t()

    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    top1 = topv[:, 0]
    top2 = topv[:, 1]
    good = (top1 - top2) > margin
    Ng = int(good.sum().item())
    if Ng < min_keep:
        return 0.0

    mi = topi[good, 0]
    d = c_xy[mi] - q_xy[good]

    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(Ng)
    m = min(topM, cnt.numel())
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(Ng)

    k = min(topk_core, Ng)
    core = float(torch.topk(top1[good], k=k, largest=True).values.mean().item())

    is_periodic = (peak_ratio < periodic_peak_thr) and (cover_topM > periodic_cover_thr)
    return float(core * (cover_topM if is_periodic else peak_ratio))

# =========================
# fusion helpers
# =========================
def _minmax(x: np.ndarray):
    x = x.astype(np.float32)
    mn = float(x.min()); mx = float(x.max())
    return (x - mn) / (mx - mn + 1e-9)

def fuse_scores(coarse01, rerank01, T=0.70, gamma=0.85, wC=0.70, wR=0.30):
    r = float(rerank01) ** gamma
    c = float(coarse01)
    if c < T:
        return 0.90 * c + 0.10 * r
    return wC * c + wR * r

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
    # IVF：设置 nprobe
    if hasattr(index, "nprobe"):
        index.nprobe = NPROBE
    print(f"[INFO] Loaded index ntotal={index.ntotal}, nprobe={getattr(index,'nprobe','-')}")

    kept_paths = np.load(OUT_META, allow_pickle=True)
    print(f"[INFO] Loaded meta={len(kept_paths)}")

    model, mean, std, to_rgb = build_model(CONFIG, CKPT, DEVICE)

    qimg = imread_unicode(QUERY_IMG)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {QUERY_IMG}")

    # ---------- Stage-1: coarse (Route-B aligned) ----------
    views = make_query_views_routeB(qimg, mean, std, to_rgb=to_rgb)
    print(f"[INFO] coarse views(RouteB) = {len(views)}")

    bt = torch.stack(views, dim=0).to(DEVICE)
    feats_v = extract_global_feat_for_coarse(model, bt)   # (V,D) L2
    qvec = aggregate_views_to_one(feats_v).unsqueeze(0).cpu().numpy().astype("float32")

    scores, ids = index.search(qvec, COARSE_K)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    coarse_ids = [i for i in ids if i >= 0]
    coarse_raw = np.array([float(s) for i, s in zip(ids, scores) if i >= 0], dtype=np.float32)
    coarse_paths = [kept_paths[i] for i in coarse_ids]
    print(f"[INFO] coarse candidates = {len(coarse_ids)}")

    # ---------- Stage-2: rerank ----------
    qx = make_single_tensor_for_rerank(qimg, mean, std, to_rgb=to_rgb).to(DEVICE)
    q_fm = extract_featmap(model, qx, FEAT_LEVEL)
    q_desc, q_xy = select_top_patches_with_xy(q_fm, keep=KEEP_PATCHES, border=BORDER)

    cand_tensors = []
    valid_paths = []
    valid_coarse_raw = []

    for p, cs in zip(coarse_paths, coarse_raw.tolist()):
        img = imread_unicode(str(p))
        if img is None:
            continue
        cand_tensors.append(make_single_tensor_for_rerank(img, mean, std, to_rgb=to_rgb))
        valid_paths.append(str(p))
        valid_coarse_raw.append(float(cs))

    if not cand_tensors:
        raise RuntimeError("All coarse candidate images failed to read.")

    rerank_raw = []
    bs2 = 8
    for st in range(0, len(cand_tensors), bs2):
        bt2 = torch.cat(cand_tensors[st:st + bs2], dim=0).to(DEVICE)
        fm = extract_featmap(model, bt2, FEAT_LEVEL)
        c_list = select_top_patches_with_xy_batch(fm, keep=KEEP_PATCHES, border=BORDER)
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

    cs01 = _minmax(coarse_raw2)
    rs01 = _minmax(rerank_raw2)

    final = np.array([fuse_scores(cs01[i], rs01[i], T=GATE_T, gamma=GAMMA) for i in range(len(cs01))],
                     dtype=np.float32)

    order_all = np.argsort(-final)

    order_top = enforce_coarse_floor(
        order_all=order_all,
        coarse_raw=coarse_raw2,
        final_scores=final,
        topk=TOPK,
        coarse_topN=COARSE_TOPN,
        min_keep=MIN_KEEP_IN_TOPK
    )

    top_paths = [valid_paths[i] for i in order_top]
    top_scores = [float(final[i]) for i in order_top]

    print("\n===== TOPK (FINAL fused) =====")
    for r, (p, s) in enumerate(zip(top_paths, top_scores), 1):
        print(f"{r:02d}  score={s:.4f}  {p}")

    out = make_topk_viz(QUERY_IMG, top_paths, top_scores, OUT_VIZ, tile=TILE)
    print(f"\n[VIZ] saved -> {out}")

    for i in order_top:
        print(f"[DBG] {valid_paths[i]}  coarse_raw={coarse_raw2[i]:.4f}  coarse01={cs01[i]:.4f}  rerank01={rs01[i]:.4f}  final={final[i]:.4f}")

if __name__ == "__main__":
    main()
