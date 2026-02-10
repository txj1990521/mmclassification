# zhanlan/ivfflat/zhanlan_retrieval/pipeline_core.py
"""
pipeline_core.py
----------------
只做“计算逻辑”，不做：
- index/model/yolo 的加载
- 任何文件 IO（读写）
- print/log

输入：
- qimg_bgr: np.ndarray
- cfg: RuntimeConfig
- assets: RetrievalAssets
- topk: int

输出：
- List[int] image indices
"""

from __future__ import annotations

from typing import List
import numpy as np

from .config import RuntimeConfig
from .assets import RetrievalAssets

from .io_utils import imread_unicode
from .faiss_utils import faiss_scores_from_D

from .global_views import get_query_global_feat
from .patch_query import get_query_patch_feats_unified

from .rank_fusion import clean_rank, rank_to_rrf_score, rrf_fuse, global_confidence
from .head_features import make_head_view, stripe_grid_head_v21
from .gating import cand_gate_score

from .cropper import crop_by_yolo_mask_final

from .geom_utils import rotate_bgr
from .model_build import extract_featmap
from .featmap_patches import (
    make_single_tensor_for_rerank,
    select_query_patches,
    batch_candidate_desc_xy,
)
from .geom_rerank import geom_score_compatible


def _aggregate_patch_hits(patch_ids, patch_scores, patch_meta, top_images=4000, topM=8, tau=0.15):
    img_scores = {}
    for pid, s in zip(patch_ids, patch_scores):
        if pid < 0:
            continue
        img_id = int(patch_meta[int(pid), 0])
        img_scores.setdefault(img_id, []).append(float(s))

    fused = []
    for img_id, ss in img_scores.items():
        ss.sort(reverse=True)
        ss = ss[:topM]
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        score = m + max(tau, 1e-6) * np.log(v + 1e-9)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]


def _aggregate_patch_hits_stripe(
    patch_ids, patch_scores, patch_meta,
    top_images=4000,
    topM=10,
    tau=0.15,
    pos_bin=500,
    min_cover_bins=3,
    w_cover=0.12,
    w_cont=0.10,
    only_ptype=1
):
    meta = patch_meta
    img_hits = {}
    for pid, s in zip(patch_ids, patch_scores):
        if pid < 0:
            continue
        pid = int(pid)
        img_id = int(meta[pid, 0])
        ptype = int(meta[pid, 6])
        if only_ptype is not None and ptype != int(only_ptype):
            continue
        pos = int(meta[pid, 7])
        b = int(pos // pos_bin)
        img_hits.setdefault(img_id, []).append((float(s), b))

    fused = []
    for img_id, sb in img_hits.items():
        sb.sort(key=lambda x: x[0], reverse=True)
        sb = sb[:topM]
        ss = [x[0] for x in sb]
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        base = m + max(tau, 1e-6) * np.log(v + 1e-9)

        bins = [x[1] for x in sb]
        ub = sorted(set(bins))
        cover = min(1.0, len(ub) / 8.0)

        longest = 1
        cur = 1
        for i in range(1, len(ub)):
            if ub[i] == ub[i - 1] + 1:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 1
        cont = longest / max(1, len(ub))

        if len(ub) < min_cover_bins:
            base *= 0.72

        score = base * (1.0 + w_cover * cover + w_cont * cont)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]


def search_indices(qimg_bgr, *, cfg: RuntimeConfig, assets: RetrievalAssets, topk: int) -> List[int]:
    # 0) 基础输入检查
    if qimg_bgr is None:
        raise ValueError("qimg_bgr is None")

    # 1) YOLO crop（复用已加载 yolo）
    qimg, qmask, qimg_raw = crop_by_yolo_mask_final(
        assets.yolo,
        qimg_bgr,
        score_thr=cfg.seg_score_thr,
        use_classes=cfg.seg_use_classes,
        merge_all=True,
        do_rectify=False,
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=None,
        yolo_imgsz=cfg.yolo_imgsz,
        yolo_iou=cfg.yolo_iou,
        yolo_retina_masks=cfg.yolo_retina_masks,
        yolo_device=cfg.yolo_device,
        yolo_max_det=cfg.yolo_max_det,
    )

    # 2) head feature（用于 gate/权重策略）
    q_head_img = make_head_view(qimg_raw, prefer_gray=True)
    q_head = stripe_grid_head_v21(q_head_img)

    # 3) global retrieval
    qvec = get_query_global_feat(
        assets.model, assets.mean, assets.std, assets.to_rgb, qimg,
        device=cfg.device,
        stripe_long_edge=cfg.stripe_long_edge,
        resize_short=cfg.resize_short,
        crop_size=cfg.crop_size,
        view_plan=cfg.view_plan,
        views_per_image=cfg.views_per_image
    )
    _, gids = assets.g_index.search(qvec, cfg.topg)
    global_rank = clean_rank(gids[0].tolist())

    # 4) patch retrieval
    q_patch_vecs, _, is_stripe, _ = get_query_patch_feats_unified(
        assets.model, assets.mean, assets.std, assets.to_rgb, qimg,
        qmask=qmask,
        device=cfg.device,
        long_edge=cfg.stripe_long_edge,
        stripe_ar_thr=cfg.stripe_ar_thr,
        stripe_win_h=cfg.stripe_win_h,
        stripe_stride=cfg.stripe_stride,
        stripe_max_patches=cfg.stripe_max_patches,
        stripe_center_frac=cfg.stripe_center_frac,
        stripe_jitter=cfg.stripe_jitter,
        stripe_win_w_frac=cfg.stripe_win_w_frac,
        stripe_win_w_min=cfg.stripe_win_w_min,
        stripe_win_w_max=cfg.stripe_win_w_max,
        min_mask_cover=0.0,
        batch_size=64
    )

    if is_stripe:
        D, I = assets.p_index_s.search(q_patch_vecs, cfg.patch_topk_per_qpatch)
        S = faiss_scores_from_D(assets.p_index_s, D.astype(np.float32))
        patch_rank = _aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            assets.patch_meta_s,
            top_images=cfg.top_patch_images,
            tau=0.15
        )
    else:
        D, I = assets.p_index_g.search(q_patch_vecs, cfg.patch_topk_per_qpatch)
        S = faiss_scores_from_D(assets.p_index_g, D.astype(np.float32))
        patch_rank = _aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            assets.patch_meta_g,
            top_images=cfg.top_patch_images,
            tau=0.15
        )

    # 5) weighted RRF
    rrf_g = rank_to_rrf_score(global_rank, k=cfg.rrf_k)
    rrf_p = rank_to_rrf_score(patch_rank,  k=cfg.rrf_k)

    conf_g = global_confidence(global_rank, assets.img_paths, topn=20)
    w_g = 0.6 + 0.35 * conf_g
    w_p = 1.0 - w_g

    if (q_head.get("stripe_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8) or \
       (q_head.get("grid_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8):
        w_g, w_p = 0.20, 0.80

    # 额外：patch 强但 global 弱的修正（保留你原逻辑）
    def _rank_pos(lst, x):
        try:
            return lst.index(x) + 1
        except Exception:
            return None

    try:
        top_patch_ids = patch_rank[:10]
        gp = []
        for pid in top_patch_ids:
            p = _rank_pos(global_rank, pid)
            if p is not None:
                gp.append(p)
        if len(gp) >= 5:
            gp_sorted = sorted(gp)
            median_gp = gp_sorted[len(gp_sorted) // 2]
            if median_gp >= 120:
                w_g, w_p = 0.15, 0.85
        if conf_g < 0.25:
            w_g, w_p = min(w_g, 0.20), max(w_p, 0.80)
    except Exception:
        pass

    w_g = float(np.clip(w_g, 0.05, 0.95))
    w_p = float(np.clip(1.0 - w_g, 0.05, 0.95))

    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    # 6) candidate pool
    cand = clean_rank(rrf_fuse(global_rank, patch_rank, cfg.rrf_k))
    cand.sort(key=lambda i: final_rrf.get(i, 0.0), reverse=True)
    fused = cand[:cfg.geom_topn]

    # 7) head gate
    fused2, cimg_cache = [], {}
    for img_id in fused:
        cimg = imread_unicode(assets.img_paths[img_id])
        if cimg is None:
            continue
        if cand_gate_score(cimg, q_head) > 0:
            fused2.append(img_id)
            cimg_cache[img_id] = cimg

    # 8) geom rerank
    angles = [-10, -5, 0, 5, 10] if is_stripe else [-30, -15, 0, 15, 30]

    q_desc_list, q_xy_list = [], []
    for ang in angles:
        qimg_r = rotate_bgr(qimg, ang)
        qx = make_single_tensor_for_rerank(
            qimg_r, assets.mean, assets.std, assets.to_rgb,
            rmac_input_size=cfg.rmac_input_size
        ).to(cfg.device)
        q_fm = extract_featmap(assets.model, qx, cfg.feat_level)
        q_desc, q_xy = select_query_patches(
            q_fm,
            keep=cfg.keep_patches,
            border=cfg.border,
            roi_frac=cfg.q_roi_frac,
            roi_ratio=cfg.q_roi_ratio
        )
        q_desc_list.append(q_desc.float())
        q_xy_list.append(q_xy.float())

    cand_desc_xy = batch_candidate_desc_xy(
        assets.model, fused2, cimg_cache, assets.img_paths,
        assets.mean, assets.std, assets.to_rgb,
        device=cfg.device, batch_size=32, feat_level=cfg.feat_level,
        rmac_input_size=cfg.rmac_input_size
    )

    scored = []
    for img_id in fused2:
        if img_id not in cand_desc_xy:
            continue
        c_desc, c_xy = cand_desc_xy[img_id]

        geom_best, cnt = 0.0, 0
        for q_desc, q_xy in zip(q_desc_list, q_xy_list):
            s = geom_score_compatible(
                q_desc, q_xy, c_desc, c_xy,
                margin=0.012, min_keep=5,
                bin_size=cfg.bin_size, topM=cfg.topm, topk_core=cfg.topk_core,
                periodic_peak_thr=cfg.periodic_peak_thr,
                periodic_cover_topM_thr=cfg.periodic_cover_topm_thr,
                periodic_cover_xy_thr=cfg.periodic_cover_xy_thr,
                tex_weight=0.85
            )
            if s > 0:
                geom_best += s
                cnt += 1
        if cnt > 0:
            scored.append((img_id, geom_best / cnt))

    # fallback：geom 全失败时仍返回可用 indices
    if not scored:
        fallback = fused2[:topk] if fused2 else fused[:topk]
        return [int(i) for i in fallback]

    geom_vals = [s for _, s in scored]
    gmin, gmax = min(geom_vals), max(geom_vals)

    final = []
    for img_id, gs in scored:
        norm_g = (gs - gmin) / (gmax - gmin + 1e-9)
        fs = final_rrf.get(img_id, 0.0) * (1.0 + 0.6 * norm_g)
        final.append((img_id, fs))

    final.sort(key=lambda x: x[1], reverse=True)
    return [int(img_id) for img_id, _ in final[:topk]]
