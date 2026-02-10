# hybrid_wrapper.py
# -*- coding: utf-8 -*-
import os
import numpy as np

# 你把原脚本保存为 hybrid_core.py（同目录）
import hybrid_core as core


def run_pipeline(query_img_path: str, out_dir: str = None, topk: int = None):
    """
    UI 调用入口：给一张 query 图，跑完整 pipeline。
    返回：
      qimg_bgr: 裁剪后的 query（BGR, np.ndarray）
      result_grid_path: 拼图图片路径（result_grid1.png）
      top: [(img_id, score), ...] topk 列表
      debug: dict
    """
    if out_dir is not None:
        core.OUT_DIR = out_dir
    if topk is not None:
        core.TOPK = int(topk)

    core.QUERY_IMG = query_img_path
    core.ensure_dir(core.OUT_DIR)

    # ---- load index & model
    g_index = core.faiss.read_index(core.GLOBAL_INDEX)
    p_index_s = core.faiss.read_index(core.PATCH_STRIPE_INDEX)
    p_index_g = core.faiss.read_index(core.PATCH_GRID_INDEX)

    core.set_faiss_nprobe(g_index, 64)
    core.set_faiss_nprobe(p_index_s, 64)
    core.set_faiss_nprobe(p_index_g, 64)

    img_paths = np.load(core.GLOBAL_META, allow_pickle=True)
    patch_meta_s = np.load(core.PATCH_STRIPE_META, allow_pickle=True)
    patch_meta_g = np.load(core.PATCH_GRID_META, allow_pickle=True)

    model, mean, std, to_rgb = core.build_model(core.CONFIG, core.CKPT)

    # ---- load query
    qimg0 = core.imread_unicode(core.QUERY_IMG)
    if qimg0 is None:
        raise RuntimeError(f"Failed to read query image: {core.QUERY_IMG}")

    # ---- seg & crop
    qimg, qmask, qimg_raw = core.crop_by_mmdet_mask_final(
        qimg0,
        score_thr=core.SEG_SCORE_THR,
        use_classes=core.SEG_USE_CLASSES,
        merge_all=True,
        do_rectify=False,
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=None
    )

    # ---- head feature (query)
    q_head_img = core.make_head_view(qimg_raw, prefer_gray=True)
    q_head = core.stripe_grid_head_v21(q_head_img)

    # ---- global feature
    qvec = core.get_query_global_feat(model, mean, std, to_rgb, qimg)

    # ---- patch features
    q_patch_vecs, n_qpatch, is_stripe2, hw = core.get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg,
        qmask=qmask,
        long_edge=core.STRIPE_LONG_EDGE,
        stripe_ar_thr=core.STRIPE_AR_THR,
        stripe_win_h=core.STRIPE_WIN_H,
        stripe_stride=core.STRIPE_STRIDE,
        stripe_max_patches=core.STRIPE_MAX_PATCHES,
        min_mask_cover=0.0,
        batch_size=64
    )
    is_vertical_stripe = bool(is_stripe2)

    # ---- global search
    _, gids = g_index.search(qvec, core.TOPG)
    global_rank = core.clean_rank(gids[0].tolist())

    # ---- patch search
    if is_stripe2:
        D, I = p_index_s.search(q_patch_vecs, core.PATCH_TOPK_PER_QPATCH)
        S = core.faiss_scores_from_D(p_index_s, D.astype(np.float32))
        patch_rank = core.aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            patch_meta_s,
            top_images=core.TOP_PATCH_IMAGES,
            tau=0.15
        )
    else:
        D, I = p_index_g.search(q_patch_vecs, core.PATCH_TOPK_PER_QPATCH)
        S = core.faiss_scores_from_D(p_index_g, D.astype(np.float32))
        patch_rank = core.aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            patch_meta_g,
            top_images=core.TOP_PATCH_IMAGES,
            tau=0.15
        )

    # ---- RRF fusion weights
    rrf_g = core.rank_to_rrf_score(global_rank, k=core.RRF_K)
    rrf_p = core.rank_to_rrf_score(patch_rank,  k=core.RRF_K)

    conf_g = core.global_confidence(global_rank, img_paths, topn=20)
    w_g = 0.6 + 0.35 * conf_g
    w_p = 1.0 - w_g

    if (q_head["stripe_score"] > 0.18 and q_head["ori_peakedness"] > 2.8) or \
       (q_head["grid_score"]   > 0.18 and q_head["ori_peakedness"] > 2.8):
        w_g, w_p = 0.20, 0.80

    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    # ---- candidate pool
    fused = core.rrf_fuse(global_rank, patch_rank, core.RRF_K)
    fused = core.clean_rank(fused)[:core.GEOM_TOPN]

    fused2 = []
    cimg_cache = {}
    for img_id in fused:
        cimg = core.imread_unicode(img_paths[img_id])
        if cimg is None:
            continue
        g0 = core.cand_gate_score(cimg, q_head)
        if g0 > 0:
            fused2.append(img_id)
            cimg_cache[img_id] = cimg

    # ---- geom rerank
    angles = [-10, -5, 0, 5, 10] if is_vertical_stripe else [-30, -15, 0, 15, 30]
    q_desc_list, q_xy_list = [], []
    for ang in angles:
        qimg_r = core.rotate_bgr(qimg, ang)
        qx = core.make_single_tensor_for_rerank(qimg_r, mean, std, to_rgb).to(core.DEVICE)
        q_fm = core.extract_featmap(model, qx, core.FEAT_LEVEL)
        q_desc, q_xy = core.select_query_patches(q_fm)
        q_desc_list.append(q_desc)
        q_xy_list.append(q_xy)

    scored = []
    for img_id in fused2:
        cimg = cimg_cache[img_id]
        cx = core.make_single_tensor_for_rerank(cimg, mean, std, to_rgb).to(core.DEVICE)
        c_fm = core.extract_featmap(model, cx, core.FEAT_LEVEL)
        c_desc, c_xy = core.select_candidate_patches(c_fm)

        geom_best, cnt = 0.0, 0
        for q_desc, q_xy in zip(q_desc_list, q_xy_list):
            s = core.geom_score_compatible(
                q_desc, q_xy, c_desc, c_xy,
                margin=0.012, min_keep=5,
                bin_size=core.BIN_SIZE, topM=core.TOPM, topk_core=core.TOPK_CORE,
                periodic_peak_thr=core.PERIODIC_PEAK_THR,
                periodic_cover_topM_thr=core.PERIODIC_COVER_TOPM_THR,
                periodic_cover_xy_thr=core.PERIODIC_COVER_XY_THR,
                tex_weight=0.85
            )
            if s > 0:
                geom_best += s
                cnt += 1
        if cnt > 0:
            scored.append((img_id, geom_best / cnt))

    result_grid_path = os.path.join(core.OUT_DIR, "result_grid1.png")

    if not scored:
        fallback = fused2[:core.TOPK] if fused2 else fused[:core.TOPK]
        imgs = [core.imread_unicode(img_paths[i]) for i in fallback]
        scores = [1.0 - i / max(1, len(fallback)) for i in range(len(fallback))]
        core.visualize_grid(qimg, imgs, scores, result_grid_path)
        top = list(zip(fallback, scores))
        debug = {"fallback": True, "is_stripe": bool(is_stripe2), "n_qpatch": int(n_qpatch), "hw": hw}
        return qimg, result_grid_path, top, debug

    geom_vals = [s for _, s in scored]
    gmin, gmax = min(geom_vals), max(geom_vals)

    final = []
    for img_id, gs in scored:
        norm_g = (gs - gmin) / (gmax - gmin + 1e-9)
        fs = final_rrf.get(img_id, 0.0) * (1.0 + 0.6 * norm_g)
        final.append((img_id, fs))

    final.sort(key=lambda x: x[1], reverse=True)
    top = final[:core.TOPK]

    imgs = [cimg_cache[i] if i in cimg_cache else core.imread_unicode(img_paths[i]) for i, _ in top]
    scores = [s for _, s in top]
    core.visualize_grid(qimg, imgs, scores, result_grid_path)

    debug = {
        "fallback": False,
        "is_stripe": bool(is_stripe2),
        "n_qpatch": int(n_qpatch),
        "hw": hw,
        "w_g": float(w_g),
        "w_p": float(w_p),
        "conf_g": float(conf_g),
    }
    return qimg, result_grid_path, top, debug
