# service.py
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import uvicorn

# 直接 import 你的原搜索脚本（不会触发 main）
import zhanlan.ivfflat.search_ivfflat_patch_global_stripe_sort as zp


# ---------- 全局单例（启动时加载一次） ----------
STATE = {
    "g_index": None,
    "p_index_s": None,
    "p_index_g": None,
    "img_paths": None,
    "patch_meta_s": None,
    "patch_meta_g": None,
    "model": None,
    "mean": None,
    "std": None,
    "to_rgb": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVICE] loading model & faiss...")

    # global index
    STATE["g_index"] = zp.faiss.read_index(zp.GLOBAL_INDEX)
    zp.set_faiss_nprobe(STATE["g_index"], 64)

    # patch indexes (stripe / grid)
    STATE["p_index_s"] = zp.faiss.read_index(zp.PATCH_STRIPE_INDEX)
    STATE["p_index_g"] = zp.faiss.read_index(zp.PATCH_GRID_INDEX)
    zp.set_faiss_nprobe(STATE["p_index_s"], 64)
    zp.set_faiss_nprobe(STATE["p_index_g"], 64)

    # meta
    STATE["img_paths"] = zp.np.load(zp.GLOBAL_META, allow_pickle=True)
    STATE["patch_meta_s"] = zp.np.load(zp.PATCH_STRIPE_META, allow_pickle=True)
    STATE["patch_meta_g"] = zp.np.load(zp.PATCH_GRID_META, allow_pickle=True)

    # model
    model, mean, std, to_rgb = zp.build_model(zp.CONFIG, zp.CKPT)
    STATE["model"] = model
    STATE["mean"] = mean
    STATE["std"] = std
    STATE["to_rgb"] = to_rgb

    print("[SERVICE] ready")
    yield
    print("[SERVICE] shutdown")


app = FastAPI(title="Zhanlan Search Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(file: UploadFile = File(...)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}

    model = STATE["model"]
    mean = STATE["mean"]
    std = STATE["std"]
    to_rgb = STATE["to_rgb"]
    g_index = STATE["g_index"]
    p_index_s = STATE["p_index_s"]
    p_index_g = STATE["p_index_g"]
    img_paths = STATE["img_paths"]
    patch_meta_s = STATE["patch_meta_s"]
    patch_meta_g = STATE["patch_meta_g"]

    # ---- seg & crop
    qimg, qmask, qimg_raw = zp.crop_by_mmdet_mask_final(
        img,
        score_thr=zp.SEG_SCORE_THR,
        use_classes=zp.SEG_USE_CLASSES,
        merge_all=True,
        do_rectify=False,
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=None
    )

    # ---- head feature (query)
    q_head_img = zp.make_head_view(qimg_raw, prefer_gray=True)
    q_head = zp.stripe_grid_head_v21(q_head_img)

    # ---- global feature (已是 multi-view 版本)
    qvec = zp.get_query_global_feat(model, mean, std, to_rgb, qimg)

    # ---- patch feats + is_stripe
    q_patch_vecs, n_qpatch, is_stripe2, hw = zp.get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg,
        qmask=qmask,
        long_edge=zp.STRIPE_LONG_EDGE,   # 你现在从 shared import 的常量
        stripe_ar_thr=zp.STRIPE_AR_THR,
        stripe_win_h=zp.STRIPE_WIN_H,
        stripe_stride=zp.STRIPE_STRIDE,
        stripe_max_patches=zp.STRIPE_MAX_PATCHES,
        min_mask_cover=0.0,
        batch_size=64
    )

    # ---- global search
    _, gids = g_index.search(qvec, zp.TOPG)
    global_rank = zp.clean_rank(gids[0].tolist())

    # ---- patch search：按 is_stripe 选不同索引/元数据
    if is_stripe2:
        D, I = p_index_s.search(q_patch_vecs, zp.PATCH_TOPK_PER_QPATCH)
        S = zp.faiss_scores_from_D(p_index_s, D.astype(np.float32))
        patch_rank = zp.aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            patch_meta_s,
            top_images=zp.TOP_PATCH_IMAGES,
            tau=0.15
        )
    else:
        D, I = p_index_g.search(q_patch_vecs, zp.PATCH_TOPK_PER_QPATCH)
        S = zp.faiss_scores_from_D(p_index_g, D.astype(np.float32))
        patch_rank = zp.aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            patch_meta_g,
            top_images=zp.TOP_PATCH_IMAGES,
            tau=0.15
        )

    patch_rank = zp.clean_rank(patch_rank)

    # ---- RRF 融合（沿用你的逻辑）
    rrf_g = zp.rank_to_rrf_score(global_rank, k=zp.RRF_K)
    rrf_p = zp.rank_to_rrf_score(patch_rank,  k=zp.RRF_K)

    conf_g = zp.global_confidence(global_rank, img_paths, topn=20)
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

    # ---- candidates（你服务里目前只返回 TopK，不做 geom rerank）
    candidates = zp.rrf_fuse(global_rank, patch_rank, zp.RRF_K)
    candidates = zp.clean_rank(candidates)

    results = []
    for rank, img_id in enumerate(candidates[:zp.TOPK], 1):
        results.append({
            "rank": rank,
            "img_id": int(img_id),
            "score": float(final_rrf.get(img_id, 0.0))
        })

    return {
        "is_stripe": bool(is_stripe2),
        "n_qpatch": int(n_qpatch),
        "results": results
    }


if __name__ == "__main__":
    # 你现在机器真实 IP 是 192.168.24.49
    uvicorn.run(app, host="0.0.0.0", port=8000)
