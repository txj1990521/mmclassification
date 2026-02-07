# service.py
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn

# 直接 import 你的原搜索脚本（不会触发 main）
import zhanlan.ivfflat.search_ivfflat_patch_global_stripe_sort as zp


app = FastAPI(title="Zhanlan Search Service")

# -------- 全局单例（启动时加载一次）--------
g_index = None
p_index = None
img_paths = None
patch_meta = None
model = None
mean = None
std = None
to_rgb = None


@app.on_event("startup")
def startup():
    global g_index, p_index, img_paths, patch_meta
    global model, mean, std, to_rgb

    print("[SERVICE] loading model & faiss...")

    # faiss
    g_index = zp.faiss.read_index(zp.GLOBAL_INDEX)
    p_index = zp.faiss.read_index(zp.PATCH_INDEX)
    zp.set_faiss_nprobe(g_index, 64)
    zp.set_faiss_nprobe(p_index, 64)

    # meta
    img_paths = zp.np.load(zp.GLOBAL_META, allow_pickle=True)
    patch_meta = zp.np.load(zp.PATCH_META, allow_pickle=True)

    # model
    model, mean, std, to_rgb = zp.build_model(zp.CONFIG, zp.CKPT)

    print("[SERVICE] ready")
    print(f"[SERVICE] listening on 192.168.24.49:8000")

@app.post("/search")
async def search(file: UploadFile = File(...)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "invalid image"}

    # ================== 以下完全复用你 main() 的逻辑 ==================

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

    # ---- head feature
    q_head_img = zp.make_head_view(qimg_raw, prefer_gray=True)
    q_head = zp.stripe_grid_head_v21(q_head_img)

    # ---- global + patch feature
    qvec = zp.get_query_global_feat(model, mean, std, to_rgb, qimg)
    q_patch_vecs, n_qpatch, is_stripe2, hw = zp.get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg,
        qmask=qmask,
        long_edge=zp.STRIPE_LONG_EDGE,
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

    # ---- patch search
    D, I = p_index.search(q_patch_vecs, zp.PATCH_TOPK_PER_QPATCH)
    S = zp.faiss_scores_from_D(p_index, D.astype(np.float32))

    if is_stripe2:
        patch_rank = zp.aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            patch_meta,
            top_images=zp.TOP_PATCH_IMAGES
        )
    else:
        patch_rank = zp.aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            patch_meta,
            top_images=zp.TOP_PATCH_IMAGES
        )

    patch_rank = zp.clean_rank(patch_rank)

    # ---- RRF 融合（与你原 main 一致）
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

    # ---- 只取前 TOPK
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
    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="0.0.0.0",  # 监听所有网卡
            port=8000,
            workers=1,  # 非常重要：GPU 场景一定是 1
            log_level="info"
        )


