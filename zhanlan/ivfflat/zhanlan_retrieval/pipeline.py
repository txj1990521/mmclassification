# zhanlan/ivfflat/zhanlan_retrieval/pipeline.py
"""
Zhanlan Hybrid Retrieval Pipeline
(Global retrieval + Patch retrieval + RRF fusion + HeadGate + GeomRerank)

目标：把原来一坨脚本的 main 流程集中到这里，
并且通过“足够详细的注释”让你只看这个文件就理解整条 pipeline 在干什么。

整体流程（高层）：
1) 读取 FAISS 索引 & 元数据（global + patch stripe + patch grid）
2) 加载 backbone 模型（mmpretrain）
3) 读取 query 图 -> YOLO 分割 -> 生成裁剪图（减少背景干扰）
4) 计算 query 的 head 特征（stripe/grid 强度，用来做 gate & 决定权重策略）
5) 计算 query 的 global embedding，跑 global FAISS 检索得到 global_rank
6) 计算 query 的 patch embedding（stripe 或 grid），跑 patch FAISS 检索得到 patch_rank
7) 用 RRF 做 global+patch 的融合，并根据置信度动态调整 global/patch 权重
8) 候选池：取融合后前 GEOM_TOPN 个候选
9) HeadGate：快速过滤“纹理类型不匹配”的候选，得到 fused2
10) GeomRerank：在深层特征图上做 patch 对齐一致性评分（含 periodic/texture fallback）
11) 用 final_rrf * (1 + 0.6 * norm_geom) 得到最终排序
12) 可视化输出 result_grid1.png
"""

import os
import numpy as np
import faiss
import torch

from .config import RuntimeConfig

# ---- 基础 IO ----
from .io_utils import ensure_dir, imread_unicode

# ---- faiss 小工具 ----
from .faiss_utils import set_faiss_nprobe, faiss_scores_from_D

# ---- 模型相关 ----
from .model_build import build_model, extract_featmap
from .global_views import get_query_global_feat

# ---- patch query 特征提取 ----
from .patch_query import get_query_patch_feats_unified

# ---- rank 与融合 ----
from .rank_fusion import clean_rank, rank_to_rrf_score, rrf_fuse, global_confidence

# ---- head 特征 + gate ----
from .head_features import make_head_view, stripe_grid_head_v21
from .gating import cand_gate_score

# ---- 几何一致性 rerank（featmap patch） ----
from .geom_utils import rotate_bgr
from .featmap_patches import (
    make_single_tensor_for_rerank,
    select_query_patches,
    batch_candidate_desc_xy,
)
from .geom_rerank import geom_score_compatible

# ---- 输出可视化 ----
from .visualize import visualize_grid

# ---- YOLO segmentation + crop ----
from .yolo_seg import YoloSegProvider
from .cropper import crop_by_yolo_mask_final


# =============================================================================
# Patch Hit Aggregation
# =============================================================================
# patch index 的检索结果是：每个 query_patch 会检索出很多 patch_id
# 但我们最终要返回 image_id（图库里的图片）排名。
# 所以需要：把 patch_id -> img_id，并把同一 img_id 的多个命中聚合成一个分数。
# 下面两个函数分别给：
# - grid patch：简单 softmax-aggregation
# - stripe patch：额外利用“条带位置 pos_bin”做覆盖度/连续性加成，减少“蹭到一点纹理”的误召回


def aggregate_patch_hits(patch_ids, patch_scores, patch_meta, top_images=4000, topM=8, tau=0.15):
    """
    grid patch 的聚合（原逻辑保持）：
    - patch_meta[pid, 0] = img_id
    - 对每个 img_id 收集它的 patch_score
    - 取 topM 个最高分做 softmax-like log-sum-exp 融合：
        score = m + tau * log( sum exp((s_i - m)/tau) )
      这样既保留 top1 的强命中，也会给“多次命中”一定奖励。
    """
    meta_is_2d = isinstance(patch_meta, np.ndarray) and patch_meta.ndim == 2

    img_scores = {}
    for pid, s in zip(patch_ids, patch_scores):
        if pid < 0:
            continue
        img_id = int(patch_meta[pid, 0]) if meta_is_2d else int(patch_meta[pid][0])
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


def aggregate_patch_hits_stripe(
        patch_ids, patch_scores, patch_meta,
        top_images=4000,
        topM=10,
        tau=0.15,
        pos_bin=500,          # 10000/500 = 20 bins, stripe 的 pos 是 0..10000
        min_cover_bins=3,     # 覆盖太少的直接惩罚（抑制随机蹭到纹理）
        w_cover=0.12,         # 覆盖加成权重
        w_cont=0.10,          # 连续性加成权重
        only_ptype=1          # 只用 stripe patch（ptype=1）
):
    """
    stripe patch 的聚合增强版（原逻辑保持）：

    patch_meta 约定（你建库时写的）：
      meta[pid, 0] = img_id
      meta[pid, 6] = patch_type（stripe=1 / grid=0 or other）
      meta[pid, 7] = pos（沿长边位置 0..10000）

    聚合：
    1) 同图命中的 patch 做 log-sum-exp 得到 base 分
    2) 统计命中的位置 bin 覆盖（cover）：命中 bin 越多，说明条带对齐更稳定
    3) 统计命中 bin 的最长连续段（cont）：越连续越像真实条带对应
    4) 覆盖 bin < min_cover_bins：认为是“蹭纹理” -> base 直接惩罚
    5) score = base * (1 + w_cover*cover + w_cont*cont)
    """
    meta = patch_meta
    assert isinstance(meta, np.ndarray) and meta.ndim == 2 and meta.shape[1] >= 8

    img_hits = {}   # img_id -> list of (score, pos_bin)
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
        # 1) similarity 融合（softmax / logsumexp）
        sb.sort(key=lambda x: x[0], reverse=True)
        sb = sb[:topM]
        ss = [x[0] for x in sb]
        m = ss[0]
        v = sum(np.exp((x - m) / max(tau, 1e-6)) for x in ss)
        base = m + max(tau, 1e-6) * np.log(v + 1e-9)

        # 2) 位置覆盖度
        bins = [x[1] for x in sb]
        ub = sorted(set(bins))
        cover = min(1.0, len(ub) / 8.0)  # 经验值：topM=10 时 8 个 bin 覆盖已经不错

        # 3) 连续性：最长连续段 / 覆盖 bin 数
        longest = 1
        cur = 1
        for i in range(1, len(ub)):
            if ub[i] == ub[i - 1] + 1:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 1
        cont = longest / max(1, len(ub))

        # 4) 覆盖太少惩罚（典型“蹭到一点纹理”）
        if len(ub) < min_cover_bins:
            base *= 0.72

        score = base * (1.0 + w_cover * cover + w_cont * cont)
        fused.append((img_id, float(score)))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_images]
    return [i for i, _ in fused]


# =============================================================================
# Pipeline Entry
# =============================================================================
def run_pipeline(cfg: RuntimeConfig):
    """
    Pipeline 总入口：
    - cfg: RuntimeConfig，包含路径/阈值/超参/设备等
    - 输出：cfg.out_dir/result_grid1.png
    """
    # ---------------------------
    # 0) 环境准备：输出目录、CUDA 配置
    # ---------------------------
    ensure_dir(cfg.out_dir)

    # 你原脚本是全局开 benchmark + matmul_precision，这里保持一致
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ---------------------------
    # 1) Load FAISS indices + metadata
    # ---------------------------
    # global.index: 图级 embedding 的 IVF/Flat 等索引（取决于你建库）
    # patch_stripe.index / patch_grid.index: patch embedding 的索引
    g_index = faiss.read_index(cfg.global_index)
    p_index_s = faiss.read_index(cfg.patch_stripe_index)
    p_index_g = faiss.read_index(cfg.patch_grid_index)

    # nprobe 控制 IVF 搜索探测多少个倒排桶：大一点更准但更慢
    set_faiss_nprobe(g_index, 64)
    set_faiss_nprobe(p_index_s, 64)
    set_faiss_nprobe(p_index_g, 64)

    # global_meta：img_id -> img_path
    img_paths = np.load(cfg.global_meta, allow_pickle=True)

    # patch_meta_s / patch_meta_g：patch_id -> [img_id, ...]
    patch_meta_s = np.load(cfg.patch_stripe_meta, allow_pickle=True)
    patch_meta_g = np.load(cfg.patch_grid_meta, allow_pickle=True)

    # ---------------------------
    # 2) Load backbone model
    # ---------------------------
    # build_model 内部：
    # - 通过 mmpretrain Config 构建模型
    # - load_checkpoint 加载权重
    # - 返回 mean/std/to_rgb（用于预处理保持一致）
    model, mean, std, to_rgb = build_model(cfg.cfg_path, cfg.ckpt_path, device=cfg.device)

    # ---------------------------
    # 3) Read query image
    # ---------------------------
    qimg0 = imread_unicode(cfg.query_img)
    if qimg0 is None:
        raise RuntimeError(f"Failed to read query image: {cfg.query_img}")

    # ---------------------------
    # 4) Segmentation & crop (YOLO)
    # ---------------------------
    # 目的：把 query 里主体抠出来，减少背景干扰，提高检索稳定性
    # 输出：
    # - qimg:   背景已填充（mean/white/edge）的 crop 图（用于检索）
    # - qmask:  crop 的 mask（u8 0/255）
    # - qimg_raw: crop 的原图（不填充背景，保留主体真实纹理，用于 head 特征更可靠）
    yolo_provider = YoloSegProvider(cfg.yolo_seg_weights)
    qimg, qmask, qimg_raw = crop_by_yolo_mask_final(
        yolo_provider,
        qimg0,
        score_thr=cfg.seg_score_thr,
        use_classes=cfg.seg_use_classes,
        merge_all=True,
        do_rectify=False,          # 原脚本是 False，保持一致
        warp_border="reflect",
        bg_mode="mean",
        debug_dir=None,
        yolo_imgsz=cfg.yolo_imgsz,
        yolo_iou=cfg.yolo_iou,
        yolo_retina_masks=cfg.yolo_retina_masks,
        yolo_device=cfg.yolo_device,
        yolo_max_det=cfg.yolo_max_det,
    )

    # ---------------------------
    # 5) Query head features (stripe/grid detector)
    # ---------------------------
    # head feature 用于：
    # - 后面 cand_gate_score：快速过滤纹理类型不匹配的候选（head gate）
    # - 决定 global/patch 的权重倾向（条纹/格纹明显时更信 patch）
    q_head_img = make_head_view(qimg_raw, prefer_gray=True)
    q_head = stripe_grid_head_v21(q_head_img)

    # ---------------------------
    # 6) Global feature + global retrieval
    # ---------------------------
    # get_query_global_feat 内部：
    # - resize 到 stripe_long_edge（为了 seed & 一致性）
    # - multi-view（旋转 + center/random crop）做 V 个 view
    # - backbone 提特征并聚合成 1 个向量（mean + power norm + L2）
    qvec = get_query_global_feat(
        model, mean, std, to_rgb, qimg,
        device=cfg.device,
        stripe_long_edge=cfg.stripe_long_edge,
        resize_short=cfg.resize_short,
        crop_size=cfg.crop_size,
        view_plan=cfg.view_plan,
        views_per_image=cfg.views_per_image
    )

    # global FAISS search：返回 TopG 个 img_id
    _, gids = g_index.search(qvec, cfg.topg)
    global_rank = clean_rank(gids[0].tolist())

    # ---------------------------
    # 7) Patch feature + patch retrieval
    # ---------------------------
    # get_query_patch_feats_unified 内部：
    # - resize query 到 long_edge
    # - 调 gen_patch_windows_unified 决定 stripe/grid，并生成 windows
    # - stripe: 还会额外 gen_stripe_windows 再加一些 stripe patch（保持原逻辑）
    # - 对每个 patch resize 224x224 -> backbone -> 得到 patch embedding (N,D)
    q_patch_vecs, n_qpatch, is_stripe2, hw = get_query_patch_feats_unified(
        model, mean, std, to_rgb, qimg,
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

    print(f"[PATCH] is_stripe={is_stripe2}, resized={hw}, n={n_qpatch}")

    # patch FAISS search：
    # - 每个 query patch 搜索 PATCH_TOPK_PER_QPATCH 个 patch_id
    # - 得到 D(距离/相似度) 和 I(patch_id)
    # - faiss_scores_from_D: 把 L2 距离转换成“越大越好”的 score（1/(1+D)）
    # - 聚合 patch hits -> image rank
    if is_stripe2:
        D, I = p_index_s.search(q_patch_vecs, cfg.patch_topk_per_qpatch)
        S = faiss_scores_from_D(p_index_s, D.astype(np.float32))
        patch_rank = aggregate_patch_hits_stripe(
            I.reshape(-1), S.reshape(-1),
            patch_meta_s,
            top_images=cfg.top_patch_images,
            tau=0.15
        )
    else:
        D, I = p_index_g.search(q_patch_vecs, cfg.patch_topk_per_qpatch)
        S = faiss_scores_from_D(p_index_g, D.astype(np.float32))
        patch_rank = aggregate_patch_hits(
            I.reshape(-1), S.reshape(-1),
            patch_meta_g,
            top_images=cfg.top_patch_images,
            tau=0.15
        )

    # ---------------------------
    # 8) RRF fusion + dynamic weighting
    # ---------------------------
    # RRF (Reciprocal Rank Fusion)：
    # - 把 rank 列表转成分数：score = 1/(k + rank)
    # - rank 越靠前分数越高
    rrf_g = rank_to_rrf_score(global_rank, k=cfg.rrf_k)
    rrf_p = rank_to_rrf_score(patch_rank,  k=cfg.rrf_k)

    # global_confidence：看 global top20 的文件名前缀是否集中（同类集中 -> global 更可信）
    conf_g = global_confidence(global_rank, img_paths, topn=20)

    # 你原默认策略：global 比重大（尤其 conf_g 高时）
    w_g = 0.6 + 0.35 * conf_g
    w_p = 1.0 - w_g

    # (策略1) query head 强条纹/强格纹：更信 patch
    if (q_head.get("stripe_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8) or \
       (q_head.get("grid_score", 0.0) > 0.18 and q_head.get("ori_peakedness", 0.0) > 2.8):
        w_g, w_p = 0.20, 0.80

    # (策略2) patch 很强但 global 排名很差：说明 global “拉胯”，提高 patch 权重
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

        # patch 前 10 个在 global 的 median 排名很靠后 -> global 不可信
        if len(gp) >= 5:
            gp_sorted = sorted(gp)
            median_gp = gp_sorted[len(gp_sorted) // 2]
            if median_gp >= 120:
                w_g, w_p = 0.15, 0.85

        # global_confidence 也低 -> 更偏 patch
        if conf_g < 0.25:
            w_g, w_p = min(w_g, 0.20), max(w_p, 0.80)
    except Exception:
        pass

    # 最后保护一下权重范围，避免极端值
    w_g = float(np.clip(w_g, 0.05, 0.95))
    w_p = float(np.clip(1.0 - w_g, 0.05, 0.95))
    print(f"[W] conf_g={conf_g:.3f}  w_g={w_g:.3f}  w_p={w_p:.3f}")

    # 加权 RRF：final_rrf 就是最终“融合召回排序”的核心分数来源
    final_rrf = {}
    for k, v in rrf_g.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_g * v
    for k, v in rrf_p.items():
        final_rrf[k] = final_rrf.get(k, 0.0) + w_p * v

    # ---------------------------
    # 9) Candidate pool selection
    # ---------------------------
    # 注意：rrf_fuse 是简单相加的融合（不带权重）
    # 你原来的修复点：不要直接用 rrf_fuse 的排序结果，
    # 而是把它当“候选集合”，再用 final_rrf 重新排序（关键修复点）
    cand = clean_rank(rrf_fuse(global_rank, patch_rank, cfg.rrf_k))

    # 用 final_rrf 重新排序候选集
    cand.sort(key=lambda i: final_rrf.get(i, 0.0), reverse=True)

    # 取前 GEOM_TOPN（后面要做 gate + geom rerank，太多会慢）
    fused = cand[:cfg.geom_topn]

    # ---------------------------
    # 10) Head gate: fast filter by texture type
    # ---------------------------
    # cand_gate_score 逻辑：
    # - 计算候选图 head 特征（stripe/grid/peakedness/fft）
    # - 与 query head 特征做相似度与若干 hard/soft 规则
    # - 返回 gate 值 g0，<=0 直接淘汰
    fused2 = []
    cimg_cache = {}  # 缓存候选图片，避免后面重复读盘

    for img_id in fused:
        cimg = imread_unicode(img_paths[img_id])
        if cimg is None:
            continue
        g0 = cand_gate_score(cimg, q_head)
        if g0 > 0:
            fused2.append(img_id)
            cimg_cache[img_id] = cimg

    # ---------------------------
    # 11) Geom rerank (featmap patch consistency)
    # ---------------------------
    # 核心思想：
    # - 在 backbone 的 feature map 上选取高能量点作为“局部描述子 patch”
    # - q_desc/q_xy 与 c_desc/c_xy 做相似度矩阵
    # - mutual NN + margin 过滤 -> 统计一致性
    # - 对 periodic/纹理图样有特殊处理（texture_score fallback）
    #
    # 角度增广（你原逻辑）：
    # - stripe 图：角度更小（[-10..10]），避免破坏条纹结构
    # - 非 stripe：角度更大（[-30..30]），更鲁棒
    is_vertical_stripe = is_stripe2
    angles = [-10, -5, 0, 5, 10] if is_vertical_stripe else [-30, -15, 0, 15, 30]

    # 11.1 构造 query 的多角度 featmap patches（数量不大，逐个 forward）
    q_desc_list, q_xy_list = [], []
    for ang in angles:
        qimg_r = rotate_bgr(qimg, ang)

        # 生成给 featmap 用的输入（pad->square, resize->512, normalize）
        qx = make_single_tensor_for_rerank(
            qimg_r, mean, std, to_rgb, rmac_input_size=cfg.rmac_input_size
        ).to(cfg.device)

        # 从 backbone 某一层拿 feature map（feat_level=-2 默认）
        q_fm = extract_featmap(model, qx, cfg.feat_level)

        # 从 feature map 选取 query patches（含 ROI + 全局部分）
        q_desc, q_xy = select_query_patches(
            q_fm,
            keep=cfg.keep_patches,
            border=cfg.border,
            roi_frac=cfg.q_roi_frac,
            roi_ratio=cfg.q_roi_ratio
        )
        q_desc_list.append(q_desc.float())
        q_xy_list.append(q_xy.float())

    # 11.2 候选图：一次 batch forward 提取 (c_desc, c_xy)（加速）
    # batch_candidate_desc_xy 内部会：
    # - 读取候选图（可用 cache）
    # - 做 make_single_tensor_for_rerank
    # - batch forward 得到 featmap
    # - select_candidate_patches 得到 c_desc/c_xy
    cand_desc_xy = batch_candidate_desc_xy(
        model, fused2, cimg_cache, img_paths, mean, std, to_rgb,
        device=cfg.device, batch_size=32, feat_level=cfg.feat_level,
        rmac_input_size=cfg.rmac_input_size
    )

    # 11.3 对每个候选做 geom_score（对多个角度取平均）
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

    # ---------------------------
    # 12) Fallback: 如果几何分数全失败
    # ---------------------------
    # 可能原因：主体太少、特征点太弱、feature map 互配很少等
    # fallback 策略：用 fused2 或 fused 的前 TOPK 直接输出
    if not scored:
        fallback = fused2[:cfg.topk] if fused2 else fused[:cfg.topk]
        imgs = [imread_unicode(img_paths[i]) for i in fallback]
        scores = [1.0 - i / max(1, len(fallback)) for i in range(len(fallback))]
        out = os.path.join(cfg.out_dir, "result_grid1.png")
        visualize_grid(qimg, imgs, scores, out)
        print(f"[DONE] saved {out}")
        return

    # ---------------------------
    # 13) Final score: fuse final_rrf with normalized geom score
    # ---------------------------
    # 把 geom score 归一化到 [0,1]，然后用一个乘法因子增强：
    #   final_score = final_rrf * (1 + 0.6 * norm_geom)
    # 这样“召回排序”仍由 RRF 决定主基调，geom 负责把“更一致的”往前推。
    geom_vals = [s for _, s in scored]
    gmin, gmax = min(geom_vals), max(geom_vals)

    final = []
    for img_id, gs in scored:
        norm_g = (gs - gmin) / (gmax - gmin + 1e-9)
        fs = final_rrf.get(img_id, 0.0) * (1.0 + 0.6 * norm_g)
        final.append((img_id, fs))

    final.sort(key=lambda x: x[1], reverse=True)
    top = final[:cfg.topk]

    # ---------------------------
    # 14) Visualization output
    # ---------------------------
    # 读取 topK 图片（优先用 cache），输出网格图
    imgs = [cimg_cache[i] if i in cimg_cache else imread_unicode(img_paths[i]) for i, _ in top]
    scores = [s for _, s in top]

    out = os.path.join(cfg.out_dir, "result_grid1.png")
    visualize_grid(qimg, imgs, scores, out)
    print(f"[DONE] saved {out}")