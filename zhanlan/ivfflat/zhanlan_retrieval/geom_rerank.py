# zhanlan/ivfflat/zhanlan_retrieval/geom_rerank.py
import cv2
import numpy as np
import torch


@torch.no_grad()
def compute_Ng0(q_desc, c_desc, margin=0.015):
    sim = q_desc @ c_desc.t()
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0
    topv, topi = torch.topk(sim, k=2, dim=1)
    q_best = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    return int(good.sum().item())


@torch.no_grad()
def texture_score(q_desc, q_xy, c_desc, c_xy,
                  bin_size=0.05, topM=6,
                  topk_core=128, min_pairs=12):
    sim = q_desc @ c_desc.t()
    q_bestv, q_best = torch.max(sim, dim=1)

    K = min(int(topk_core), q_desc.shape[0])
    sel = torch.topk(q_bestv, k=K, largest=True).indices
    if K < min_pairs:
        return 0.0

    qg = q_xy[sel]
    cg = c_xy[q_best[sel]]
    core = float(q_bestv[sel].mean().item())

    d = cg - qg
    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(K)
    m = min(int(topM), int(cnt.numel()))
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(K)

    if peak_ratio > 0.85 and cover_topM < 0.35:
        return 0.0
    if core < 0.45 and peak_ratio > 0.6:
        return 0.0

    return float(core * (0.35 + 0.65 * cover_topM) * (0.5 + 0.5 * peak_ratio))


@torch.no_grad()
def geom_score_adaptive(
        q_desc, q_xy, c_desc, c_xy,
        margin=0.02, min_keep=8,
        bin_size=4.0, topM=6, topk_core=64,
        periodic_peak_thr=0.22,
        periodic_cover_topM_thr=0.70,
        periodic_cover_xy_thr=0.18,
):
    sim = q_desc @ c_desc.t()
    if sim.shape[1] < 2 or sim.shape[0] < 1:
        return 0.0

    topv, topi = torch.topk(sim, k=2, dim=1, largest=True)
    q_best = topi[:, 0]
    q_bestv = topv[:, 0]
    q_2ndv = topv[:, 1]

    c_best = torch.argmax(sim, dim=0)
    idx_q = torch.arange(q_desc.shape[0], device=sim.device)
    mutual = (c_best[q_best] == idx_q)

    good = mutual & ((q_bestv - q_2ndv) > margin)
    Ng0 = int(good.sum().item())
    if Ng0 < min_keep:
        return 0.0

    good_idx = torch.nonzero(good, as_tuple=False).squeeze(1)

    K = min(topk_core, good_idx.numel())
    sel = torch.topk(q_bestv[good_idx], k=K, largest=True).indices
    good_idx = good_idx[sel]

    mi = q_best[good_idx]
    qg = q_xy[good_idx]
    cg = c_xy[mi]
    Ng = int(good_idx.numel())
    if Ng < min_keep:
        return 0.0

    P = min(64, Ng)
    qg2 = qg[:P]
    cg2 = cg[:P]

    ratios = []
    for i in range(P):
        j = (i * 7 + 13) % P
        dq = torch.norm(qg2[i] - qg2[j]) + 1e-6
        if float(dq.item()) < 0.08:
            continue
        dc = torch.norm(cg2[i] - cg2[j]) + 1e-6
        rr = (dc / dq).clamp(0.25, 4.0)
        ratios.append(torch.log(rr))

    if len(ratios) < 8:
        return 0.0

    ratio_std = float(torch.stack(ratios).std().item())

    if ratio_std > 1.2:
        shape_gate = 0.40
    elif ratio_std > 0.8:
        shape_gate = 0.70
    else:
        shape_gate = 1.0

    shape_scale = float(np.exp(-ratio_std / 0.55)) * float(shape_gate)

    d = cg - qg
    dx_bin = torch.round(d[:, 0] / bin_size)
    dy_bin = torch.round(d[:, 1] / bin_size)
    keys = dx_bin * 10000 + dy_bin

    _, cnt = torch.unique(keys, return_counts=True)
    cntf = cnt.float()

    peak_ratio = float(cntf.max().item()) / float(Ng)
    m = min(topM, cnt.numel())
    cover_topM = float(torch.topk(cntf, k=m).values.sum().item()) / float(Ng)

    qx = qg[:, 0]
    qy = qg[:, 1]
    cover_x = float((qx.max() - qx.min()).item())
    cover_y = float((qy.max() - qy.min()).item())
    cover_xy = min(cover_x, cover_y)

    core = float(q_bestv[good_idx].mean().item())
    ng_scale = float(min(1.0, Ng / 32.0))

    is_periodic = (
            (peak_ratio < periodic_peak_thr) and
            (cover_topM > periodic_cover_topM_thr) and
            (cover_xy < periodic_cover_xy_thr)
    )

    score = core * peak_ratio * ng_scale * shape_scale
    if is_periodic:
        score *= (0.25 + 0.75 * peak_ratio)

    return float(score)


@torch.no_grad()
def geom_score_compatible(q_desc, q_xy, c_desc, c_xy,
                          margin=0.012, min_keep=5,
                          bin_size=0.05, topM=6, topk_core=64,
                          periodic_peak_thr=0.25,
                          periodic_cover_topM_thr=0.70,
                          periodic_cover_xy_thr=0.22,
                          tex_topk_core=128, tex_min_pairs=12,
                          tex_weight=0.85, return_ng0=False):
    Ng0 = compute_Ng0(q_desc, c_desc, margin=margin)
    if Ng0 >= min_keep:
        s = geom_score_adaptive(
            q_desc, q_xy, c_desc, c_xy,
            margin=margin, min_keep=min_keep,
            bin_size=bin_size, topM=topM, topk_core=topk_core,
            periodic_peak_thr=periodic_peak_thr,
            periodic_cover_topM_thr=periodic_cover_topM_thr,
            periodic_cover_xy_thr=periodic_cover_xy_thr,
        )
        return (float(s), Ng0) if return_ng0 else float(s)

    s_tex = texture_score(q_desc, q_xy, c_desc, c_xy,
                          bin_size=bin_size, topM=topM,
                          topk_core=tex_topk_core, min_pairs=tex_min_pairs)
    out = float(s_tex * tex_weight)
    return (out, Ng0) if return_ng0 else out


# ---- ORB+RANSAC (kept)
def make_orb(nfeatures=2000):
    return cv2.ORB_create(nfeatures)


def geom_score_orb(qimg, cimg, orb=None, min_inliers=8, ransac_thresh=5.0):
    if orb is None:
        orb = make_orb(2000)

    gq = cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)

    kq, dq = orb.detectAndCompute(gq, None)
    kc, dc = orb.detectAndCompute(gc, None)
    if dq is None or dc is None:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(dq, dc, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < int(min_inliers):
        return 0.0

    pts_q = np.float32([kq[m.queryIdx].pt for m in good])
    pts_c = np.float32([kc[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_q, pts_c, cv2.RANSAC, float(ransac_thresh))
    if mask is None:
        return 0.0

    inliers = int(mask.sum())
    return inliers / (len(good) + 1e-6)
