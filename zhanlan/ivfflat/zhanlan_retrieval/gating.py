# zhanlan/ivfflat/zhanlan_retrieval/gating.py
import numpy as np
from .head_features import stripe_grid_head_v21, make_head_view, is_grid_like


def cand_gate_score(cimg_bgr, q_head: dict, return_dbg=False):
    ch = stripe_grid_head_v21(make_head_view(cimg_bgr, prefer_gray=False))

    wP = 0.25
    qs = np.array([q_head["stripe_score"], q_head["grid_score"],
                   wP * np.clip(q_head["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)
    cs = np.array([ch["stripe_score"], ch["grid_score"],
                   wP * np.clip(ch["ori_peakedness"]/6.0, 0.0, 1.0)], np.float32)

    qsn = float(np.linalg.norm(qs))
    csn = float(np.linalg.norm(cs))

    if csn < 0.06 or qsn < 1e-6:
        sim = 0.0
    else:
        sim = float((qs * cs).sum() / (qsn * csn + 1e-6))
        sim = max(0.0, min(1.0, sim))

    def _ret(g, reason):
        if return_dbg:
            return float(g), str(reason), float(sim), ch
        return float(g)

    q_is_grid = is_grid_like(q_head)
    if q_is_grid and (not is_grid_like(ch)):
        if q_head.get("grid_score", 0.0) >= 0.35:
            return _ret(0.0, "grid_mismatch_hard")
        return _ret(0.20, "grid_mismatch_soft")

    plaid_penalty = 1.0
    if q_head.get("grid_score", 0.0) > 0.18:
        if not is_grid_like(ch):
            return _ret(0.0, "plaid_miss")
        plaid_penalty = 1.0
    elif q_head.get("grid_score", 0.0) > 0.10:
        if ch.get("grid_score", 0.0) < 0.02:
            plaid_penalty = 0.45

    if q_head.get("stripe_score", 0.0) > 0.15 and ch.get("stripe_score", 0.0) < 0.08:
        return _ret(0.0, "stripe_miss")

    if sim < 0.15:
        return _ret(0.0, "sim_low")

    peak_penalty = 1.0
    if q_head.get("ori_peakedness", 0.0) > 5.5 and ch.get("ori_peakedness", 0.0) < 3.8:
        peak_penalty = 0.4

    scale_penalty = 1.0
    if ("r_peak" in q_head) and ("r_peak" in ch):
        rq = float(q_head.get("r_peak", 0.0))
        rc = float(ch.get("r_peak", 0.0))
        if q_head.get("ori_peakedness", 0.0) >= 2.5 and rq > 1e-6 and rc > 1e-6:
            ratio = rc / (rq + 1e-6)
            if ratio < 0.50 or ratio > 3.00:
                scale_penalty = 0.35
            elif ratio < 0.75 or ratio > 1.60:
                scale_penalty = 0.75

    g = float((0.60 + 0.55 * sim) * peak_penalty * plaid_penalty * scale_penalty)
    return _ret(g, "pass")
