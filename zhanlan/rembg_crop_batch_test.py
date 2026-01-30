import os, glob, csv
import cv2
import numpy as np
import torch

# ====== 可选：用 rembg 做粗定位 ======
USE_REMBG_BOX = True
try:
    from rembg import remove, new_session
    _REMBG_SESSION = None
    def _get_rembg_session(model_name="u2net"):
        global _REMBG_SESSION
        if _REMBG_SESSION is None:
            _REMBG_SESSION = new_session(model_name)
        return _REMBG_SESSION
except Exception:
    USE_REMBG_BOX = False

# ====== SAM2 ======
# 你需要把 sam2 仓库路径加入 sys.path，或者已经 pip install 过 sam2
import sys
SAM2_REPO = r"D:\sam2"  # TODO: 改成你的 sam2 repo 路径（包含 sam2/ 目录）
if SAM2_REPO not in sys.path:
    sys.path.append(SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ====== I/O ======
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_base(rel_path):
    base = os.path.splitext(rel_path)[0]
    base = base.replace("\\", "_").replace("/", "_").replace(":", "_")
    return base

# ====== 辅助：从mask求bbox ======
def bbox_from_mask(mask_u8, pad=10):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 20:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = x2 + pad; y2 = y2 + pad
    return (x1, y1, x2, y2)

# ====== 兜底粗定位：边缘密度找主体区域（不需要调阈值很细） ======
def coarse_box_by_edges(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 60, 150)  # 这组一般不用怎么调
    # 扩张让纹理连起来
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    edges = cv2.dilate(edges, ker, iterations=2)

    # 找最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats((edges>0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (0, 0, w-1, h-1), edges
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    x, y, ww, hh, area = stats[best]
    if area < 0.05 * (h*w):
        return (0, 0, w-1, h-1), edges

    pad = int(0.02 * max(h, w))
    x1 = max(0, x-pad); y1 = max(0, y-pad)
    x2 = min(w-1, x+ww+pad); y2 = min(h-1, y+hh+pad)
    return (x1, y1, x2, y2), edges

# ====== 可选粗定位：rembg mask -> box（不做阈值花活） ======
def coarse_box_by_rembg(img_bgr, model_name="u2net"):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sess = _get_rembg_session(model_name)
    out = remove(img_rgb, session=sess)
    if out is None or not (out.ndim == 3 and out.shape[2] == 4):
        return None, None
    alpha = out[:,:,3].astype(np.uint8)

    # 不再搞一堆阈值：用很宽松的阈值拿个“粗区域”
    thr = max(10, int(np.percentile(alpha, 70)))
    mask = (alpha >= thr).astype(np.uint8) * 255

    # 轻微闭运算连通一下就行
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)

    box = bbox_from_mask(mask, pad=int(0.02*max(h,w)))
    return box, mask

# ====== SAM2：box prompt -> mask -> bbox -> crop ======
@torch.no_grad()
def sam2_crop_bbox(predictor: SAM2ImagePredictor, img_bgr, box_xyxy, pad=10):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    predictor.set_image(img_rgb)

    x1, y1, x2, y2 = box_xyxy
    x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 0, w-1)
    y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 0, h-1)

    # SAM2 期望 box shape: (1,4) float32, xyxy
    box = np.array([[x1, y1, x2, y2]], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True
    )

    # 选最高分 mask
    best = int(np.argmax(scores))
    m = masks[best].astype(np.uint8) * 255

    # bbox from mask
    bb = bbox_from_mask(m, pad=pad)
    if bb is None:
        return None, m, float(scores[best])

    cx1, cy1, cx2, cy2 = bb
    crop = img_bgr[cy1:cy2+1, cx1:cx2+1].copy()
    return (crop, bb, m, float(scores[best]))

def draw_overlay(img_bgr, mask_u8, alpha=0.45):
    if mask_u8 is None:
        return img_bgr
    out = img_bgr.copy()
    green = np.zeros_like(out); green[:,:,1] = 255
    idx = mask_u8 > 0
    out[idx] = (out[idx]*(1-alpha) + green[idx]*alpha).astype(np.uint8)
    return out

def batch_crop_with_sam2(
    in_dir, out_dir,
    sam2_cfg, sam2_ckpt,
    device="cuda",
    save_debug=True
):
    ensure_dir(out_dir)
    crop_dir = os.path.join(out_dir, "crop"); ensure_dir(crop_dir)
    dbg_mask_dir = os.path.join(out_dir, "dbg_mask"); ensure_dir(dbg_mask_dir)
    dbg_ov_dir = os.path.join(out_dir, "dbg_overlay"); ensure_dir(dbg_ov_dir)

    # build predictor
    sam2 = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2)

    # gather images
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(in_dir, f"**/*{ext}"), recursive=True)
    paths.sort()

    report_path = os.path.join(out_dir, "report.csv")
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "path", "method", "coarse_box", "sam_score",
            "final_bbox", "final_area_ratio", "note"
        ])

        for idx, p in enumerate(paths, 1):
            img = imread_unicode(p)
            if img is None:
                continue
            H, W = img.shape[:2]

            rel = os.path.relpath(p, in_dir)
            base = safe_base(rel)

            note = ""
            method = ""

            # 1) coarse box
            coarse_box = None
            coarse_mask = None

            if USE_REMBG_BOX:
                coarse_box, coarse_mask = coarse_box_by_rembg(img, model_name="u2net")
                if coarse_box is not None:
                    method = "rembg_box"
                else:
                    note += "rembg_fail;"
            if coarse_box is None:
                coarse_box, edges = coarse_box_by_edges(img)
                method = "edge_box"

            # 2) SAM2 refine -> bbox crop
            ret = sam2_crop_bbox(predictor, img, coarse_box, pad=int(0.02*max(H,W)))
            if ret is None:
                # fallback: 不裁
                crop = img
                final_bbox = (0, 0, W-1, H-1)
                sam_score = ""
                final_area_ratio = 1.0
                note += "sam_fail;"
                final_mask = coarse_mask if coarse_mask is not None else None
            else:
                crop, final_bbox, final_mask, sam_score = ret
                area = (final_bbox[2]-final_bbox[0]+1) * (final_bbox[3]-final_bbox[1]+1)
                final_area_ratio = float(area / (H*W + 1e-9))

                # 3) 轻量逻辑兜底：如果几乎全图/过小，直接不裁（避免奇怪误切）
                if final_area_ratio > 0.95:
                    note += "too_large_skip;"
                    crop = img
                    final_bbox = (0, 0, W-1, H-1)
                    final_area_ratio = 1.0
                elif final_area_ratio < 0.12:
                    note += "too_small_skip;"
                    crop = img
                    final_bbox = (0, 0, W-1, H-1)
                    final_area_ratio = 1.0

            # save crop
            out_crop = os.path.join(crop_dir, f"{base}_crop.jpg")
            cv2.imwrite(out_crop, crop)

            # debug
            if save_debug:
                if final_mask is not None:
                    cv2.imwrite(os.path.join(dbg_mask_dir, f"{base}_mask.png"), final_mask)
                    ov = draw_overlay(img, final_mask, alpha=0.45)
                    cv2.imwrite(os.path.join(dbg_ov_dir, f"{base}_overlay.jpg"), ov)

            w.writerow([
                p, method,
                str(tuple(map(int, coarse_box))) if coarse_box is not None else "",
                sam_score,
                str(tuple(map(int, final_bbox))) if final_bbox is not None else "",
                f"{final_area_ratio:.4f}",
                note
            ])

            if idx % 50 == 0:
                print(f"[{idx}/{len(paths)}] done")

    print("Saved:", report_path)
    print("Crops:", crop_dir)
    if save_debug:
        print("Debug:", dbg_mask_dir, dbg_ov_dir)

if __name__ == "__main__":
    IN_DIR  = r"D:\zhanlan\qurrey_data"
    OUT_DIR = r"D:\zhanlan\sam2_crop_out"

    SAM2_CFG  = r"D:\sam2\configs\sam2\sam2_hiera_l.yaml"      # TODO 改
    SAM2_CKPT = r"D:\sam2\checkpoints\sam2_hiera_large.pt"     # TODO 改

    batch_crop_with_sam2(
        IN_DIR, OUT_DIR,
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        device="cuda",
        save_debug=True
    )
