# seg_cropper.py
# -*- coding: utf-8 -*-
"""
Reusable mask-based cropper using MMDetection DetInferencer.

Example:
    from seg_cropper import SegMaskCropper
    import cv2

    cropper = SegMaskCropper(
        model_config=r"D:/.../mask-rcnn_r50_fpn_1x_coco.py",
        weights=r"D:/.../epoch_12.pth",
        device="cuda:0",
    )

    img = cv2.imread("query.jpg")
    crop = cropper.crop(img, debug_dir="D:/out")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Any, Dict

import cv2
import numpy as np

try:
    from mmdet.apis import DetInferencer
except Exception as e:
    DetInferencer = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass
class CropResult:
    cropped: np.ndarray
    ok: bool
    bbox_xyxy: Optional[tuple] = None  # (x1,y1,x2,y2) in original image coords
    used_instances: int = 0
    best_score: float = 0.0


class SegMaskCropper:
    """
    A reusable cropper that uses instance segmentation masks from DetInferencer
    to crop the main foreground region.

    - Initializes DetInferencer lazily (first call).
    - Input/Output: BGR uint8 ndarray.
    """

    def __init__(
        self,
        model_config: str,
        weights: str,
        device: str = "cuda:0",
        palette: str = "random",
        # Morphology parameters (can be tuned)
        morph_kernel: int = 11,      # ellipse size
        close_iter: int = 2,
        open_iter: int = 1,
        erode_iter: int = 1,
        # For rare versions that don't accept ndarray input:
        force_file_input: bool = False,
        tmp_dir: Optional[str] = None,
    ):
        if DetInferencer is None:
            raise ImportError(
                "mmdet is not available / DetInferencer import failed. "
                f"Original error: {_IMPORT_ERR}"
            )

        self.model_config = model_config
        self.weights = weights
        self.device = device
        self.palette = palette

        self.morph_kernel = int(morph_kernel)
        self.close_iter = int(close_iter)
        self.open_iter = int(open_iter)
        self.erode_iter = int(erode_iter)

        self.force_file_input = bool(force_file_input)
        self.tmp_dir = tmp_dir

        self._inferencer: Optional[DetInferencer] = None

    # -------------------------
    # Public API
    # -------------------------
    def crop(
        self,
        img_bgr: np.ndarray,
        *,
        pad: int = 20,
        min_area_frac: float = 0.08,
        score_thr: float = 0.6,
        use_classes: Optional[Sequence[int]] = None,
        merge_all: bool = True,
        debug_dir: Optional[str] = None,
        debug_prefix: str = "dbg",
    ) -> np.ndarray:
        """
        Returns cropped BGR image. If fails, returns the original image.
        """
        res = self.crop_with_info(
            img_bgr,
            pad=pad,
            min_area_frac=min_area_frac,
            score_thr=score_thr,
            use_classes=use_classes,
            merge_all=merge_all,
            debug_dir=debug_dir,
            debug_prefix=debug_prefix,
        )
        return res.cropped

    def crop_with_info(
        self,
        img_bgr: np.ndarray,
        *,
        pad: int = 20,
        min_area_frac: float = 0.08,
        score_thr: float = 0.6,
        use_classes: Optional[Sequence[int]] = None,
        merge_all: bool = True,
        debug_dir: Optional[str] = None,
        debug_prefix: str = "dbg",
    ) -> CropResult:
        """
        Returns detailed crop result with bbox and status.
        """
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return CropResult(cropped=img_bgr, ok=False)

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            # Expect BGR 3-ch
            return CropResult(cropped=img_bgr, ok=False)

        H, W = img_bgr.shape[:2]

        infer = self._get_inferencer()

        # Run inference
        try:
            res = self._infer(
                inferencer=infer,
                img_bgr=img_bgr,
                score_thr=float(score_thr),
                debug_dir=debug_dir,
                debug_prefix=debug_prefix,
            )
        except Exception:
            # Any infer failure => fallback
            return CropResult(cropped=img_bgr, ok=False)

        inst = self._extract_instances(res)
        if inst is None:
            return CropResult(cropped=img_bgr, ok=False)

        scores = inst.get("scores", None)
        masks = inst.get("masks", None)
        labels = inst.get("labels", None)

        if scores is None or masks is None:
            return CropResult(cropped=img_bgr, ok=False)

        scores = np.asarray(scores)
        masks = np.asarray(masks)

        # masks: (N,H,W)
        if masks.ndim != 3 or masks.shape[1] != H or masks.shape[2] != W:
            return CropResult(cropped=img_bgr, ok=False)

        keep = scores >= float(score_thr)

        if labels is not None and use_classes is not None:
            labels = np.asarray(labels)
            keep = keep & np.isin(labels, np.asarray(list(use_classes), dtype=labels.dtype))

        idx = np.where(keep)[0]
        if idx.size == 0:
            return CropResult(cropped=img_bgr, ok=False)

        best_score = float(scores[idx].max()) if idx.size else 0.0

        if merge_all:
            merged = np.any(masks[idx].astype(bool), axis=0)
        else:
            best = idx[int(np.argmax(scores[idx]))]
            merged = masks[best].astype(bool)

        mask_u8 = (merged.astype(np.uint8) * 255)

        # Morph cleanup
        mask_u8 = self._cleanup_mask(mask_u8)

        # Largest CC
        ok, bbox = self._largest_cc_bbox(mask_u8, min_area_frac=float(min_area_frac))
        if not ok or bbox is None:
            return CropResult(cropped=img_bgr, ok=False, used_instances=int(idx.size), best_score=best_score)

        x0, y0, ww, hh, area = bbox
        x1 = max(0, int(x0 - pad))
        y1 = max(0, int(y0 - pad))
        x2 = min(W, int(x0 + ww + pad))
        y2 = min(H, int(y0 + hh + pad))

        crop = img_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return CropResult(cropped=img_bgr, ok=False, used_instances=int(idx.size), best_score=best_score)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_mask.png"), mask_u8)
            dbg = img_bgr.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_bbox.png"), dbg)

        return CropResult(
            cropped=crop,
            ok=True,
            bbox_xyxy=(x1, y1, x2, y2),
            used_instances=int(idx.size),
            best_score=best_score,
        )

    # -------------------------
    # Internal helpers
    # -------------------------
    def _get_inferencer(self) -> DetInferencer:
        if self._inferencer is None:
            self._inferencer = DetInferencer(
                model=self.model_config,
                weights=self.weights,
                device=self.device,
                palette=self.palette,
            )
        return self._inferencer

    def _infer(
        self,
        inferencer: DetInferencer,
        img_bgr: np.ndarray,
        score_thr: float,
        debug_dir: Optional[str],
        debug_prefix: str,
    ) -> Dict[str, Any]:
        """
        Calls DetInferencer. Tries ndarray input by default.
        If force_file_input=True, writes temporary file and passes path.
        """
        if self.force_file_input:
            return self._infer_by_temp_file(inferencer, img_bgr, score_thr)

        # Try ndarray input
        try:
            return inferencer(
                inputs=[img_bgr],
                pred_score_thr=float(score_thr),
                batch_size=1,
                show=False,
                no_save_vis=True,
                no_save_pred=True,
                print_result=False,
                out_dir="",
            )
        except Exception:
            # fallback to file input
            return self._infer_by_temp_file(inferencer, img_bgr, score_thr)

    def _infer_by_temp_file(self, inferencer: DetInferencer, img_bgr: np.ndarray, score_thr: float) -> Dict[str, Any]:
        tmp_dir = self.tmp_dir or os.getcwd()
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, "_tmp_seg_cropper_input.png")
        cv2.imwrite(tmp_path, img_bgr)
        return inferencer(
            inputs=[tmp_path],
            pred_score_thr=float(score_thr),
            batch_size=1,
            show=False,
            no_save_vis=True,
            no_save_pred=True,
            print_result=False,
            out_dir="",
        )

    @staticmethod
    def _extract_instances(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        preds = res.get("predictions", None) or res.get("preds", None)
        if preds is None:
            return None

        p0 = preds[0] if isinstance(preds, list) and len(preds) > 0 else preds

        if isinstance(p0, dict):
            if "pred_instances" in p0 and isinstance(p0["pred_instances"], dict):
                return p0["pred_instances"]
            if "instances" in p0 and isinstance(p0["instances"], dict):
                return p0["instances"]
            # flat
            if ("masks" in p0) or ("scores" in p0):
                return p0
        return None

    def _cleanup_mask(self, mask_u8: np.ndarray) -> np.ndarray:
        k = max(3, int(self.morph_kernel) | 1)  # ensure odd >=3
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        if self.close_iter > 0:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker, iterations=int(self.close_iter))
        if self.open_iter > 0:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, ker, iterations=int(self.open_iter))
        if self.erode_iter > 0:
            mask_u8 = cv2.erode(mask_u8, ker, iterations=int(self.erode_iter))

        return mask_u8

    @staticmethod
    def _largest_cc_bbox(mask_u8: np.ndarray, min_area_frac: float) -> tuple[bool, Optional[tuple]]:
        H, W = mask_u8.shape[:2]
        num, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num <= 1:
            return False, None

        # stats: [label, x, y, w, h, area] (0 is background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_i = 1 + int(np.argmax(areas))
        x0, y0, ww, hh, area = stats[best_i]

        if float(area) < float(min_area_frac) * float(H * W):
            return False, None

        return True, (int(x0), int(y0), int(ww), int(hh), int(area))


if __name__ == "__main__":
    # Minimal self-test (optional)
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    cropper = SegMaskCropper(args.cfg, args.ckpt, device="cuda:0")
    out = cropper.crop_with_info(img, debug_dir=args.out, debug_prefix="test")
    cv2.imwrite(os.path.join(args.out, "test_crop.png"), out.cropped)
    print(out)
