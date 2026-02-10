# zhanlan/ivfflat/zhanlan_retrieval/yolo_seg.py
import numpy as np
import torch
from ultralytics import YOLO


class YoloSegProvider:
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self._model = None

    def model(self):
        if self._model is None:
            self._model = YOLO(self.weights_path)
        return self._model


def yolo_extract_mask_u8(result, H, W, conf_thr=0.6, use_classes=None, merge_all=True):
    if result is None:
        return None

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if masks is None or boxes is None or masks.data is None or len(masks.data) == 0:
        return None

    m = masks.data
    conf = boxes.conf if boxes.conf is not None else torch.ones((m.shape[0],), device=m.device)
    cls = boxes.cls

    keep = conf >= float(conf_thr)
    if use_classes is not None and cls is not None:
        use = torch.tensor(use_classes, device=m.device, dtype=cls.dtype)
        keep = keep & torch.isin(cls, use)

    idx = torch.where(keep)[0]
    if idx.numel() == 0:
        return None

    m_keep = m[idx]
    if not merge_all:
        best_local = torch.argmax(conf[idx]).item()
        mm = m_keep[best_local]
    else:
        mm = torch.any(m_keep > 0.5, dim=0)

    mm = mm.float().unsqueeze(0).unsqueeze(0)
    mm = torch.nn.functional.interpolate(mm, size=(H, W), mode="nearest")
    mm = mm[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    return mm
