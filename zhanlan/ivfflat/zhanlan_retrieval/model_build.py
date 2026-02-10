# zhanlan/ivfflat/zhanlan_retrieval/model_build.py
import numpy as np
import torch
import torch.nn.functional as F

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS


@torch.no_grad()
def extract_backbone_last(model, batch_tensor: torch.Tensor):
    feat_map = model.backbone(batch_tensor)

    if isinstance(feat_map, dict):
        if 'feat' in feat_map:
            feat_map = feat_map['feat']
        elif 'features' in feat_map:
            feat_map = feat_map['features']
        else:
            feat_map = list(feat_map.values())[-1]

    if isinstance(feat_map, (tuple, list)):
        feat_map = feat_map[-1]

    if feat_map.dim() == 2:
        feat = feat_map
    else:
        feat = feat_map.mean(dim=(2, 3))

    feat = F.normalize(feat, p=2, dim=1)
    return feat


@torch.no_grad()
def extract_featmap(model, batch_tensor: torch.Tensor, prefer_level: int):
    out = model.backbone(batch_tensor)

    if isinstance(out, dict):
        if "feat" in out:
            out = out["feat"]
        elif "features" in out:
            out = out["features"]
        else:
            out = list(out.values())[-1]

    if isinstance(out, (tuple, list)):
        n = len(out)
        lvl = prefer_level
        if lvl < -n: lvl = -n
        if lvl > n - 1: lvl = n - 1
        return out[lvl]

    if isinstance(out, torch.Tensor):
        return out

    raise TypeError(f"Unsupported backbone output type: {type(out)}")


@torch.no_grad()
def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1, 1, 3)
    std  = np.array(dp.get("std",  [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", True))
    return model, mean, std, to_rgb
