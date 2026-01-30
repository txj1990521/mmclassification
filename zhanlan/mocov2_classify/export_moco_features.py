#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pickle

import torch
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner

# =========================
# 直接改这里
# =========================
CONFIG_PATH = r"zhanlan/mocov2_classify/mocov2_resnet50_test_zhanlan.py"
CKPT_PATH = r"work_dirs/mocov2_resnet50_8xb32-coslr-200e_in1k_zhanlan/epoch_200.pth"
OUT_PKL = r"work_dirs/features/test_features.pkl"

# 用 config 里的 val_dataloader 来读 test（你已经配好了 data_prefix='test'）
USE_VAL_DATALOADER = True

# 输出哪一层特征：
# - "backbone": 2048-d (resnet50 最后 stage 的全局池化前/后要看实现)
# - "neck": 128-d (moco 投影后的特征，通常用于对比学习)
FEATURE_FROM = "neck"  # "backbone" or "neck"

# batch 推理设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = False
# =========================
def _unwrap_feat(x):
    # backbone/neck 有时返回 tuple/list
    if isinstance(x, (list, tuple)):
        # 通常最后一个是最深层特征
        x = x[-1]
    return x

def _global_pool_if_map(x):
    # (N,C,H,W) -> (N,C)
    if x.ndim == 4:
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
    return x


@torch.no_grad()
def main():
    cfg = Config.fromfile(CONFIG_PATH)
    # 不让 runner 走 val/test loop，我们只借它来 build dataloader/model
    cfg.setdefault('work_dir', './work_dirs/export_features')
    runner = Runner.from_cfg(cfg)

    # build model & load ckpt
    runner.load_checkpoint(CKPT_PATH)
    model = runner.model.to(DEVICE)
    model.eval()

    # build dataloader（用你配的 val_dataloader 读取 test）
    dl_cfg = cfg.test_dataloader

    dataloader = runner.build_dataloader(dl_cfg)

    out_path = Path(OUT_PKL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    scaler = torch.cuda.amp.autocast(enabled=FP16)

    for batch in dataloader:
        # batch 是 PackInputs 产物：inputs + data_samples
        inputs = batch["inputs"].to(DEVICE, non_blocking=True)
        # 1) Byte -> Float
        if inputs.dtype == torch.uint8:
            inputs = inputs.float().div_(255.0)
        else:
            inputs = inputs.float()

        # 2) 按 config 里的 mean/std 做 normalize（注意：mean/std 是 0~255 标准）
        mean = torch.tensor([123.675, 116.28, 103.53], device=inputs.device).view(1, 3, 1, 1) / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], device=inputs.device).view(1, 3, 1, 1) / 255.0
        inputs = (inputs - mean) / std
        # 取每张图的路径（data_samples 里有路径信息）
        # 不同版本字段名可能略有差异，做个兼容
        paths = []
        for ds in batch["data_samples"]:
            if hasattr(ds, "img_path"):
                paths.append(ds.img_path)
            elif hasattr(ds, "metainfo") and "img_path" in ds.metainfo:
                paths.append(ds.metainfo["img_path"])
            else:
                paths.append(None)

        with scaler:
            if FEATURE_FROM == "backbone":
                feat = _unwrap_feat(model.backbone(inputs))
                feat = _global_pool_if_map(feat)

            elif FEATURE_FROM == "neck":
                feat = _unwrap_feat(model.backbone(inputs))
                feat = _global_pool_if_map(feat)

                feat = model.neck(feat)
                feat = _unwrap_feat(feat)  # neck 也可能返回 tuple/list

            else:
                raise ValueError("FEATURE_FROM must be 'backbone' or 'neck'")

        feat = feat.detach().cpu().numpy().astype(np.float32)

        for p, f in zip(paths, feat):
            results.append({"img_path": p, "feat": f})

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[DONE] Saved {len(results)} samples to: {out_path}")


if __name__ == "__main__":
    main()
