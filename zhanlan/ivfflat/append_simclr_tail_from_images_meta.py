#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODE A: FAST append to SimCLR index by consuming the TAIL of shared images_meta.json.
✅ Does NOT modify images_meta.json
✅ Assumes: CLIP append script has already appended new entries into images_meta.json
✅ SimCLR side only "catches up" to the same length by processing images_meta[N_done: ]

Outputs:
    - simclr_global.index (updated in-place)
    - simclr_vec_to_imgid.npy (updated in-place)
    - bad_paths_append_simclr.log (optional)

Important:
    - For perfect alignment, we should NOT silently skip images.
    Default: STRICT_ALIGN=True -> if any tail image fails to read/encode, raise error.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.registry import MODELS

# ============================================================
# ✅ CONFIG (只改这里)
# ============================================================
CONFIG: Dict[str, Any] = {
    # Shared meta (same as CLIP side output)
    "IMAGES_META_JSON": r"D:\zhanlanProject\openai_search\outputs_hybrid_folder_big\images_meta.json",

    # SimCLR index folder/files
    "SIMCLR_OUT_DIR": r"D:\zhanlan\faiss_database_simclr_aligned",
    "SIMCLR_INDEX": "simclr_global.index",
    "SIMCLR_VEC2IMG": "simclr_vec_to_imgid.npy",
    "BAD_LOG": "bad_paths_append_simclr.log",

    # SimCLR model
    "CFG_PATH": r"D:\zhanlanProject\mmpretrain\zhanlan\simclr_resnet50_8xb32-coslr-200e_in1k_build_zhanlan.py",
    "CKPT_PATH": r"D:\zhanlanProject\mmpretrain\work_dirs\simclr_resnet50_8xb32-coslr-200e_in1k_zhanlan\epoch_200.pth",

    # Runtime
    "DEVICE": "auto",  # "auto" / "cpu" / "cuda"
    "BATCH": 256,
    "INPUT_SIZE": 224,

    # Alignment policy
    "STRICT_ALIGN": True,  # True: any failure in tail -> raise (recommended)
    # False: log and skip (will break perfect img_id alignment)
}


# ============================================================
# Utils
# ============================================================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def pick_device(cfg: Dict[str, Any]) -> str:
    dv = str(cfg.get("DEVICE", "auto")).lower()
    if dv in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dv == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_images_meta(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("images_meta.json must be a list")
    # ensure sorted by img_id
    data = sorted(data, key=lambda d: int(d["img_id"]))
    return data


def load_npy_if_exists(path: str, dtype=None):
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=False)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


# ============================================================
# SimCLR model
# ============================================================
def build_model(cfg_path: str, ckpt_path: str, device: str):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    model.eval().to(device)
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)

    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32).reshape(1, 1, 3)
    std = np.array(dp.get("std", [58.395, 57.12, 57.375]), dtype=np.float32).reshape(1, 1, 3)
    to_rgb = bool(dp.get("to_rgb", True))

    return model, mean, std, to_rgb


@torch.no_grad()
def extract_backbone_last(model, bt):
    feat = model.backbone(bt)
    if isinstance(feat, dict):
        feat = feat.get("feat", list(feat.values())[-1])
    if isinstance(feat, (tuple, list)):
        feat = feat[-1]
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))
    feat = F.normalize(feat, p=2, dim=1)
    return feat


def prep_tensor(img_bgr, mean, std, to_rgb, input_size: int):
    img = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) - mean) / std
    x = x.transpose(2, 0, 1)
    return torch.from_numpy(x)


# ============================================================
# Main
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser("append simclr by tail (mode-a)")
    ap.add_argument("--images_meta", type=str, default=None, help="shared images_meta.json")
    ap.add_argument("--simclr_out_dir", type=str, default=None)
    ap.add_argument("--simclr_index", type=str, default=None, help="index file name or full path")
    ap.add_argument("--simclr_vec2img", type=str, default=None, help="map file name or full path")
    ap.add_argument("--cfg_path", type=str, default=None)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--input_size", type=int, default=None)
    ap.add_argument("--strict_align", type=int, default=None, help="1/0")
    return ap.parse_args()


def apply_args(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    cfg = dict(cfg)
    if args.images_meta:
        cfg["IMAGES_META_JSON"] = args.images_meta
    if args.simclr_out_dir:
        cfg["SIMCLR_OUT_DIR"] = args.simclr_out_dir
    if args.simclr_index:
        cfg["SIMCLR_INDEX"] = args.simclr_index
    if args.simclr_vec2img:
        cfg["SIMCLR_VEC2IMG"] = args.simclr_vec2img
    if args.cfg_path:
        cfg["CFG_PATH"] = args.cfg_path
    if args.ckpt_path:
        cfg["CKPT_PATH"] = args.ckpt_path
    if args.device:
        cfg["DEVICE"] = args.device
    if args.batch is not None:
        cfg["BATCH"] = int(args.batch)
    if args.input_size is not None:
        cfg["INPUT_SIZE"] = int(args.input_size)
    if args.strict_align is not None:
        cfg["STRICT_ALIGN"] = bool(int(args.strict_align))
    return cfg


def main():
    args = parse_args()
    cfg = apply_args(CONFIG, args)

    device = pick_device(cfg)
    images_meta_path = cfg["IMAGES_META_JSON"]
    simclr_dir = cfg["SIMCLR_OUT_DIR"]

    def smart_join(base_dir: str, maybe_path: str) -> str:
        # 传了全路径就直接用；否则 join base_dir
        if os.path.isabs(maybe_path) or (":" in maybe_path):
            return maybe_path
        return os.path.join(base_dir, maybe_path)

    simclr_index_path = smart_join(simclr_dir, cfg["SIMCLR_INDEX"])
    simclr_vec2img_path = smart_join(simclr_dir, cfg["SIMCLR_VEC2IMG"])
    bad_log_path = smart_join(simclr_dir, cfg["BAD_LOG"])
    strict = bool(cfg.get("STRICT_ALIGN", True))

    ensure_dir(simclr_dir)

    if not os.path.exists(simclr_index_path):
        raise FileNotFoundError(simclr_index_path)
    if not os.path.exists(images_meta_path):
        raise FileNotFoundError(images_meta_path)

    # load meta
    images_meta = load_images_meta(images_meta_path)
    N_meta = len(images_meta)

    # load vec2img (must exist for MODE A)
    vec2img = load_npy_if_exists(simclr_vec2img_path, dtype=np.int32)
    if vec2img is None:
        raise FileNotFoundError(
            f"Missing {simclr_vec2img_path}. MODE A requires an existing full-build once to create it."
        )

    N_done = int(vec2img.shape[0])
    if N_done > N_meta:
        raise RuntimeError(f"SimCLR vec2img({N_done}) > images_meta({N_meta}) -> meta rollback or mismatch")

    # load index
    index = faiss.read_index(simclr_index_path)
    if not index.is_trained:
        raise RuntimeError("SimCLR index is not trained. Build/train it first (Flat/IVF/IVFPQ).")
    if index.ntotal != N_done:
        raise RuntimeError(f"[SIMCLR] index.ntotal={index.ntotal} != vec2img_len={N_done} (broken state)")

    # tail to process
    todo = images_meta[N_done:N_meta]
    if not todo:
        print("[OK] SimCLR already up-to-date. ntotal=", index.ntotal, "meta=", N_meta)
        return

    # sanity: tail img_id must be continuous and start from N_done
    if int(todo[0]["img_id"]) != N_done:
        raise RuntimeError(f"Tail start img_id={todo[0]['img_id']} != N_done={N_done} (images_meta not continuous?)")

    for j, item in enumerate(todo):
        expect = N_done + j
        if int(item["img_id"]) != expect:
            raise RuntimeError(f"images_meta img_id not continuous at tail: got {item['img_id']} expect {expect}")

    # build model
    model, mean, std, to_rgb = build_model(cfg["CFG_PATH"], cfg["CKPT_PATH"], device)
    batch_size = int(cfg.get("BATCH", 256))
    input_size = int(cfg.get("INPUT_SIZE", 224))

    bad_records: List[str] = []
    added_vecs = 0
    buf_t: List[torch.Tensor] = []
    buf_img_ids: List[int] = []

    def flush():
        nonlocal added_vecs, buf_t, buf_img_ids
        if not buf_t:
            return
        bt = torch.stack(buf_t, dim=0).to(device, non_blocking=True)
        fv = extract_backbone_last(model, bt).cpu().numpy().astype("float32")
        index.add(fv)
        added_vecs += int(fv.shape[0])
        buf_t, buf_img_ids = [], []

    # append in strict tail order
    for item in tqdm(todo, desc="SimCLR catch-up (MODE A)"):
        img_id = int(item["img_id"])
        p = item.get("abs_path", "")

        if (not isinstance(p, str)) or (not p) or (not os.path.exists(p)):
            msg = f"[MISSING]\timg_id={img_id}\t{p}"
            bad_records.append(msg)
            if strict:
                raise RuntimeError("STRICT_ALIGN=True, tail has missing path: " + msg)
            continue

        img = imread_unicode(p)
        if img is None:
            msg = f"[BAD_IMAGE]\timg_id={img_id}\t{p}"
            bad_records.append(msg)
            if strict:
                raise RuntimeError("STRICT_ALIGN=True, tail has unreadable image: " + msg)
            continue

        try:
            t = prep_tensor(img, mean, std, to_rgb, input_size)
        except Exception as e:
            msg = f"[PREP_FAIL]\timg_id={img_id}\t{p}\t{repr(e)}"
            bad_records.append(msg)
            if strict:
                raise RuntimeError("STRICT_ALIGN=True, tail prep failed: " + msg)
            continue

        buf_t.append(t)
        buf_img_ids.append(img_id)
        if len(buf_t) >= batch_size:
            flush()

    flush()

    # update vec2img
    # MODE A: img_id sequence appended must be exactly N_done..N_meta-1
    # If strict=False and skips happened, we'd break alignment; we block it unless you accept it.
    if strict:
        # In strict mode, we guarantee no skips, so:
        new_tail = np.arange(N_done, N_meta, dtype=np.int32)
        vec2img_new = np.concatenate([vec2img, new_tail], axis=0)
        if added_vecs != (N_meta - N_done):
            raise RuntimeError(f"[SIMCLR] strict: added_vecs={added_vecs} != tail_len={N_meta - N_done}")
    else:
        # non-strict: infer from index size difference (but alignment may break)
        # safer: still append the successful ids in order we processed (buf_img_ids were reset at flush),
        # so we need to re-collect. In non-strict, better to keep a list.
        # For simplicity, we forbid non-strict here unless you want it.
        raise RuntimeError("STRICT_ALIGN=False is disabled by default because it breaks img_id alignment.")

    # final sanity
    if index.ntotal != len(vec2img_new):
        raise RuntimeError(f"[SIMCLR] after append: ntotal={index.ntotal} != vec2img_len={len(vec2img_new)}")

    # write outputs (in-place update)
    faiss.write_index(index, simclr_index_path)
    np.save(simclr_vec2img_path, vec2img_new.astype(np.int32))

    if bad_records:
        with open(bad_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(bad_records))

    print("\n===== DONE: SIMCLR MODE-A CATCH-UP =====")
    print(f"device={device}")
    print(f"meta_total={N_meta} | before_done={N_done} | appended={N_meta - N_done}")
    print(f"SIMCLR ntotal={index.ntotal}")
    print("saved index:", simclr_index_path)
    print("saved map :", simclr_vec2img_path)
    if bad_records:
        print("bad log:", bad_log_path)


if __name__ == "__main__":
    main()