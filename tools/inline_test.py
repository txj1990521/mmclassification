# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from copy import deepcopy
from types import SimpleNamespace

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

# ============================================================
# ✅ 配置区域：直接在这里改，不用传命令行参数
# ============================================================

CONFIG = dict(
    # 必填：配置文件 & checkpoint
    config="zhanlan/mocov2_classify/mocov2_resnet50_8xb32-coslr-200e_in1k_zhanlan.py",
    checkpoint="work_dirs/mocov2_resnet50_8xb32-coslr-200e_in1k_zhanlan/epoch_200.pth",

    # 可选：工作目录（评测指标、日志等）
    work_dir=None,  # 例如 "./work_dirs/test_run"

    # 可选：输出结果文件（pred 或 metrics）
    out=None,       # 例如 "./outputs/results.pkl" / "./outputs/metrics.json"
    out_item="pred",  # "pred" / "metrics"；默认 pred

    # 可选：等价于 --cfg-options（用 dict 覆盖配置）
    cfg_options=None,  # 例如 dict(model=dict(backbone=dict(depth=50)))

    # 可选：AMP
    amp=False,

    # 可选：可视化
    show=False,
    show_dir=None,   # 例如 "./vis"
    interval=1,
    wait_time=2.0,

    # 可选：dataloader
    no_pin_memory=False,

    # 可选：TTA
    tta=False,

    # 可选：分布式/启动方式
    launcher="none",  # "none" / "pytorch" / "slurm" / "mpi"
    local_rank=0,
)

# ============================================================
# 下面是脚本逻辑（通常不需要改）
# ============================================================


def build_args_from_config(cfg_dict) -> SimpleNamespace:
    """把顶部 CONFIG 字典转换成 args 对象，模拟原 argparse 的 args。"""
    args = SimpleNamespace(**cfg_dict)

    # 兼容原脚本的 LOCAL_RANK 环境变量逻辑
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(getattr(args, "local_rank", 0))
    return args


def merge_args(cfg: Config, args: SimpleNamespace) -> Config:
    """Merge arguments from CONFIG to mmengine Config."""
    cfg.launcher = args.launcher

    # work_dir: CONFIG > cfg.work_dir > default(./work_dirs/<config_name>)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs",
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # AMP
    if args.amp:
        # 某些项目里可能是 cfg.test_cfg.fp16 或 cfg.fp16 等；这里保持原逻辑
        if cfg.get("test_cfg", None) is None:
            cfg.test_cfg = ConfigDict()
        cfg.test_cfg.fp16 = True

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert "visualization" in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of config. ' \
            'Please set `visualization=dict(type="VisualizationHook")`.'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- TTA --------------------
    if args.tta:
        if "tta_model" not in cfg:
            cfg.tta_model = dict(type="mmpretrain.AverageClsScoreTTA")
        if "tta_pipeline" not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(type="RandomFlip", prob=1.0),
                        dict(type="RandomFlip", prob=0.0),
                    ],
                    [test_pipeline[-1]],
                ],
            )
            cfg.tta_pipeline[-1] = flip_tta

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type="default_collate"),
    )

    def set_default_dataloader_cfg(cfg: Config, field: str):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]["pin_memory"] = False

    set_default_dataloader_cfg(cfg, "test_dataloader")

    # cfg_options 覆盖
    if args.cfg_options is not None:
        if not isinstance(args.cfg_options, dict):
            raise TypeError("cfg_options must be a dict (e.g. dict(model=..., test_dataloader=...))")
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = build_args_from_config(CONFIG)

    if args.out is None and args.out_item is not None:
        # 与原脚本一致：使用 out_item 必须给 out
        # 但这里更友好一点：如果 out_item 不是 None，且 out 没配，就给个提醒并继续跑（不输出文件）
        print("[WARN] out_item is set but out is None. Will not dump results to file.")

    # load config
    cfg = Config.fromfile(args.config)

    # merge config from CONFIG area
    cfg = merge_args(cfg, args)

    # build runner
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # dump predictions
    if args.out and args.out_item in ["pred", None]:
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))

    # start testing
    metrics = runner.test()

    # dump metrics
    if args.out and args.out_item == "metrics":
        mmengine.dump(metrics, args.out)

    return metrics


if __name__ == "__main__":
    main()
