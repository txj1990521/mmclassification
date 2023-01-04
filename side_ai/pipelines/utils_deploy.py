import os
from pathlib import Path
import PIL.Image
from side_ai.pipelines.utils_crypto import open_file


def deploy(model_path):
    import logging
    import numpy as np

    model_path = Path(model_path)

    # 读取模型
    import torch
    if model_path.suffix == '.pth':
        checkpoint = torch.load(model_path, map_location='cpu')
    elif model_path.suffix == '.cpth':
        checkpoint = torch.load(open_file(model_path), map_location='cpu')
    else:
        raise Exception(f'不支持的文件类型：{model_path.suffix}')
    logging.info(f'读取模型完成')

    # 读取配置
    import mmcv

    model_config_path = Path(model_path.parent, model_path.stem + '.py')
    if not os.path.exists(model_config_path):
        if 'config' in checkpoint:
            model_cfg = checkpoint['config']
        elif 'meta' in checkpoint:
            config_str = checkpoint['meta']['config']
            model_cfg = mmcv.Config.fromstring(config_str, file_format='.py')
        else:
            raise Exception('找不到模型配置')
    else:
        model_cfg = mmcv.Config.fromfile(model_config_path)
    logging.info(f'读取模型配置完成')

    model_cfg.merge_from_dict(
        {'data.test.pipeline.0.type': 'LoadImageFromFile'})

    # 一些参数
    import mmcv

    null = None
    true = True
    false = False
    deploy_cfg = mmcv.Config(
        {
            "version": "0.4.0",
            "codebase_config": {
                "type": "mmdet",
                "task": "ObjectDetection",
                "model_type": "end2end_optimized",
                "post_processing": {
                    "score_threshold": 0.05,
                    "confidence_threshold": 0.005,
                    "iou_threshold": 0.5,
                    "max_output_boxes_per_class": 200,
                    "pre_top_k": 5000,
                    "keep_top_k": 100,
                    "background_label_id": -1,
                    "export_postprocess_mask": false
                }
            },
            "onnx_config": {
                "type": "onnx",
                "export_params": true,
                "keep_initializers_as_inputs": false,
                "opset_version": 11,
                "save_file": "end2end.onnx",
                "input_names": ["input"],
                "output_names": ["dets", "labels", "masks"],
                "input_shape": null,
                "dynamic_axes": {
                    "input": {
                        0: "batch",
                        2: "height",
                        3: "width"
                    },
                    "dets": {
                        0: "batch",
                        1: "num_dets"
                    },
                    "labels": {
                        0: "batch",
                        1: "num_dets"
                    },
                    "masks": {
                        0: "batch",
                        1: "num_dets",
                        2: "height",
                        3: "width"
                    }
                }
            },
            "backend_config": {
                "type": "tensorrt",
                "common_config": {
                    "fp16_mode": true,
                    "max_workspace_size": 10000000000
                },
                "model_inputs": [
                    {
                        "input_shapes": {
                            "input": {
                                "min_shape": [1, 3, 64, 64],
                                "opt_shape": [1, 3, 640, 640],
                                "max_shape": [1, 3, 5000, 5000]
                            }
                        }
                    }
                ]
            },
            "calib_config": {}
        })
    deploy_cfg = mmcv.Config.fromstring(
        deploy_cfg.pretty_text, file_format='.py')

    work_dir = Path(model_path.parent, model_path.stem)
    os.makedirs(work_dir, exist_ok=True)
    deploy_cfg_path = str(Path(work_dir, 'deploy_cfg.py'))
    model_cfg_path = str(Path(work_dir, model_path.stem + '.py'))
    checkpoint_path = str(Path(work_dir, model_path.stem + '.pth'))
    img_path = Path(work_dir, 'test_img.jpg')

    deploy_cfg.dump(deploy_cfg_path)
    model_cfg.dump(model_cfg_path)
    torch.save(checkpoint, checkpoint_path)
    PIL.Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8)).save(img_path)
    logging.info(f'导出必备参数完成')

    class A:

        def __init__(self):
            self.device = 'cuda'
            self.log_level = 'INFO'
            self.show = False
            self.dump_info = True
            self.calib_dataset_cfg = None
            self.work_dir = str(work_dir)
            self.img = str(img_path)
            self.test_img = self.img
            self.deploy_cfg = str(deploy_cfg_path)
            self.model_cfg = str(model_cfg_path)
            self.checkpoint = str(checkpoint_path)

    args = A()
    deploy_main(args)


def deploy_main(args):
    import os.path as osp
    from functools import partial

    import mmcv
    import torch.multiprocessing as mp
    from multiprocessing import Process, set_start_method

    from mmdeploy.apis import (
        create_calib_table, extract_model, get_predefined_partition_cfg,
        torch2onnx, torch2torchscript, visualize_model)
    from mmdeploy.utils import (
        IR, Backend, get_backend, get_calib_filename, get_ir_config,
        get_model_inputs, get_partition_config, get_root_logger, load_config,
        target_wrapper)
    from mmdeploy.utils.export_info import dump_info

    def create_process(name, target, args, kwargs, ret_value=None):
        logger = get_root_logger()
        logger.info(f'{name} start.')
        log_level = logger.level

        wrap_func = partial(target_wrapper, target, log_level, ret_value)

        process = Process(target=wrap_func, args=args, kwargs=kwargs)
        process.daemon = True
        process.start()
        process.join()

        if ret_value is not None:
            if ret_value.value != 0:
                logger.error(f'{name} failed.')
                exit(1)
            else:
                logger.info(f'{name} success.')

    def torch2ir(ir_type: IR):
        """Return the conversion function from torch to the intermediate
        representation.

        Args:
            ir_type (IR): The type of the intermediate representation.
        """
        if ir_type == IR.ONNX:
            return torch2onnx
        elif ir_type == IR.TORCHSCRIPT:
            return torch2torchscript
        else:
            raise KeyError(f'Unexpected IR type {ir_type}')

    set_start_method('spawn', force=True)
    logger = get_root_logger()
    logger.setLevel(args.log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg
    checkpoint_path = args.checkpoint

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # create work_dir if not
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    if args.dump_info:
        dump_info(deploy_cfg, model_cfg, args.work_dir, pth=checkpoint_path)

    ret_value = mp.Value('d', 0, lock=False)

    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    create_process(
        f'torch2{ir_type.value}',
        target=torch2ir(ir_type),
        args=(
            args.img, args.work_dir, ir_save_file, deploy_cfg_path,
            model_cfg_path, checkpoint_path),
        kwargs=dict(device=args.device),
        ret_value=ret_value)

    # convert backend
    ir_files = [osp.join(args.work_dir, ir_save_file)]

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:

        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            create_process(
                f'partition model {save_file} with start: {start}, end: {end}',
                extract_model,
                args=(origin_ir_file, start, end),
                kwargs=dict(dynamic_axes=dynamic_axes, save_file=save_path),
                ret_value=ret_value)

            ir_files.append(save_path)

    # calib data
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(args.work_dir, calib_filename)

        create_process(
            'calibration',
            create_calib_table,
            args=(
                calib_path, deploy_cfg_path, model_cfg_path, checkpoint_path),
            kwargs=dict(
                dataset_cfg=args.calib_dataset_cfg,
                dataset_type='val',
                device=args.device),
            ret_value=ret_value)

    backend_files = ir_files
    # convert backend
    backend = get_backend(deploy_cfg)
    if backend == Backend.TENSORRT:
        model_params = get_model_inputs(deploy_cfg)
        assert len(model_params) == len(ir_files)

        from mmdeploy.apis.tensorrt import is_available as trt_is_available
        from mmdeploy.apis.tensorrt import onnx2tensorrt
        assert trt_is_available(
        ), 'TensorRT is not available,' \
            + ' please install TensorRT and build TensorRT custom ops first.'
        backend_files = []
        for model_id, model_param, onnx_path in zip(range(len(ir_files)),
                                                    model_params, ir_files):
            onnx_name = osp.splitext(osp.split(onnx_path)[1])[0]
            save_file = model_param.get('save_file', onnx_name + '.engine')

            partition_type = 'end2end' if partition_cfgs is None \
                else onnx_name
            create_process(
                f'onnx2tensorrt of {onnx_path}',
                target=onnx2tensorrt,
                args=(
                    args.work_dir, save_file, model_id, deploy_cfg_path,
                    onnx_path),
                kwargs=dict(device=args.device, partition_type=partition_type),
                ret_value=ret_value)

            backend_files.append(osp.join(args.work_dir, save_file))

    elif backend == Backend.NCNN:
        from mmdeploy.apis.ncnn import is_available as is_available_ncnn

        if not is_available_ncnn():
            logger.error('ncnn support is not available.')
            exit(1)

        from mmdeploy.apis.ncnn import get_output_model_file, onnx2ncnn

        backend_files = []
        for onnx_path in ir_files:
            model_param_path, model_bin_path = get_output_model_file(
                onnx_path, args.work_dir)
            create_process(
                f'onnx2ncnn with {onnx_path}',
                target=onnx2ncnn,
                args=(onnx_path, model_param_path, model_bin_path),
                kwargs=dict(),
                ret_value=ret_value)
            backend_files += [model_param_path, model_bin_path]

    elif backend == Backend.OPENVINO:
        from mmdeploy.apis.openvino import \
            is_available as is_available_openvino
        assert is_available_openvino(), \
            'OpenVINO is not available, please install OpenVINO first.'

        from mmdeploy.apis.openvino import (
            get_input_info_from_cfg, get_mo_options_from_cfg,
            get_output_model_file, onnx2openvino)
        openvino_files = []
        for onnx_path in ir_files:
            model_xml_path = get_output_model_file(onnx_path, args.work_dir)
            input_info = get_input_info_from_cfg(deploy_cfg)
            output_names = get_ir_config(deploy_cfg).output_names
            mo_options = get_mo_options_from_cfg(deploy_cfg)
            create_process(
                f'onnx2openvino with {onnx_path}',
                target=onnx2openvino,
                args=(
                    input_info, output_names, onnx_path, args.work_dir,
                    mo_options),
                kwargs=dict(),
                ret_value=ret_value)
            openvino_files.append(model_xml_path)
        backend_files = openvino_files

    elif backend == Backend.PPLNN:
        from mmdeploy.apis.pplnn import is_available as is_available_pplnn
        assert is_available_pplnn(), \
            'PPLNN is not available, please install PPLNN first.'

        from mmdeploy.apis.pplnn import onnx2pplnn
        pplnn_files = []
        for onnx_path in ir_files:
            algo_file = onnx_path.replace('.onnx', '.json')
            model_inputs = get_model_inputs(deploy_cfg)
            assert 'opt_shape' in model_inputs, 'Expect opt_shape ' \
                'in deploy config for PPLNN'
            # PPLNN accepts only 1 input shape for optimization,
            # may get changed in the future
            input_shapes = [model_inputs.opt_shape]
            create_process(
                f'onnx2pplnn with {onnx_path}',
                target=onnx2pplnn,
                args=(algo_file, onnx_path),
                kwargs=dict(device=args.device, input_shapes=input_shapes),
                ret_value=ret_value)
            pplnn_files += [onnx_path, algo_file]
        backend_files = pplnn_files

    # if args.test_img is None:
    #     args.test_img = args.img
    # import os
    # is_display = os.getenv('DISPLAY')
    # # for headless installation.
    # if is_display is not None:
    #     # visualize model of the backend
    #     create_process(
    #         f'visualize {backend.value} model',
    #         target=visualize_model,
    #         args=(
    #             model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
    #             args.device),
    #         kwargs=dict(
    #             backend=backend,
    #             output_file=osp.join(
    #                 args.work_dir, f'output_{backend.value}.jpg'),
    #             show_result=args.show),
    #         ret_value=ret_value)
    #
    #     # visualize pytorch model
    #     create_process(
    #         'visualize pytorch model',
    #         target=visualize_model,
    #         args=(
    #             model_cfg_path, deploy_cfg_path, [checkpoint_path],
    #             args.test_img, args.device),
    #         kwargs=dict(
    #             backend=Backend.PYTORCH,
    #             output_file=osp.join(args.work_dir, 'output_pytorch.jpg'),
    #             show_result=args.show),
    #         ret_value=ret_value)
    # else:
    #     logger.warning(
    #         '\"visualize_model\" has been skipped may be because it\'s \
    #         running on a headless device.')
    logger.info('All process success.')
