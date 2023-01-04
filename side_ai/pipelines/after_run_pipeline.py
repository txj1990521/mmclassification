import datetime
import os
import re
import time
import traceback
from multiprocessing import freeze_support
from pathlib import Path

import torch

from mmcls.datasets.builder import PIPELINES
from mmdet.utils import get_root_logger
from side_ai.pipelines.utils_crypto import save_file, save_files
from side_ai.pipelines.utils_deploy import deploy


@PIPELINES.register_module()
class DeployModel:

    def __call__(self, results, *args, **kwargs):
        model_path = results.get('model_path', None)

        logger = get_root_logger()
        logger.propagate = False

        if model_path is None:
            logger.warning('没有设置保存模型路径的参数，将无法进行加速操作')
            return results

        start_time = time.time()
        try:
            freeze_support()
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            cuda_visible_devices2 = cuda_visible_devices.split(',')
            if len(cuda_visible_devices2) > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices2[1]
            torch.cuda.empty_cache()
            deploy(model_path)

            end_time = time.time()
            logger.info(f"加速共用了{(end_time - start_time) / 60:.2f}分钟")

            model_path = Path(model_path)
            work_dir = Path(model_path.parent, model_path.stem)
            results['engine_path'] = f"{work_dir}/end2end.engine"
            logger.info(f'加速模型路径：{results["engine_path"]}')
        except:
            logger.warning(f'模型加速失败:\n{traceback.format_exc()}')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class EncryptModel:

    def __init__(self, encrypt_model_path=None):
        self.encrypt_model_path = encrypt_model_path

    def __call__(self, results, *args, **kwargs):
        timestamp = results['timestamp']

        logger = get_root_logger()
        logger.propagate = False

        if self.encrypt_model_path is None:
            model_path = results.get('engine_path', None)
            if model_path is None:
                logger.warning("无法获取加速模型的路径，故加密原模型")
                model_path = results['model_path']
            config_path = Path(Path(model_path).parent, timestamp + '.py')
        else:
            model_path = self.encrypt_model_path
            config_path = Path(model_path).with_suffix(".py")

        try:
            self.encrypt_model(model_path, config_path)
        except:
            logger.warning(f"模型加密失败:\n{traceback.format_exc()}")

        return results

    def encrypt_model(self, model_path, config_path):
        logger = get_root_logger()
        logger.propagate = False

        model_path = Path(model_path)
        logger.info(f'模型加密开始，模型地址：{model_path}')
        ext_map = {'.pth': '.cpth', '.engine': '.ctrt'}
        output_path = Path(
            model_path.parent,
            model_path.stem + ext_map.get(model_path.suffix, '.bin'))
        logger.info(f'加密模型路径：{output_path}')

        logger.info(f'开始加密模型')
        try:
            with open(model_path, 'rb') as f_model:
                if model_path.suffix == '.pth':
                    save_file(output_path, f_model.read())
                elif model_path.suffix == '.engine':
                    file_dict = {model_path.name: f_model.read()}
                    model_config_path = Path(
                        model_path.parent, model_path.stem + '.py')
                    if not os.path.exists(model_config_path):
                        model_config_path = config_path
                    logger.info(f'模型配置文件：{model_config_path}')
                    with open(model_config_path, 'rb') as f_config:
                        file_dict[model_config_path.name] = f_config.read()
                    save_files(output_path, file_dict)
        except:
            logger.error(traceback.format_exc())
            logger.error(f'模型加密失败')
            return
        logger.info(f'模型加密完成')

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SaveLog:

    def __init__(self, create_briefing=True):
        self.create_briefing = create_briefing

    def __call__(self, results, *args, **kwargs):
        out_dir = results['out_dir']
        log_timestamp = results['log_timestamp']
        save_model_path = results['save_model_path']
        timestamp = results['timestamp']

        logger = get_root_logger()
        logger.propagate = False

        log_path = Path(out_dir, log_timestamp + '.log')
        new_log_path = Path(save_model_path, timestamp, log_timestamp + '.log')
        os.makedirs(new_log_path.parent, exist_ok=True)
        try:
            with open(log_path,
                      encoding='utf-8') as f1, open(new_log_path, 'w',
                                                    encoding='utf-8') as f2:
                f2.write(f1.read())
            logger.info('拷贝log成功')
        except:
            logger.error('拷贝log失败')
            logger.error(traceback.format_exc())

        if not self.create_briefing:
            return results

        try:
            logger.info('生成简报中')
            brief_log_path = Path(save_model_path, timestamp, 'briefing.log')
            with open(new_log_path, encoding='utf-8') as srcFile, open(
                    brief_log_path, 'w', encoding='utf-8') as dstFile:
                time_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
                pattern_list = [
                    rf'({time_pattern} .+?bbox_mAP[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?\n Average Precision[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?\n\+----[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?统计置信度分布[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?\n验证集数量[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?统计类别分布[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+\n?数据集的路径[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+\n?映射表为[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?加密模型路径[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?加速模型路径[\s\S]+?){time_pattern}',
                    rf'({time_pattern} .+?加速共用了[\s\S]+?){time_pattern}'
                ]
                pattern_loss = rf'({time_pattern} .+?Epoch [\s\S]+?){time_pattern}'

                text = srcFile.read()
                new_text = []
                for pattern in pattern_list:
                    new_text += re.findall(pattern, text)
                loss_text = re.findall(pattern_loss, text)
                if len(loss_text) > 0:
                    new_text += [loss_text[-1]]
                new_text = sorted(
                    new_text, key=lambda x: [self._get_time(x.split('- ')[0])])

                dstFile.writelines(new_text)
                logger.info('生成简报成功')
        except:
            logger.error('生成简报失败')
            logger.error(traceback.format_exc())

        return results

    def _get_time(self, date):
        return datetime.datetime.strptime(date,
                                          "%Y-%m-%d %H:%M:%S,%f ").timestamp()

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
