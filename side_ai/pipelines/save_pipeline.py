import os
import os.path as osp

from torch import distributed as dist

from mmcls.datasets.builder import PIPELINES
from mmdet.utils import get_root_logger
from side_ai.pipelines.utils_checkpoint import save_checkpoint_original
from side_ai.pipelines.utils_crypto import save_checkpoint as save_checkpoint_encrypted


@PIPELINES.register_module()
class SaveEachEpochModel:

    def __init__(
            self,
            save_each_epoch=True,
            encrypt_each_epoch=False,
            save_latest=True,
            encrypt_latest=False):
        self.save_each_epoch = save_each_epoch
        self.encrypt_each_epoch = encrypt_each_epoch
        self.save_latest = save_latest
        self.encrypt_latest = encrypt_latest

    def __call__(self, results, *args, **kwargs):
        filename = results['filename']
        out_dir = results['out_dir']
        checkpoint = results['checkpoint']
        checkpoint_no_optimizer = results['checkpoint_no_optimizer']
        timestamp = results['timestamp']

        filepath = osp.join(out_dir, filename)
        encrypted_filepath = osp.join(
            out_dir, filename.replace('.pth', '.cpth'))

        if self.save_each_epoch:
            if self.encrypt_each_epoch:
                save_checkpoint_encrypted(
                    encrypted_filepath, checkpoint_no_optimizer)
            else:
                save_checkpoint_original(filepath, checkpoint)

        if self.save_latest:
            filepath = osp.join(out_dir, f'{timestamp}.pth')
            encrypted_filepath = osp.join(out_dir, f'{timestamp}.cpth')
            if self.encrypt_latest:
                save_checkpoint_encrypted(
                    encrypted_filepath, checkpoint_no_optimizer)
            else:
                save_checkpoint_original(filepath, checkpoint_no_optimizer)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SaveLatestModel:

    def __init__(self, encrypt=False):
        self.encrypt = encrypt

    def __call__(self, results, *args, **kwargs):
        epoch = results['epoch']
        max_epochs = results['max_epochs']
        save_model_path = results['save_model_path']
        checkpoint_no_optimizer = results['checkpoint_no_optimizer']
        timestamp = results['timestamp']

        logger = get_root_logger()
        logger.propagate = False

        # 最后一代保存模型进特定路径
        if epoch == max_epochs - 1:
            if save_model_path is None:
                logger.warning('没有设置保存模型路径的参数，将无法进行保存最终模型的操作')
                return results
            # 分布式只保存一次模型，非分布式保存模型
            if (dist.is_initialized()
                    and dist.get_rank() == 0) or not dist.is_initialized():
                try:
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path, exist_ok=True)
                    filepath = osp.join(save_model_path, f'{timestamp}.pth')
                    encrypted_filepath = osp.join(
                        save_model_path, f'{timestamp}.cpth')
                    if self.encrypt:
                        save_checkpoint_encrypted(
                            encrypted_filepath, checkpoint_no_optimizer)
                        results['model_path'] = encrypted_filepath
                        logger.info(f"保存的模型为加密模型, 模型保存在 {encrypted_filepath}")
                    else:
                        save_checkpoint_original(
                            filepath, checkpoint_no_optimizer)
                        results['model_path'] = filepath
                        logger.info(f"保存的模型为不加密模型, 模型保存在 {filepath}")
                except:
                    logger.warning(
                        f'保存最后一代模型失败，请检查路径{os.path.abspath(save_model_path)}')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
