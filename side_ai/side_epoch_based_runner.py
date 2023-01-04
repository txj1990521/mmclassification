import time
import traceback

from mmcv.runner.builder import RUNNERS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.epoch_based_runner import EpochBasedRunner

from mmcls.datasets.pipelines import Compose
from mmdet.utils.logger import get_root_logger
from side_ai.pipelines.utils_checkpoint import get_checkpoint


@RUNNERS.register_module()
class SideEpochBasedRunner(EpochBasedRunner):

    def __init__(
            self,
            save_pipeline,  # 保存模型的pipeline
            after_run_pipeline=None,  # 训练结束后的pipeline
            save_model_path=None,  # 保存模型的路径
            timestamp=None,  # 时间戳，将会成为生成的文件名，跟dataset的timestamp保持同步
            *args,
            **kwargs):
        super(SideEpochBasedRunner, self).__init__(*args, **kwargs)

        self.save_model_path = save_model_path
        if timestamp is not None:
            self.timestamp2 = timestamp
        else:
            self.timestamp2 = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.save_pipeline_compose = Compose(save_pipeline)
        self.after_run_compose = Compose(after_run_pipeline)

    def save_checkpoint(
            self,
            out_dir,
            filename_tmpl='epoch_{}.pth',
            save_optimizer=True,
            meta=None,
            create_symlink=True):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)

        checkpoint = get_checkpoint(
            self.model, optimizer=self.optimizer, meta=meta)
        checkpoint_no_optimizer = get_checkpoint(self.model, meta=meta)

        logger = get_root_logger()
        logger.propagate = False

        results = dict(
            out_dir=out_dir,
            filename=filename,
            epoch=self.epoch,
            max_epochs=self.max_epochs,
            checkpoint=checkpoint,
            checkpoint_no_optimizer=checkpoint_no_optimizer,
            save_model_path=self.save_model_path,
            timestamp=self.timestamp2,
            log_timestamp=self.timestamp)
        try:
            self.save_pipeline_compose(results)
        except:
            logger.warning(f'save_pipeline执行出现出错：\n{traceback.format_exc()}')

        self.model_path = results.get('model_path', None)
        self.out_dir = out_dir

    @master_only
    def after_run(self):
        logger = get_root_logger()
        logger.propagate = False
        results = dict(
            out_dir=self.out_dir,
            model_path=self.model_path,
            save_model_path=self.save_model_path,
            timestamp=self.timestamp2,
            log_timestamp=self.timestamp)
        try:
            self.after_run_compose(results)
        except:
            logger.warning(f'after_run执行出现出错：\n{traceback.format_exc()}')
