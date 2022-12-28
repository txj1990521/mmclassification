import time
from .builder import DATASETS
from .base_dataset import BaseDataset
import os.path as osp
from os import PathLike
import numpy as np
# from mmcls.datasets.pipelines.__init__ import Compose
from mmdet.datasets.pipelines import Compose

# -*- coding: utf-8 -*-
def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path
@DATASETS.register_module()
class MyDataset(BaseDataset):
    CLASSES = None
    def __init__(self,
                 dataset_path_list,  # 数据集路径 string，list
                 init_pipeline,# 数据预处理的pipeline
                 label_path,# 映射表路径
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 timestamp=None,  # 时间戳，将会成为生成的文件名
                 start=None,  # 数据的开始，0~1
                 end=None,  # 数据的结束，0~1
                 times=None,  # 重复次数
                 ignore_labels=None,  # 屏蔽的label，默认为['屏蔽']
                 test_mode=False):
        super(BaseDataset, self).__init__()
        self.data = dict(
            dataset_path_list=dataset_path_list,
            label_path=label_path,
            start=start,
            end=end,
            times=times,
            ignore_labels=ignore_labels)
        compose = Compose(init_pipeline)
        compose(self.data)
        self.data_prefix = expanduser(data_prefix)
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.ann_file = expanduser(ann_file)
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()
    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos

