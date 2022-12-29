import time
from .builder import DATASETS
from .base_dataset import BaseDataset
import os.path as osp
from os import PathLike
import numpy as np
# from mmcls.datasets.pipelines.__init__ import Compose
from mmcls.datasets.pipelines import Compose

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
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False):
        super(BaseDataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
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

