import io
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from glob import glob
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch.distributed
from mmcv.runner.dist_utils import master_only
from torch import distributed as dist
from tqdm import tqdm
from mmdet.datasets.builder import PIPELINES
from mmdet.utils import get_root_logger
from side_ai.pipelines.utils_labelme import copy_json_and_img, shape_to_points, _annotation

'''
主要是操作数据
'''


@PIPELINES.register_module()
# 将lablelme的keypoint数据转化为coco数据格式
class Labelme2COCOKeypoints:

    def __init__(self, bbox_full_image=False,*args, **kwargs):
        self.bbox_full_image = bbox_full_image


    def __call__(self, results, *args, **kwargs):
        category_list = results['category_list']
        category_map = results['category_map']
        json_data_list = results['json_data_list']
        ignore_labels = results['ignore_labels']
        error_file_path = results['error_file_path']
        joint_num = len(results['error_file_path'])
        logger = get_root_logger()
        logger.propagate = False

        annotations = []
        images = []

        obj_count = 0
        if (dist.is_initialized()
            and dist.get_rank() == 0) or not dist.is_initialized():
            disable = False
        else:
            disable = True
        with tqdm(json_data_list, desc='Labelme2COCOKeypoints',
                  disable=disable) as pbar:
            for idx, data in enumerate(pbar):
                filename = os.path.basename(data['imagePath'])
                height, width = data['imageHeight'], data['imageWidth']
                images.append(
                    dict(
                        id=idx, file_name=filename, height=height,
                        width=width))
                bboxes_list, keypoints_list = [], []
                boxinfo = dict(
                    mark='',
                    label='bbox',
                    points=[],
                    group_id=None,
                    shape_type='rectangle',
                    flags={})
                # pointinfo = dict(
                #     mark='',
                #     label='bbox',
                #     points=[],
                #     group_id=None,
                #     shape_type='rectangle',
                #     flags={})

                for shape in data['shapes']:
                    if shape['shape_type'] == 'point':
                        # shape['points'][0][0] = width // 2+50
                        keypoints_list.append(shape)
                    elif shape['shape_type'] == 'rectangle':  # bboxs
                        bboxes_list.append(shape)

                    if shape['label'] not in category_map:
                        logger.warning('发现未知标签', idx, shape)
                        continue
                    if category_map[shape['label']] in ignore_labels:
                        continue
                    new_points = []
                    try:
                        new_points = shape_to_points(shape, height, width)
                    except:
                        logger.error(traceback.format_exc())

                    if len(new_points) == 0:
                        logger.warning(
                            f"解析 shape 失败, {shape['label']}，图片路径为{filename}")
                        error_file_path.append(data['json_path'])
                        continue
                    category_id = category_list.index(
                        category_map[shape['label']])
                if bboxes_list == []:
                    # d = 10
                    # bbox = [[width // 2+50 - d, int(height * 0)], [width // 2+50 + d, int(height * 1)]]
                    bbox = [[0, 0], [width, height]]
                    boxinfo['points'] = bbox
                    bboxes_list = [boxinfo]

                if self.bbox_full_image:
                    bbox = [[0, 0], [width, height]]
                    boxinfo['points'] = bbox
                    bboxes_list = [boxinfo]

                _annotation(
                    annotations,
                    bboxes_list, keypoints_list, data['json_path'], 1,
                    len(keypoints_list), category_id, idx)
                obj_count += 1

        categories = [
            {
                'id': i,
                'name': x
            } for i, x in enumerate(category_list)
        ]
        results['error_file_path'] = error_file_path
        results['json_data_dic'] = dict(
            images=images, annotations=annotations, categories=categories)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadCategoryList:

    def __init__(self, ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ['屏蔽']
        self.ignore_labels = ignore_labels

    def __call__(self, results, *args, **kwargs):
        label_path = results['label_path']
        if results.get('ignore_labels', False):
            ignore_labels = results['ignore_labels'] if results[
                                                            'ignore_labels'] is not None else self.ignore_labels
        else:
            ignore_labels = self.ignore_labels
        results['ignore_labels'] = ignore_labels

        with open(label_path, encoding='utf-8') as f:
            lines = f.readlines()
        category_map = {}
        point_list = {}
        for line in lines:
            line2 = line.replace('\n', '').split()
            if len(line2) == 2:
                category_map[line2[0]] = line2[1]
            else:
                print(f'解析类别映射错误：{line}')

        category_list = list(category_map.values())

        category_list = sorted(set(category_list), key=category_list.index)
        for ignore_label in ignore_labels:
            if ignore_label in category_list:
                category_list.remove(ignore_label)

        results['category_map'] = category_map
        results['category_list'] = category_list
        point_list = []
        for m in category_list:
            if '识别框' not in m:
                point_list.append(m)

        results['point_list'] = point_list

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadPathList:

    def __call__(self, results, *args, **kwargs):
        dataset_path_list = results['dataset_path_list']
        label_path = results.get('label_path', None)

        logger = get_root_logger()
        logger.propagate = False
        logger.info(f'数据集的路径为{dataset_path_list}, 映射表的路径为{label_path}')
        path_list = []
        if isinstance(dataset_path_list, list):
            for dataset_path in dataset_path_list:
                path_list += glob(f'{dataset_path}/**/*', recursive=True)
        elif isinstance(dataset_path_list, str):
            path_list = glob(f'{dataset_path_list}/**/*', recursive=True)
        else:
            logger.error(f'数据集的路径的格式为{type(dataset_path_list)}, 请检查格式')
            raise Exception('请检查数据集的路径')

        if len(path_list) == 0:
            logger.error(f"在数据集的路径下找不到数据，请检查路径填写是否正确")
            raise Exception('路径下没有数据集，请检查路径填写是否正确')

        results['path_set'] = set([str(Path(x)) for x in path_list])
        results['json_path_list'] = [
            x for x in path_list if x.endswith('.json')
        ]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadJsonDataList:

    def __call__(self, results, *args, **kwargs):
        json_path_list = results['json_path_list']
        logger = get_root_logger()
        logger.propagate = False

        disable_progressbar = False
        if dist.is_initialized() and dist.get_rank() != 0:
            disable_progressbar = True

        if platform.system() == 'Windows':
            json_data_list = [
                mmcv.load(x)
                for x in tqdm(json_path_list, disable=disable_progressbar)
            ]
        else:
            logger.info(f'使用多进程读取数据集')
            stream = sys.stdout
            if disable_progressbar:
                stream = io.StringIO()
            json_data_list = mmcv.track_parallel_progress(
                mmcv.load, json_path_list, nproc=8, file=stream)

        for i, x in enumerate(json_data_list):
            json_path = Path(json_path_list[i])
            x['imagePath'] = str(Path(json_path.parent, x['imagePath']))
            json_data_list[i]['json_path'] = json_path_list[i]

        results['json_data_list'] = json_data_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadLabelmeDataset:

    def __call__(self, results):
        category_map = results['category_map']
        json_path_list = results['json_path_list']
        path_set = results['path_set']
        json_data_list = results['json_data_list']
        ignore_labels = results['ignore_labels']

        error_file_path = []

        logger = get_root_logger()
        logger.propagate = False

        logger.info(f'筛选图片文件存在，筛选前：{len(json_data_list)}')
        keep_index = [
            i for i, x in enumerate(json_data_list)
            if x['imagePath'] in path_set
        ]
        json_data_list = [json_data_list[x] for x in keep_index]
        json_path_list = [json_path_list[x] for x in keep_index]
        logger.info(f'筛选图片文件存在，筛选后：{len(json_data_list)}')

        all_ann_label_list = {
            y['label']
            for x in json_data_list for y in x['shapes']
        }
        if len(set(all_ann_label_list) - set(category_map.keys())):
            logger.warning(
                f'发现未知类别：{set(all_ann_label_list) - set(category_map.keys())}')
            for path, x in zip(json_path_list, json_data_list):
                if set(shape['label'] for shape in x['shapes']) - set(
                        category_map.keys()) != set():
                    categories = set(shape['label'] for shape in x['shapes'])
                    logger.warning(f'异常样本：{categories}, {path}')

                    error_file_path.append(path)

        logger.info('筛选json')
        new_data_list = [
            x for x in json_data_list
            if all(y['label'] in category_map
                   for y in x['shapes']) and not all(category_map[y['label']] in ignore_labels for y in x['shapes'])
        ]
        logger.info(f'筛选后的样本数量：{len(new_data_list)}')
        # 临时测试
        # logger.info(f'路径样例：{new_data_list[0]["imagePath"]}')

        results['json_data_list'] = new_data_list
        results['error_file_path'] = error_file_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class StatCategoryCounter:

    def __call__(self, results, *args, **kwargs):
        category_map = results['category_map']
        category_list = results['category_list']
        json_data_list = results['json_data_list']

        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"\n映射表为{category_map}\n列表为{category_list}")

        category_counter = Counter(
            category_map.get(y['label'], y['label']) for x in json_data_list
            for y in x['shapes'])
        s = '\n'
        for k, v in category_counter.most_common():
            s += f'{k}\t{v}\n'
        logger.info(f'统计类别分布{s}')

        results['category_counter'] = category_counter
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SplitData:

    def __init__(
            self, start=0, end=1, seed=42, shuffle=True, key='json_path_list'):
        self.start = start
        self.end = end
        self.seed = seed
        self.shuffle = shuffle
        self.key = key

    def __call__(self, results):
        if results.get('start', False):
            start = results['start'] if results[
                                            'start'] is not None else self.start
        else:
            start = self.start
        if results.get('end', False):
            end = results['end'] if results['end'] is not None else self.end
        else:
            end = self.end

        target_list = results[self.key]
        if self.shuffle:
            state = random.getstate()
            random.seed(self.seed)
            random.shuffle(target_list)
            random.setstate(state)
        n = len(target_list)
        start_index, end_index = int(start * n), int(end * n)
        target_list = target_list[start_index:end_index]
        results[self.key] = target_list
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CopyData:

    def __init__(self, times, key='json_data_list'):
        self.times = times
        self.key = key

    def __call__(self, results, *args, **kwargs):
        if results.get('times', False):
            times = results['times'] if results[
                                            'times'] is not None else self.times
        else:
            times = self.times
        results[self.key] *= times

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadMutilChannelImgPathList:

    def __init__(self, channels, *args, **kwargs):
        self.channels = channels

    def __call__(self, results, *args, **kwargs):
        json_data_list = results['json_data_list']
        error_file_path = results['error_file_path']
        new_json_data_list = []

        logger = get_root_logger()
        logger.propagate = False

        logger.info(f"筛选多通道图中，筛选前{len(json_data_list)}组")
        for idx, json_data in enumerate(json_data_list):
            json_path = json_data['json_path']
            if 'image_path_list' not in json_data:
                error_file_path.append(json_path)
                logger.warning(f"{json_path}没有image_path_list的键，不参与训练")
                continue
            if len(json_data['image_path_list']) != self.channels:
                error_file_path.append(json_path)
                logger.warning(
                    f"{json_path}中image_path_list的长度不等于{self.channels}，不参与训练")
                continue
            dirname = Path(json_data['imagePath']).parent
            flag = False
            for idx, image_path in enumerate(json_data['image_path_list']):
                json_data['image_path_list'][idx] = str(
                    Path(dirname, image_path))
                if not os.path.exists(json_data['image_path_list'][idx]):
                    flag = True
                    logger.info(
                        f"{json_data['json_path']}的{json_data['image_path_list'][idx]}不存在，不参与训练"
                    )
            if flag:
                continue
            json_data['imagePath'] = json_data['image_path_list']
            new_json_data_list.append(json_data)
        logger.info(f"筛选后{len(new_json_data_list)}组")
        results['json_data_list'] = new_json_data_list
        results['error_file_path'] = error_file_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SaveJson:

    def __call__(self, results, *args, **kwargs):
        save_path = results['ann_save_path']
        json_data_dic = results['json_data_dic']

        logger = get_root_logger()
        logger.info(f'保存路径{save_path}')
        logger.info(f'样本数量：{len(json_data_dic["images"])}')
        logger.info(f'标注shape数量{len(json_data_dic["annotations"])}')
        mmcv.dump(json_data_dic, save_path)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CalculateMeanAndStd:

    def __init__(self, img_scale):
        self.img_scale = img_scale

    def __call__(self, results, *args, **kwargs):
        json_data_list = results['json_data_list']

        logger = get_root_logger()
        logger.propagate = False

        logger.info('读取图像中')
        if isinstance(json_data_list[0]['imagePath'], list):
            imgs = [
                [
                    np.resize(
                        cv2.imdecode(
                            np.fromfile(path, dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED), self.img_scale)
                    for path in json_data['imagePath']
                ] for json_data in tqdm(json_data_list)
            ]
        else:
            imgs = [
                [
                    np.resize(
                        cv2.imdecode(
                            np.fromfile(
                                json_data['imagePath'], dtype=np.uint8),
                            cv2.IMREAD_UNCHANGED), self.img_scale + (3,))
                ] for json_data in tqdm(json_data_list)
            ]

        imgs = np.array(imgs)
        mean = np.mean(imgs, axis=(0, 2, 3), where=imgs > 0)
        std = np.std(imgs, axis=(0, 2, 3), where=imgs > 0)
        logger.info("计算均值和标注差中")
        logger.info(
            f"图像列表的平均值为{[round(i, 2) for i in list(mean)]}，标准差为{[round(i, 2) for i in list(std)]}"
        )

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadOKPathList:

    def __call__(self, results, *args, **kwargs):
        ok_dataset_path_list = results['ok_dataset_path_list']
        logger = get_root_logger()
        logger.propagate = False
        path_list = []
        if isinstance(ok_dataset_path_list, list):
            for ok_dataset_path in ok_dataset_path_list:
                path_list += glob(f'{ok_dataset_path}/**/*', recursive=True)
        elif isinstance(ok_dataset_path_list, str):
            path_list = glob(f'{ok_dataset_path_list}/**/*', recursive=True)
        else:
            logger.error(
                f'良品数据集的路径为{ok_dataset_path_list}，格式为{type(ok_dataset_path_list)}, 请检查格式'
            )
            return results

        if len(path_list) == 0:
            logger.error(f"良品数据集的路径{ok_dataset_path_list}下找不到数据，请检查路径填写是否正确")
            return results

        logger.info(f"良品的数量为{len(path_list)}")
        images = results['json_data_dic']['images']
        min_idx = len(images)
        for idx, path in enumerate(path_list):
            img = mmcv.imread(path, 'unchanged')
            (height, width) = img.shape[:2]
            images.append(
                dict(
                    id=min_idx + idx,
                    file_name=path,
                    height=height,
                    width=width))
        results['json_data_dic']['images'] = images
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ShuffleCocoImage:

    def __init__(self, seed=42):
        self.seed = seed

    def __call__(self, results, *args, **kwargs):
        images = results['json_data_dic']['images']
        state = random.getstate()
        random.seed(self.seed)
        random.shuffle(images)
        random.setstate(state)
        results['json_data_dic']['images'] = images
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class CopyData2Local:

    def __init__(self, target_dir='/data/公共数据缓存', run_rsync=True):
        self.target_dir = target_dir
        self.run_rsync = run_rsync

    @master_only
    def rsync(self, cmd, run_rsync=True):
        if run_rsync:
            subprocess.run(cmd.split(' '))

    def __call__(self, results, *args, **kwargs):
        src = results['dataset_path_list']
        if isinstance(src, str):
            src = [src]
        if not isinstance(src, list):
            raise Exception(f'dataset_path_list 异常：{src}')

        if results.get('target_dir', False):
            target_dir = results['target_dir']
        else:
            target_dir = self.target_dir

        logger = get_root_logger()
        logger.propagate = False
        if platform.system() == 'Windows':
            logger.warning('Windows下不支持rsync，故无法拷贝数据到本地')
            return results
        try:
            logger.info(f'开始缓存 {src} 到 {target_dir}')
            dataset_path_list = []
            for x in src:
                cmd = f"rsync -avPR --delete --chmod 777 {x} {target_dir}"
                logger.info(cmd)
                self.rsync(cmd, self.run_rsync)
                dataset_path_list.append(str(Path(target_dir + x)))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            logger.info(
                f'数据已经缓存到了{target_dir}，dataset_path_list = {dataset_path_list}'
            )
            results['dataset_path_list'] = dataset_path_list
        except:
            logger.warning(f'缓存数据失败， 使用原路径进行训练')
            logger.error(traceback.format_exc())

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class LoadXndMutilChannelImgPathList:

    def __init__(self, channels, *args, **kwargs):
        self.channels = channels

    def __call__(self, results, *args, **kwargs):
        json_data_list = results['json_data_list']
        error_file_path = results['error_file_path']
        new_json_data_list = []

        logger = get_root_logger()
        logger.propagate = False

        logger.info(f"筛选多通道图中，筛选前{len(json_data_list)}组")
        for idx, json_data in enumerate(json_data_list):
            json_path = json_data['json_path']
            if 'image_path_list' not in json_data:
                error_file_path.append(json_path)
                logger.warning(f"{json_path}没有image_path_list的键，不参与训练")
                continue
            new_image_path_list = []
            for idx, image_path in enumerate(json_data['image_path_list']):
                if not image_path.endswith(
                        'albedo.jpg') and not image_path.endswith(
                    'normal.jpg'):
                    new_image_path_list.append(image_path)
                    new_image_path_list = sorted(
                        new_image_path_list,
                        key=lambda x: int(x.split('_')[-2][-1]))
            json_data['image_path_list'] = new_image_path_list

            if len(json_data['image_path_list']) != self.channels:
                error_file_path.append(json_path)
                logger.warning(
                    f"{json_path}中image_path_list的长度不等于{self.channels}，不参与训练")
                continue
            dirname = Path(json_data['imagePath']).parent
            flag = False
            for idx, image_path in enumerate(json_data['image_path_list']):
                json_data['image_path_list'][idx] = str(
                    Path(dirname, image_path))
                if not os.path.exists(json_data['image_path_list'][idx]):
                    logger.warning(
                        f"{json_path}的存储的路径{json_data['image_path_list'][idx]}不存在，不参与训练"
                    )
                    flag = True
                    break
            if flag:
                error_file_path.append(json_path)
                continue
            json_data['imagePath'] = json_data['image_path_list']
            new_json_data_list.append(json_data)
        logger.info(f"筛选后{len(new_json_data_list)}组")
        results['json_data_list'] = new_json_data_list
        results['error_file_path'] = error_file_path
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
# 将无法参与训练的数据保存在指定路径。例如类别没有出现在映射表中，没有找到图片，json损坏等
class CopyErrorPath:

    def __init__(
            self, copy_error_file_path='/data/14-调试数据/cyf', *args, **kwargs):
        self.copy_error_file_path = copy_error_file_path

    def __call__(self, results, *args, **kwargs):
        error_file_path = results['error_file_path']
        dataset_path_list = results['dataset_path_list']
        date = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"无法用来训练的数据的长度为{len(error_file_path)}")

        dataset_commonpath = os.path.commonpath(dataset_path_list)
        for path in error_file_path:
            copy_json_and_img(
                path,
                Path(
                    self.copy_error_file_path, date,
                    Path(path).relative_to(dataset_commonpath)))
        logger = get_root_logger()
        logger.propagate = False
        logger.info(f"复制错误数据至{str(Path(self.copy_error_file_path, date))}完成")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class GenerateMmclsAnn:
    def __call__(self, results, *args, **kwargs):
        data_dir = results['data_dir']
        img_type = results['img_type']
        data_dir = str(Path(data_dir)) + '/'
        classes = results['classes']
        class2id = dict(zip(classes, range(len(classes))))
        data_dir = str(Path(data_dir)) + '/'
        dir_types = ['train', 'val', 'test']
        sub_dirs = os.listdir(data_dir)
        ann_dir = data_dir + 'meta/'
        if not os.path.exists(ann_dir):
            os.makedirs(ann_dir)
        for sd in sub_dirs:
            if sd not in dir_types:
                continue
            annotations = []
            target_dir = data_dir + sd + '/'
            for d in os.listdir(target_dir):
                class_id = str(class2id[d])
                images = glob(target_dir + d + '/*' + img_type)
                for img in images:
                    img = d + '/' + os.path.basename(img)
                    annotations.append(img + ' ' + class_id + '\n')
            annotations[-1] = annotations[-1].strip()
            with open(ann_dir + sd + '.txt', 'w') as f:
                f.writelines(annotations)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str