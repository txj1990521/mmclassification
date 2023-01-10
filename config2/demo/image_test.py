# encoding:utf-8
import sys
import mmcv
import os
import os.path as osp
import time
import logging
from argparse import ArgumentParser
from glob import glob
from mmdet.utils import get_root_logger
from tqdm import tqdm
from mmcls.apis import inference_model, init_model
from sklearn.metrics import confusion_matrix

dataset_path = f'Z:/txj/data/十分区数据/val2'
file_root = dataset_path  # 当前文件夹下的所有图片
Run_config = 'config2/数字测试_具有txt的.py'
class_checkpoint = 'work_dirs/数字测试_具有txt的/只训练训练集的权重.pth'
goodnum = 0
badnum = 0
totalnum = 0


def main():
    mmcv.utils.logging.logger_initialized = {}
    global Run_config, class_checkpoint, file_root, goodnum, badnum, totalnum
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default=file_root, help='Image file')
    parser.add_argument('--config', type=str, default=Run_config, help='Config file')
    parser.add_argument('--checkpoint', type=str, default=class_checkpoint, help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    dir = os.path.dirname(__file__)
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_path = osp.join(os.path.dirname(dir),
                        'test/' + Run_config.split('/')[-1].replace('.py', '') + '/' + str(current_time) + 'test.txt')
    print('日志路径：' + log_path)
    log_path.replace('\\', '/')
    if not os.path.exists(log_path):
        open(log_path, "w", encoding='utf-8')
    logging.basicConfig(level=logging.DEBUG, filename=log_path,
                        filemode='a', format='%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__)
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    path_list = []
    if isinstance(file_root, list):
        for file_root in args.img_path:
            path_list += glob(f'{file_root}/**/*', recursive=True)
    elif isinstance(args.img_path, str):
        path_list = glob(f'{args.img_path}/**/*', recursive=True)
    badpath_list = []
    print('识别的图片路径:' + dataset_path)
    logger.info('识别的图片路径:' + dataset_path)
    y_true = []
    y_pred = []
    for imgpath in tqdm(path_list, desc='读取进度'):
        if imgpath.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            totalnum += 1
            result = inference_model(model, imgpath)
            # 细分类
            # truthlable = imgpath.replace('\\', '/').split('/')[-2]
            # currentlabel = result['pred_class']
            # 粗分类
            truthlable = imgpath.replace('\\', '/').split('/')[-2].split('-')[0].split('.')[-1]
            currentlabel = result['pred_class'].split('-')[0].split('.')[-1]
            y_true.append(truthlable)
            y_pred.append(currentlabel)
            if currentlabel == truthlable:
                goodnum += 1
            else:
                badpath_list.append(imgpath + ' ' + '实际预测:' + currentlabel)
                badnum += 1
    matrix_array = confusion_matrix(y_true, y_pred)
    print('总数:' + str(totalnum))
    logger.info('总数:' + str(totalnum))
    print('准确率:' + str(goodnum / totalnum))
    logger.info('正确数量:' + str(goodnum))
    logger.info('准确率:' + str(goodnum / totalnum))
    print('错误率:' + str(badnum / totalnum))
    logger.info('错误数量:' + str(badnum))
    logger.info('错误率:' + str(badnum / totalnum))
    for y in badpath_list:
        print('错误的识别名称:' + y)
        logger.warning('错误的识别名称:' + y)
    sys.exit()


if __name__ == '__main__':
    main()
