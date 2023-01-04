# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

from mmcls.apis import inference_model, init_model, show_result_pyplot

# dataset_path = f'D:/data/MNIST/mnista_data/val/0/0.1.jpg'
dataset_path = f'D:/data/胃壁地图数据v2.0_train/胃壁地图数据v2.0/十分区/1.食管-管腔/'
# dataset_path = f'/data/txj/Train/test.jpg'
file_root = dataset_path  # 当前文件夹下的所有图片
# Run_config = "configs/mobilenet_v3/mobilenet-v3-small_8xb16_数字.py"
Run_config = 'config2/数字测试_具有txt的.py'
# class_checkpoint = 'work_dirs/mobilenet-v3-small_8xb16_数字/epoch_200.pth'
class_checkpoint = 'work_dirs/数字测试_具有txt的/epoch_200.pth'


def main():
    global Run_config, class_checkpoint, file_root
    parser = ArgumentParser()
    parser.add_argument('--img', type=str,
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

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    print(mmcv.dump(result, file_format='json', indent=4))
    if args.show:
        show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
