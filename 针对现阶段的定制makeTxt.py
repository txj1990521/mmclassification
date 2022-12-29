import os
from glob import glob
from pathlib import Path


def generate_mmcls_ann(data_dir, img_type='.png'):
    data_dir = str(Path(data_dir)) + '/'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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


if __name__ == '__main__':
    data_dir = 'D:/data/MNIST/mnista_data/'
    generate_mmcls_ann(data_dir,img_type='.jpg')