import time
_base_ = [
    # '/home/txj/mmclassification/config2/_base_/datasets/base_side_datasets.py',
    # '/home/txj/mmclassification/configs/_base_/schedules/cifar10_bs128_数字测试.py',
    # '/home/txj/mmclassification/configs/_base_/default_runtime.py'
    './_base_/datasets/base_side_datasets.py',
    './_base_/schedules/side_schedules.py',
    './../configs/_base_/default_runtime.py'
]
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
dataset_path = '/mnt/AIData/txj/data/十分区数据/train'
val_path = '/mnt/AIData/txj/data/十分区数据/val'
project_name = '十分区数据'
save_model_path = '/home/txj/model'
dataset_path_list = [f'{dataset_path}']
dataset_val_path_list = [f'{val_path}']
label_path = '/mnt/AIData/txj/data/十分区数据/' + 'label.ini'
checkpoint_config = dict(interval=10)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
data = dict(
    persistent_workers=True,
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        label_path=label_path,
        dataset_path_list=dataset_path_list,
        data_prefix=dataset_path_list,
    ),
    val=dict(
        label_path=label_path,
        dataset_path_list=dataset_val_path_list,
        data_prefix=dataset_val_path_list,
    ),
    test=dict(
        label_path=label_path,
        dataset_path_list=dataset_val_path_list,
        data_prefix=dataset_val_path_list,
    )
)
evaluation = dict(interval=1, metric='accuracy')
lr_config = dict(policy='step', step=[120, 170])
# runner = dict(type='EpochBasedRunner', max_epochs=200)
runner = dict(
    save_model_path=f"{save_model_path}/{project_name}",
    timestamp=timestamp,
    max_epochs=200)
# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
