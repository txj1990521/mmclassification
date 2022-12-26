_base_ = [
    '../_base_/models/mobilenet_v3_small_cifar.py',
    '../_base_/datasets/数字测试datasets.py',
    '../_base_/schedules/cifar10_bs128_数字测试.py',
    '../_base_/default_runtime.py'
]

lr_config = dict(policy='step', step=[120, 170])
runner = dict(type='EpochBasedRunner', max_epochs=200)

