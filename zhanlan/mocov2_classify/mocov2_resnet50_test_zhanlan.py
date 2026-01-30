_base_ = [
    './mocov2_resnet50_8xb32-coslr-200e_in1k_zhanlan.py',  # 复用你的模型定义
]

# 覆盖：测试数据
data_root = r'D:/zhanlan/Classify/split_data/'
metainfo = dict(classes=['格子', '条纹', '正常'])

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

test_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=False,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='CustomDataset',          # ✅ 推荐用 CustomDataset，避免 ImageNet 1000 类
        data_root=data_root,
        data_prefix='test',            # ✅ 指向 test/
        metainfo=metainfo,
        with_label=False,              # ✅ 自监督导出特征/预测，不需要 label
        pipeline=test_pipeline,
    )
)

# ✅ test 三件套必须齐全
test_cfg = dict(type='TestLoop')
test_evaluator = []  # 不做指标评估时就用空列表（最稳）
