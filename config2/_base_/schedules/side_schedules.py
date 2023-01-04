# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100, 150])
# runner = dict(type='EpochBasedRunner', max_epochs=200)
custom_hooks = [dict(type='SonicAfterRunHook')]
save_pipeline = [
    dict(
        type='SaveEachEpochModel',
        save_each_epoch=True,
        encrypt_each_epoch=False,
        save_latest=True,
        encrypt_latest=False),
    dict(type='SaveLatestModel', encrypt=False),
]
after_run_pipeline = [
    dict(type='SaveLog', create_briefing=True),
]
runner = dict(
    type='SideEpochBasedRunner',
    save_pipeline=save_pipeline,
    after_run_pipeline=after_run_pipeline,
    max_epochs=12,
)
