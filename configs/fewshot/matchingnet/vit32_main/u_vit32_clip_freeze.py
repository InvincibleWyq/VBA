# train with 5 shot, test 1-shot acc 91.92, test 5-shot acc 97.47
_base_ = [
    '../../../_base_/default_runtime.py',
    '../../../_base_/models/fewshot_vit_1.py',
    '../../../_base_/datasets/fewshot_u.py'
]

model = dict(
    neck=dict(layers_with_batch_self_attn=[]),
    cls_head=dict(temperature=100, fusion_mode=None))

# optimizer
runner = dict(type='IterBasedRunner', max_iters=2000)
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='fixed', warmup=None)

log_config = dict(interval=50)
checkpoint_config = dict(interval=50, max_keep_ckpts=1)
evaluation = dict(by_epoch=False, interval=50, metric='accuracy')
