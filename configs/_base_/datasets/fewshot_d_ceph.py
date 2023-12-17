# dataset settings
dataset_type = 'VideoFewShotDataset'
data_root = 'data/diving48/videos'
ann_file_train = 'data/diving48/fewshot_split/train.txt'
ann_file_val = 'data/diving48/fewshot_split/val.txt'
ann_file_test = 'data/diving48/fewshot_split/test.txt'
train_labels = [
    0, 1, 2, 3, 7, 8, 10, 13, 15, 17, 20, 21, 22, 23, 24, 25, 26, 29, 32, 34,
    35, 37, 38, 41, 42, 45, 46
]
val_labels = [4, 11, 14, 16, 28, 31, 33, 36, 39, 47]
test_labels = [5, 6, 9, 12, 18, 19, 27, 40, 43, 44]

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict(
        {'data/diving48/videos': 's3://openmmlab/datasets/action/diving48'}))
# file_client_args = dict(io_backend='disk')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='CenterCrop', crop_size=(320, 320)),
    # dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['filename']),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='CenterCrop', crop_size=(320, 320)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['filename']),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    meta_samples_per_gpu=3,
    workers_per_gpu=8,
    test_dataloader=dict(meta_samples_per_gpu=1),
    train=dict(
        type='EpisodicDataset',
        num_episodes=30000,
        num_ways=5,
        num_shots=1,
        num_queries=5,
        episodes_seed=42,
        subset='train',
        train_labels=train_labels,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_root,
            pipeline=train_pipeline,
            sample_by_class=True)),
    val=dict(
        type='MetaTestDataset',
        num_episodes=100,
        num_ways=5,
        num_shots=1,
        num_queries=1,
        episodes_seed=42,
        subset='val',
        val_labels=val_labels,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_val,
            data_prefix=data_root,
            pipeline=val_pipeline,
            sample_by_class=True),
        meta_test_cfg=dict(
            num_episodes=100,
            num_ways=5,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=10, num_workers=4),
            support=dict(batch_size=5 * 1, num_workers=0),
            query=dict(batch_size=5 * 1, num_workers=0))),
    test=dict(
        type='MetaTestDataset',
        num_episodes=2000,
        num_ways=5,
        num_shots=1,
        num_queries=1,
        episodes_seed=42,
        subset='test',
        test_labels=test_labels,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_test,
            data_prefix=data_root,
            pipeline=val_pipeline,
            sample_by_class=True),
        meta_test_cfg=dict(
            num_episodes=2000,
            num_ways=5,
            # whether to cache features in fixed-backbone methods for
            # testing acceleration.
            fast_test=True,
            test_set=dict(batch_size=10, num_workers=4),
            support=dict(batch_size=5 * 1, num_workers=0),
            query=dict(batch_size=5 * 1, num_workers=0))))

runner = dict(type='IterBasedRunner', max_iters=10000)
optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
