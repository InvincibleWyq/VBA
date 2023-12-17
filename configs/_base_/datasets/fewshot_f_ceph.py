# dataset settings
dataset_type = 'VideoFewShotDataset'
data_root = 'data/sthv2/videos'
# data_root = 'data/minissv2/videos/full'
ann_file_train = 'data/minissv2/minissv2_full/train.txt'
ann_file_val = 'data/minissv2/minissv2_full/val.txt'
ann_file_test = 'data/minissv2/minissv2_full/test.txt'
train_labels = [
    73, 112, 23, 65, 141, 156, 138, 124, 91, 103, 169, 57, 22, 61, 6, 88, 56,
    16, 52, 99, 9, 68, 147, 5, 45, 159, 149, 19, 34, 37, 14, 171, 3, 102, 111,
    168, 49, 38, 25, 157, 140, 125, 114, 29, 10, 166, 143, 137, 95, 82, 93,
    132, 66, 60, 165, 170, 59, 43, 78, 8, 100, 92, 32, 145
]
val_labels = [128, 50, 70, 105, 63, 97, 28, 123, 146, 31, 153, 20]
test_labels = [
    158, 94, 89, 121, 47, 62, 71, 129, 72, 79, 136, 126, 0, 11, 76, 7, 12, 13,
    54, 154, 67, 86, 30, 148
]
file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict({'data/sthv2': 's3://openmmlab/datasets/action/sthv2'}))
# file_client_args = dict(io_backend='disk')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.875, 0.75, 0.66),
    #     random_crop=False,
    #     max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    # dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(
    #     type='RandomErasing',
    #     erase_prob=0.25,
    #     max_area_ratio=0.33,
    #     fill_color=(0, 0, 0)),
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
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['filename']),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    meta_samples_per_gpu=3,
    workers_per_gpu=25,
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
