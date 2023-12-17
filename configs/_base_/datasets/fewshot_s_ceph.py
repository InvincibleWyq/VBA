# dataset settings
dataset_type = 'VideoFewShotDataset'
data_root = 'data/sthv2/videos'
# data_root = 'data/minissv2/videos/small'
ann_file_train = 'data/minissv2/minissv2_small/train.txt'
ann_file_val = 'data/minissv2/minissv2_small/val.txt'
ann_file_test = 'data/minissv2/minissv2_small/test.txt'
train_labels = [
    36, 23, 63, 121, 74, 69, 141, 129, 173, 156, 35, 136, 84, 0, 76, 33, 117,
    22, 75, 146, 88, 21, 26, 55, 56, 153, 20, 12, 128, 159, 32, 19, 34, 106,
    54, 171, 3, 102, 98, 111, 49, 38, 104, 17, 67, 137, 31, 125, 166, 105, 4,
    48, 30, 64, 59, 58, 1, 78, 160, 144, 100, 87, 92, 99
]
val_labels = [113, 70, 89, 18, 60, 62, 41, 27, 162, 120, 157, 155]
test_labels = [
    44, 172, 46, 123, 24, 122, 107, 169, 81, 110, 52, 13, 119, 51, 134, 140,
    10, 143, 95, 109, 148, 170, 53, 101
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
    workers_per_gpu=8,
    test_dataloader=dict(meta_samples_per_gpu=1),
    train=dict(
        type='EpisodicDataset',
        num_episodes=5000,
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

runner = dict(type='IterBasedRunner', max_iters=2000)
optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
