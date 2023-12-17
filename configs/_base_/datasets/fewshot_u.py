# dataset settings
dataset_type = 'VideoFewShotDataset'
data_root = 'data/ucf101/videos'
ann_file_train = 'data/ucf101/fewshot_split/train.txt'
ann_file_val = 'data/ucf101/fewshot_split/val.txt'
ann_file_test = 'data/ucf101/fewshot_split/test.txt'
train_labels = [
    6, 58, 69, 55, 81, 18, 38, 60, 54, 22, 26, 51, 33, 45, 99, 15, 83, 66, 89,
    100, 63, 97, 74, 4, 77, 49, 9, 34, 27, 50, 46, 2, 90, 19, 14, 16, 8, 30,
    28, 95, 57, 5, 67, 0, 42, 61, 86, 7, 65, 53, 12, 11, 17, 10, 84, 93, 72,
    92, 98, 52, 71, 88, 3, 94, 44, 64, 56, 31, 75, 48
]
val_labels = [62, 39, 23, 79, 36, 59, 35, 78, 1, 41]
test_labels = [
    87, 68, 24, 70, 40, 76, 29, 37, 47, 43, 91, 13, 32, 25, 20, 21, 82, 80, 96,
    73, 85
]
file_client_args = dict(io_backend='disk')

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
    dict(type='Imgaug', transforms='default_1augs'),
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

runner = dict(type='IterBasedRunner', max_iters=3000)
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
