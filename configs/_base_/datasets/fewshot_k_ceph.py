# dataset settings
dataset_type = 'VideoFewShotDataset'
data_root = 'data/kinetics400/videos_train'
# data_root = 'data/minikinetics/videos'
ann_file_train = 'data/minikinetics/kinetics-100-replaced/train.txt'
ann_file_val = 'data/minikinetics/kinetics-100-replaced/val.txt'
ann_file_test = 'data/minikinetics/kinetics-100-replaced/test.txt'
train_labels = [
    140, 264, 167, 298, 22, 156, 70, 24, 203, 235, 252, 80, 108, 275, 109, 99,
    220, 88, 325, 258, 201, 218, 322, 31, 236, 172, 336, 389, 217, 289, 27,
    123, 48, 229, 292, 254, 382, 40, 291, 315, 97, 69, 55, 6, 16, 105, 364,
    313, 249, 330, 390, 34, 75, 262, 379, 383, 1, 180, 189, 373, 104, 193, 301,
    60
]
val_labels = [11, 306, 107, 307, 152, 286, 130, 78, 365, 166, 124, 247]
test_labels = [
    83, 230, 349, 371, 161, 133, 238, 85, 164, 84, 356, 205, 269, 87, 333, 302,
    126, 248, 93, 261, 294, 41, 159, 23
]
file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict(
        {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))
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

runner = dict(type='IterBasedRunner', max_iters=1000)
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.02)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed', warmup=None)
