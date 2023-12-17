# model settings
model = dict(
    type='BaseFewShotMetricRecognizer',
    backbone=dict(
        type='EVL',
        num_frames=8,
        backbone_name='ViT-B/32',
        backbone_path='data/clip_weight/ViT-B-32.pt',
        backbone_type='clip',
        backbone_mode='finetune_temporal',
        decoder_num_layers=4,
        layers_with_temporal_before_attn=[8, 9, 10, 11],
        layers_with_temporal_before_ffn=[]),
    neck=dict(
        type='EVLDecoder',
        num_frames=8,
        num_layers=4,
        layers_with_batch_self_attn=[],
        in_feature_dim=768,
        qkv_dim=768,
        num_heads=12,
        num_self_attn_heads=12,
        mlp_factor=4.0,
        enable_temporal_conv=True,
        enable_temporal_pos_embed=True,
    ),
    cls_head=dict(
        type='MatchingHead',
        temperature=10,
        spatial_type=None,
        temporal_type='avg'),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob',
                  num_frames=8))  # for tsn-style sampling, add num_frames

fp16 = dict(loss_scale='dynamic')

pin_memory = True
use_infinite_sampler = True

# only valid when using single-gpu, will be ignored by slurm_train.sh
gpu_ids = range(1)
