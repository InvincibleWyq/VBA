# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MYEVL',
        num_frames=8,
        backbone_name='ViT-B/16',
        backbone_type='clip',
        backbone_mode='freeze_fp16',
        decoder_num_layers=4,
        temporal_conv_layer_index=[],
        temporal_pos_embed_layer_index=[3],
        cross_frame_decoder_layer_index=[3]),
    cls_head=dict(
        type='MyEVLHead',
        num_classes=400,
        in_channels=768,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
