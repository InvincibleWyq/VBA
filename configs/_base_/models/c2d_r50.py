# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C2D',
        depth=50,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        norm_eval=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))
