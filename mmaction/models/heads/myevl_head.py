# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class MyEVLHead(BaseHead):
    """Classification head for MyEVL.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            temporal_type=None,
            temporal_frames=1,  # will be ignored if temporal_type is None
            dropout_ratio=0.5,
            init_std=0.01,
            **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        if temporal_type is not None:
            assert temporal_type in ['avg_score', 'concat_feat', 'avg_feat']

        self.spatial_type = spatial_type
        self.temporal_type = temporal_type
        self.dropout_ratio = dropout_ratio
        self.temporal_frames = temporal_frames
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.temporal_type == 'avg_score':
            self.fc_list = nn.ModuleList([
                nn.Linear(self.in_channels, self.num_classes)
                for _ in range(self.temporal_frames)
            ])
        elif self.temporal_type == 'concat_feat':
            self.fc_list = nn.ModuleList([
                nn.Linear(self.in_channels * self.temporal_frames,
                          self.num_classes)
            ])
        elif self.temporal_type == 'avg_feat':
            self.fc_list = nn.ModuleList(
                [nn.Linear(self.in_channels, self.num_classes)])
        else:
            self.fc_list = nn.ModuleList(
                [nn.Linear(self.in_channels, self.num_classes)])
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for i, _ in enumerate(self.fc_list):
            normal_init(self.fc_list[i], std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if self.temporal_type == 'concat_feat':
            # [N, temporal_frames, in_channels]
            # -> [N, temporal_frames * in_channels]
            x = x.view(x.size(0), -1)
        elif self.temporal_type == 'avg_feat':
            # [N, temporal_frames, in_channels] -> [N, in_channels]
            x = x.mean(dim=1)

        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)

        if self.temporal_type == 'avg_score':
            cls_score = torch.stack(
                [fc(x[:, i, :]) for i, fc in enumerate(self.fc_list)],
                dim=1).mean(dim=1)

        else:
            x = x.view(x.shape[0], -1)
            cls_score = self.fc_list[0](x)

        # [N, num_classes]
        return cls_score
