# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmaction.utils import label_wrapper
from ..builder import HEADS
from .base_fewshot import BaseFewShotHead


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


@HEADS.register_module()
class MatchingHead(BaseFewShotHead):
    """Classification head for `MatchingNet.

    <https://arxiv.org/abs/1606.04080>`_.

    Note that this implementation is without FCE(Full Context Embeddings).

    Args:
        temperature (float): The scale factor of `cls_score`.
        loss (dict): Config of training loss.
    """

    def __init__(self,
                 temperature: float = 10.0,
                 loss: Dict = dict(type='NLLLoss', loss_weight=1.0),
                 spatial_type='avg',
                 temporal_type=None,
                 fusion_mode=None,
                 fusion_ratio=0.5,
                 in_feature_dim=768,
                 *args,
                 **kwargs) -> None:
        super().__init__(loss=loss, *args, **kwargs)
        self.temperature = temperature
        self.spatial_type = spatial_type
        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        elif self.spatial_type == 'avg2d':
            self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        self.temporal_type = temporal_type

        # used in meta testing
        self.support_feats_list = []
        self.support_labels_list = []
        self.support_feats = None
        self.support_labels = None
        self.class_ids = None

        self.fusion_mode = fusion_mode
        if self.fusion_mode is not None:
            assert self.fusion_mode in ['fuse_all_atom', 'fuse_avg_atom']
            self.fusion_ratio = fusion_ratio
            self.in_feature_dim = in_feature_dim

            self.self_attn = nn.MultiheadAttention(
                in_feature_dim, num_heads=12, batch_first=True)
            self.norm1 = nn.LayerNorm(in_feature_dim)
            self.norm2 = nn.LayerNorm(in_feature_dim)
            self.mlp1 = nn.Sequential(
                OrderedDict([
                    ('fc1', nn.Linear(in_feature_dim, 4 * in_feature_dim)),
                    ('act', QuickGELU()),
                    ('fc2', nn.Linear(4 * in_feature_dim, in_feature_dim)),
                ]))

    def forward_train(self, support_feats: Tensor, support_labels: Tensor,
                      query_feats: Tensor, query_labels: Tensor,
                      **kwargs) -> Dict:
        """Forward training data.

        Args:
            support_feats (Tensor): Features of support data with shape (N, C).
            support_labels (Tensor): Labels of support data with shape (N).
            query_feats (Tensor): Features of query data with shape (N, C).
            query_labels (Tensor): Labels of query data with shape (N).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        support_labels = support_labels.squeeze()
        query_labels = query_labels.squeeze()
        class_ids = torch.unique(support_labels).cpu().tolist()

        cosine_distance = self.get_cosine_distance(support_feats, query_feats)\
            if (self.fusion_mode is None) else self.get_fusion_cosine_distance(
            support_feats, query_feats)

        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, support_labels == class_id].mean(1, keepdim=True)
            for class_id in class_ids
        ],
                           dim=1).log()
        query_labels = label_wrapper(query_labels, class_ids)
        losses = self.loss(scores, query_labels)
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> None:
        """Forward support data in meta testing."""
        self.support_feats_list.append(x)
        self.support_labels_list.append(gt_label)

    def forward_query(self, x: Tensor, **kwargs) -> List:
        """Forward query data in meta testing."""
        support_feats = self.support_feats

        cosine_distance = self.get_cosine_distance(support_feats, x)\
            if (self.fusion_mode is None) else self.get_fusion_cosine_distance(
            support_feats, x)

        scores = F.softmax(cosine_distance * self.temperature, dim=-1)
        scores = torch.cat([
            scores[:, self.support_labels == class_id].mean(1, keepdim=True)
            for class_id in self.class_ids
        ],
                           dim=1)
        pred = F.softmax(scores, dim=1)
        pred = list(pred.detach().cpu().numpy())
        return pred

    def get_cosine_distance(self, support_feats: Tensor, query_feats: Tensor):
        if self.spatial_type == 'avg':
            support_feats = self.avg_pool(support_feats)
            query_feats = self.avg_pool(query_feats)
        if self.spatial_type == 'avg2d':
            support_feats = self.avg_pool(support_feats)
            query_feats = self.avg_pool(query_feats)
            support_feats = support_feats.squeeze().permute(0, 2, 1)
            query_feats = query_feats.squeeze().permute(0, 2, 1)
        support_feats = support_feats.squeeze()
        query_feats = query_feats.squeeze()

        if self.temporal_type == 'avg' and len(support_feats.shape) > 2:
            cosine_distance = torch.einsum('qtd,std->tqs',
                                           F.normalize(query_feats, dim=2),
                                           F.normalize(support_feats, dim=2))

            cosine_distance = cosine_distance.mean(dim=0)
            # cosine_distance = cosine_distance[0]
        else:
            cosine_distance = torch.einsum('qd,sd->qs',
                                           F.normalize(query_feats, dim=1),
                                           F.normalize(support_feats, dim=1))
        return cosine_distance

    def get_fusion_cosine_distance(self, support_feats: Tensor,
                                   query_feats: Tensor):
        if self.spatial_type == 'avg':
            support_feats = self.avg_pool(support_feats)
            query_feats = self.avg_pool(query_feats)
        if self.spatial_type == 'avg2d':
            support_feats = self.avg_pool(support_feats)
            query_feats = self.avg_pool(query_feats)
            support_feats = support_feats.squeeze().permute(0, 2, 1)
            query_feats = query_feats.squeeze().permute(0, 2, 1)
        support_feats = support_feats.squeeze()
        query_feats = query_feats.squeeze()

        if support_feats.dim() == 2:
            support_feats = support_feats.unsqueeze(1)
        if query_feats.dim() == 2:
            query_feats = query_feats.unsqueeze(1)

        num_query = query_feats.shape[0]
        num_support = support_feats.shape[0]
        num_atom = query_feats.shape[1]

        support_feats_rpt = support_feats.repeat(num_query, 1, 1, 1)
        query_feats_rpt = query_feats.unsqueeze(1)
        fusion_feats = torch.cat([support_feats_rpt, query_feats_rpt], dim=1)
        if self.fusion_mode == 'fuse_all_atom':
            fusion_feats = fusion_feats.flatten(start_dim=1, end_dim=2)
        elif self.fusion_mode == 'fuse_avg_atom':
            fusion_feats = fusion_feats.mean(dim=2)
        fusion_feats = fusion_feats + self.self_attn(
            self.norm1(fusion_feats), self.norm1(fusion_feats),
            fusion_feats)[0]
        fusion_feats = fusion_feats + self.mlp1(self.norm2(fusion_feats))

        fusion_feats = fusion_feats.view(num_query, num_support + 1, -1,
                                         self.in_feature_dim)
        support_feats_rpt = self.fusion_ratio * fusion_feats[:, :-1, :, :] + (
            1 - self.fusion_ratio) * support_feats_rpt
        query_feats_rpt = self.fusion_ratio * fusion_feats[:, -1:, :, :] + (
            1 - self.fusion_ratio) * query_feats_rpt

        # fusion_feats = fusion_feats.view(num_query, num_support + 1, -1,
        #                                  self.in_feature_dim).expand(
        #                                      -1, -1, num_atom, -1)
        # support_feats_rpt = torch.cat(
        #     [support_feats_rpt, fusion_feats[:, :-1, :, :]], dim=3)
        # query_feats_rpt = torch.cat(
        #     [query_feats_rpt, fusion_feats[:, -1:, :, :]], dim=3)

        if self.temporal_type == 'avg' and len(support_feats.shape) > 2:
            cosine_distance = torch.einsum(
                'qitd,qjtd->tqij', F.normalize(query_feats_rpt, dim=3),
                F.normalize(support_feats_rpt, dim=3))
            cosine_distance = cosine_distance.squeeze(dim=2).mean(dim=0)
        else:
            cosine_distance = torch.einsum('qik,qjk->qij',
                                           F.normalize(query_feats, dim=2),
                                           F.normalize(support_feats, dim=2))
            cosine_distance = cosine_distance.squeeze()
        return cosine_distance

    def before_forward_support(self) -> None:
        """Used in meta testing.

        This function will be called before model forward support data during
        meta testing.
        """
        # reset saved features for testing new task
        self.support_feats_list.clear()
        self.support_labels_list.clear()
        self.support_feats = None
        self.support_labels = None
        self.class_ids = None

    def before_forward_query(self) -> None:
        """Used in meta testing.

        This function will be called before model forward query data during
        meta testing.
        """
        self.support_feats = torch.cat(self.support_feats_list, dim=0)
        self.support_labels = torch.cat(self.support_labels_list, dim=0)
        self.class_ids, _ = torch.unique(self.support_labels).sort()
        if max(self.class_ids) + 1 != len(self.class_ids):
            warnings.warn(
                f'the max class id is {max(self.class_ids)}, while '
                f'the number of different number of classes is '
                f'{len(self.class_ids)}, it will cause label '
                f'mismatching problem.', UserWarning)
