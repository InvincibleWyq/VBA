# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
from torch import Tensor

from ..builder import HEADS
from .base_fewshot import BaseFewShotHead


@HEADS.register_module()
class CollaborateHead(BaseFewShotHead):

    def __init__(self,
                 loss: Dict = dict(
                     type='MarginRankingLoss', loss_weight=1.0, margin=3.2),
                 var_loss_weight=1.0,
                 *args,
                 **kwargs) -> None:
        super().__init__(loss=loss, *args, **kwargs)
        self.var_loss_weight = var_loss_weight

    def forward_train(self, support_feats: Tensor, support_labels: Tensor,
                      query_feats: Tensor, query_labels: Tensor,
                      **kwargs) -> Dict:
        """Forward training data."""

        attn_scores = torch.cat((support_feats, query_feats), dim=0)
        N, num_atoms, T = attn_scores.size()

        time_idx = torch.arange(1, T + 1).view(T, 1)
        time_idx = time_idx.float().to(attn_scores.device)

        cluster_mean = torch.einsum('nat,tk->nak', attn_scores, time_idx)

        # repeat time_idx from T x 1 to N x num_atoms x T
        time_idx = time_idx.repeat(N, num_atoms, 1, 1).squeeze(-1)
        var = attn_scores.mul(torch.abs(time_idx - cluster_mean)).sum(-1)
        cluster_mean = cluster_mean.squeeze(-1)
        hinge_loss = 0.
        for i in range(num_atoms - 1):
            cluster_mean_former = cluster_mean[:, i]
            cluster_mean_latter = cluster_mean[:, i + 1]
            hinge_loss += self.compute_loss(
                cluster_mean_latter, cluster_mean_former,
                torch.ones_like(cluster_mean_former))
            if i == 0:
                hinge_loss += self.compute_loss(
                    cluster_mean_former, torch.ones_like(cluster_mean_former),
                    torch.ones_like(cluster_mean_former))
            if i == (num_atoms - 2):
                hinge_loss += self.compute_loss(
                    torch.ones_like(cluster_mean_former) * T,
                    cluster_mean_latter, torch.ones_like(cluster_mean_former))

        hinge_loss = hinge_loss / (num_atoms - 1)
        var_loss = self.var_loss_weight * var.mean()
        losses = {'loss': hinge_loss + var_loss}
        return losses

    def forward_support(self, x: Tensor, gt_label: Tensor, **kwargs) -> None:
        return None

    def forward_query(self, x: Tensor, **kwargs) -> None:
        return None

    def before_forward_support(self) -> None:
        return None

    def before_forward_query(self) -> None:
        return None
