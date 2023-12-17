# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class MarginRankingLoss(BaseWeightedLoss):
    """Margin Ranking Loss."""

    def __init__(self, loss_weight=1.0, margin=3.2):
        super().__init__(loss_weight=loss_weight)
        self.margin = margin

    def _forward(self, input1, input2, target):
        """Forward function."""
        loss = F.margin_ranking_loss(
            input1, input2, target, margin=self.margin)
        return loss
