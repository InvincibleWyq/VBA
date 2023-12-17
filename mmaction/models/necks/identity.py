from math import sqrt

import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class IdentityNeck(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        pass

    def forward(self, in_features):
        assert len(in_features) == 1
        in_features = in_features[0].squeeze()

        # (N, T, HxW, C) -> (N, C, T, H, W)
        in_features = in_features.permute(0, 3, 1, 2)
        in_features = in_features.reshape(*in_features.shape[:-1],
                                          int(sqrt(in_features.shape[-1])),
                                          int(sqrt(in_features.shape[-1])))

        ret_dict = {'feats': in_features}
        return ret_dict
