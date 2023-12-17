# Copyright (c) OpenMMLab. All rights reserved.
# Adopted from CLIP, MIT License, Copyright (c) 2021 OpenAI.
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES


def load_weights_clip(load_path: str) -> Dict[str, torch.Tensor]:
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())
    dst_state_dict = src_state_dict
    return dst_state_dict


weight_loader_fn_dict = {
    'clip': load_weights_clip,
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed
        # after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2d(stride)),
                             ("0",
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ("1", nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)


class MyTempModule(nn.Module):
    """MyTempModule.
    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
    """

    def __init__(self,
                 d_model: int,
                 dw_reduction: float = 1.5,
                 pos_kernel_size: int = 3) -> None:
        super().__init__()

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's
    but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1,
        with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions,
        where an avgpool is prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.
    """

    def __init__(self,
                 layers=(3, 4, 6, 3),
                 output_dim=1024,
                 heads=32,
                 input_resolution=224,
                 width=64,
                 blocks_with_temporal: list = [13, 14, 15]):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim,
                                        heads, output_dim)

        self.temporal_modules = nn.ModuleDict()
        for i in blocks_with_temporal:
            if i <= layers[0]:
                temp_embed_dim = embed_dim // 8
            elif i <= layers[0] + layers[1]:
                temp_embed_dim = embed_dim // 4
            elif i <= layers[0] + layers[1] + layers[2]:
                temp_embed_dim = embed_dim // 2
            else:
                temp_embed_dim = embed_dim
            self.temporal_modules[str(i)] = MyTempModule(temp_embed_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # x = x.type(self.conv1.weight.dtype)

        N = x.shape[0]

        def _convert_to_2d(x: torch.Tensor) -> torch.Tensor:
            """(N, C, T, H, W) -> (N x T, C, H, W)"""
            x = x.permute((0, 2, 1, 3, 4))
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            return x

        def _convert_to_3d(x: torch.Tensor) -> torch.Tensor:
            """(N x T, C, H, W) -> (N, C, T, H, W)"""
            x = x.reshape(N, -1, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute((0, 2, 1, 3, 4))
            return x

        x = _convert_to_2d(x)

        x = stem(x)

        all_features = []
        block_id = 0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if str(block_id) in self.temporal_modules:
                    x = _convert_to_3d(x)
                    x = self.temporal_modules[str(block_id)](x)
                    x = _convert_to_2d(x)

                x = block(x)

                x_duplicate = _convert_to_3d(x.clone()).flatten(3).permute(
                    0, 2, 3, 1)
                all_features.append(x_duplicate)  # (N, T, H*W, C)
                block_id += 1

        return all_features


resnet_presets = {
    'RN50':
    dict(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=32,
        input_resolution=224,
        width=64),
    'RN101':
    dict(
        layers=(3, 4, 23, 3),
        output_dim=512,
        heads=32,
        input_resolution=224,
        width=64,
    ),
    'RN50x4':
    dict(
        layers=(4, 6, 10, 6),
        output_dim=640,
        heads=40,
        input_resolution=288,
        width=80,
    ),
    'RN50x16':
    dict(
        layers=(6, 8, 18, 8),
        output_dim=768,
        heads=48,
        input_resolution=384,
        width=96,
    ),
    'RN50x64':
    dict(
        layers=(3, 15, 36, 10),
        output_dim=1024,
        heads=64,
        input_resolution=448,
        width=128,
    )
}


@BACKBONES.register_module()
class MyResNet(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'RN50',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'finetune_temporal',
        decoder_num_layers: int = 4,
        blocks_with_temporal: list = [12, 13, 14, 15],
    ):
        super().__init__()
        self.num_frames = num_frames
        if blocks_with_temporal is None:
            blocks_with_temporal = []
        self._create_backbone(backbone_name, backbone_type, backbone_path,
                              backbone_mode, blocks_with_temporal)
        self.decoder_num_layers = decoder_num_layers

    def init_weights(self):
        pass

    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
        blocks_with_temporal: list,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = ModifiedResNet(
            **resnet_presets[backbone_name],
            blocks_with_temporal=blocks_with_temporal)

        assert backbone_mode in ['finetune', 'finetune_temporal', 'freeze']

        backbone.load_state_dict(
            state_dict, strict=True if backbone_mode == 'freeze' else False)
        # weight_loader_fn is expected to strip unused parameters

        if backbone_mode == 'finetune':
            self.backbone = backbone
        elif backbone_mode == 'finetune_temporal':
            for name, param in backbone.named_parameters():
                if 'temporal' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            self.backbone = backbone

        return resnet_presets[backbone_name]

    def forward(self, x: torch.Tensor):
        x = x.squeeze()
        NT, C, H, W = x.shape
        T = self.num_frames
        N = NT // T
        # (NT, C, H, W) -> (N, C, T, H, W)
        x = x.reshape(N, T, C, H, W).permute(0, 2, 1, 3, 4)

        features = self.backbone(x)[-self.decoder_num_layers:]
        # a list of (N, T, L, C)
        # (N, T, 49, 2048) for RN50
        return features
