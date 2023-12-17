# https://github.com/OpenGVLab/efficient-video-recognition

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..builder import BACKBONES


def load_weights_clip(load_path: str) -> Dict[str, torch.Tensor]:
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}

    dst_state_dict['cls_token'] = src_state_dict['class_embedding']
    dst_state_dict['pos_embed'] = src_state_dict['positional_embedding']
    dst_state_dict['conv1.weight'] = src_state_dict['conv1.weight']
    dst_state_dict['ln_pre.weight'] = src_state_dict['ln_pre.weight']
    dst_state_dict['ln_pre.bias'] = src_state_dict['ln_pre.bias']

    block_idx = 0
    while True:
        src_prefix = 'transformer.resblocks.%d.' % block_idx
        dst_prefix = 'blocks.%d.' % block_idx

        src_block_state_dict = dict((k[len(src_prefix):], v)
                                    for k, v in src_state_dict.items()
                                    if k.startswith(src_prefix))
        if len(src_block_state_dict) == 0:
            break

        dst_block_state_dict = {}
        feat_dim = src_block_state_dict['ln_1.weight'].size(0)

        for i, dst_name in enumerate(('q', 'k', 'v')):
            dst_block_state_dict['attn.%s_proj.weight' %
                                 dst_name] = src_block_state_dict[
                                     'attn.in_proj_weight'][feat_dim *
                                                            i:feat_dim *
                                                            (i + 1)]
            dst_block_state_dict['attn.%s_proj.bias' %
                                 dst_name] = src_block_state_dict[
                                     'attn.in_proj_bias'][feat_dim *
                                                          i:feat_dim * (i + 1)]

        dst_block_state_dict['attn.out_proj.weight'] = src_block_state_dict[
            'attn.out_proj.weight']
        dst_block_state_dict['attn.out_proj.bias'] = src_block_state_dict[
            'attn.out_proj.bias']

        dst_block_state_dict['mlp.fc1.weight'] = src_block_state_dict[
            'mlp.c_fc.weight']
        dst_block_state_dict['mlp.fc1.bias'] = src_block_state_dict[
            'mlp.c_fc.bias']
        dst_block_state_dict['mlp.fc2.weight'] = src_block_state_dict[
            'mlp.c_proj.weight']
        dst_block_state_dict['mlp.fc2.bias'] = src_block_state_dict[
            'mlp.c_proj.bias']

        dst_block_state_dict['norm1.weight'] = src_block_state_dict[
            'ln_1.weight']
        dst_block_state_dict['norm1.bias'] = src_block_state_dict['ln_1.bias']
        dst_block_state_dict['norm2.weight'] = src_block_state_dict[
            'ln_2.weight']
        dst_block_state_dict['norm2.bias'] = src_block_state_dict['ln_2.bias']

        dst_state_dict.update(
            dict((dst_prefix + k, v) for k, v in dst_block_state_dict.items()))
        block_idx += 1

    return dst_state_dict


def load_weights_openclip(load_path: str) -> Dict[str, torch.Tensor]:
    src_state_dict = torch.load(load_path, map_location='cpu')
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}

    dst_state_dict['cls_token'] = src_state_dict[
        'vision_model.embeddings.class_embedding']
    dst_state_dict['pos_embed'] = src_state_dict[
        'vision_model.embeddings.position_embedding.weight']
    dst_state_dict['conv1.weight'] = src_state_dict[
        'vision_model.embeddings.patch_embedding.weight']
    dst_state_dict['ln_pre.weight'] = src_state_dict[
        'vision_model.pre_layrnorm.weight']
    dst_state_dict['ln_pre.bias'] = src_state_dict[
        'vision_model.pre_layrnorm.bias']

    block_idx = 0
    prefix = 'vision_model.encoder.layers.'
    while prefix + f'{block_idx}.self_attn.k_proj.weight' in src_state_dict:

        dst_state_dict[f'blocks.{block_idx}.attn.q_proj.weight'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.q_proj.weight']
        dst_state_dict[f'blocks.{block_idx}.attn.q_proj.bias'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.q_proj.bias']
        dst_state_dict[f'blocks.{block_idx}.attn.k_proj.weight'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.k_proj.weight']
        dst_state_dict[f'blocks.{block_idx}.attn.k_proj.bias'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.k_proj.bias']
        dst_state_dict[f'blocks.{block_idx}.attn.v_proj.weight'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.v_proj.weight']
        dst_state_dict[f'blocks.{block_idx}.attn.v_proj.bias'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.v_proj.bias']

        dst_state_dict[f'blocks.{block_idx}.attn.out_proj.weight'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.out_proj.weight']
        dst_state_dict[f'blocks.{block_idx}.attn.out_proj.bias'] = \
            src_state_dict[prefix + f'{block_idx}.self_attn.out_proj.bias']

        dst_state_dict[f'blocks.{block_idx}.mlp.fc1.weight'] = \
            src_state_dict[prefix+f'{block_idx}.mlp.fc1.weight']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc1.bias'] = \
            src_state_dict[prefix+f'{block_idx}.mlp.fc1.bias']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc2.weight'] = \
            src_state_dict[prefix+f'{block_idx}.mlp.fc2.weight']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc2.bias'] = \
            src_state_dict[prefix+f'{block_idx}.mlp.fc2.bias']

        dst_state_dict[f'blocks.{block_idx}.norm1.weight'] = \
            src_state_dict[prefix + f'{block_idx}.layer_norm1.weight']
        dst_state_dict[f'blocks.{block_idx}.norm1.bias'] = \
            src_state_dict[prefix + f'{block_idx}.layer_norm1.bias']
        dst_state_dict[f'blocks.{block_idx}.norm2.weight'] = \
            src_state_dict[prefix + f'{block_idx}.layer_norm2.weight']
        dst_state_dict[f'blocks.{block_idx}.norm2.bias'] = \
            src_state_dict[prefix + f'{block_idx}.layer_norm2.bias']

        block_idx += 1

    return dst_state_dict


def load_weights_mmcls(load_path: str) -> Dict[str, torch.Tensor]:
    src_state_dict = torch.load(load_path, map_location='cpu')
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}

    dst_state_dict['cls_token'] = src_state_dict['backbone.cls_token'].squeeze(
    )
    dst_state_dict['pos_embed'] = src_state_dict['backbone.pos_embed'].squeeze(
    )
    dst_state_dict['conv1.weight'] = src_state_dict[
        'backbone.patch_embed.projection.weight']

    block_idx = 0
    while f'backbone.layers.{block_idx}.attn.qkv.weight' in src_state_dict:

        qkv_weight = src_state_dict[
            f'backbone.layers.{block_idx}.attn.qkv.weight']
        qkv_bias = src_state_dict[f'backbone.layers.{block_idx}.attn.qkv.bias']
        qkv_dim = int(qkv_weight.shape[0] / 3)
        dst_state_dict[
            f'blocks.{block_idx}.attn.q_proj.weight'] = qkv_weight[:qkv_dim]
        dst_state_dict[
            f'blocks.{block_idx}.attn.q_proj.bias'] = qkv_bias[:qkv_dim]
        dst_state_dict[f'blocks.{block_idx}.attn.k_proj.weight'] = qkv_weight[
            qkv_dim:2 * qkv_dim]
        dst_state_dict[f'blocks.{block_idx}.attn.k_proj.bias'] = qkv_bias[
            qkv_dim:2 * qkv_dim]
        dst_state_dict[f'blocks.{block_idx}.attn.v_proj.weight'] = qkv_weight[
            2 * qkv_dim:]
        dst_state_dict[f'blocks.{block_idx}.attn.v_proj.bias'] = qkv_bias[
            2 * qkv_dim:]
        dst_state_dict[f'blocks.{block_idx}.attn.out_proj.weight'] = \
            src_state_dict[f'backbone.layers.{block_idx}.attn.proj.weight']
        dst_state_dict[f'blocks.{block_idx}.attn.out_proj.bias'] = \
            src_state_dict[f'backbone.layers.{block_idx}.attn.proj.bias']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc1.weight'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ffn.layers.0.0.weight']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc1.bias'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ffn.layers.0.0.bias']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc2.weight'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ffn.layers.1.weight']
        dst_state_dict[f'blocks.{block_idx}.mlp.fc2.bias'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ffn.layers.1.bias']
        dst_state_dict[f'blocks.{block_idx}.norm1.weight'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ln1.weight']
        dst_state_dict[f'blocks.{block_idx}.norm1.bias'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ln1.bias']
        dst_state_dict[f'blocks.{block_idx}.norm2.weight'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ln2.weight']
        dst_state_dict[f'blocks.{block_idx}.norm2.bias'] = \
            src_state_dict[f'backbone.layers.{block_idx}.ln2.bias']

        block_idx += 1

    return dst_state_dict


weight_loader_fn_dict = {
    'clip': load_weights_clip,
    'openclip': load_weights_openclip,
    'im21k': load_weights_mmcls,
}


class QuickGELU(nn.Module):
    """from official CLIP repo"""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self,
        q_in_dim: int,
        k_in_dim: int,
        v_in_dim: int,
        qk_proj_dim: int,
        v_proj_dim: int,
        num_heads: int,
        out_dim: int,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0)
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1)
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk**0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        return out


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


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        temporal_before_attn=True,
        temporal_before_ffn=True,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim,
            k_in_dim=in_feature_dim,
            v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim,
            v_proj_dim=qkv_dim,
            num_heads=num_heads,
            out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
                ('act', act()),
                ('dropout', nn.Dropout(mlp_dropout)),
                ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
            ]))

        self.norm1 = nn.LayerNorm(in_feature_dim)
        self.norm2 = nn.LayerNorm(in_feature_dim)

        self.temporal_before_attn = temporal_before_attn
        if self.temporal_before_attn:
            self.temporal_module1 = MyTempModule(in_feature_dim)
        self.temporal_before_ffn = temporal_before_ffn
        if self.temporal_before_ffn:
            self.temporal_module2 = MyTempModule(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        N, T, L, C = x.size()
        H = W = int((L - 1)**0.5)

        if self.temporal_before_attn:
            temp_x = x[:, :, 1:, :].contiguous().permute(0, 3, 1,
                                                         2).contiguous()
            temp_x = temp_x.view(N, C, T, H, W)
            temp_x = temp_x + self.temporal_module1(temp_x)
            temp_x = temp_x.view(N, C, T, H * W).permute(0, 2, 3,
                                                         1).contiguous()
            x = torch.cat([x[:, :, 0:1, :], temp_x], dim=2)  # (N, T, L, C)

        x = x.view(N * T, L, C)

        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)

        if self.temporal_before_ffn:
            x = x.view(N, T, L, C)
            temp_x = x[:, :, 1:, :].contiguous().permute(0, 3, 1,
                                                         2).contiguous()
            temp_x = temp_x.view(N, C, T, H, W)
            temp_x = temp_x + self.temporal_module2(temp_x)
            temp_x = temp_x.view(N, C, T, H * W).permute(0, 2, 3,
                                                         1).contiguous()
            x = torch.cat([x[:, :, 0:1, :], temp_x], dim=2)  # (N, T, L, C)
            x = x.view(N * T, L, C)

        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):

    def __init__(
        self,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        ln_pre: bool = False,
        layers_with_temporal_before_attn: list = [8, 9, 10, 11],
        layers_with_temporal_before_ffn: list = [8, 9, 10, 11],
    ):
        super().__init__()

        self.patch_size = patch_size
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=feature_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        # self.patch_proj = PatchProj(embed_dim=feature_dim)
        self.num_patches = np.prod(
            [x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(
            torch.zeros([self.num_patches, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim,
                qkv_dim=feature_dim,
                num_heads=num_heads,
                mlp_factor=mlp_factor,
                act=act,
                temporal_before_attn=(
                    True if i in layers_with_temporal_before_attn else False),
                temporal_before_ffn=(
                    True if i in layers_with_temporal_before_ffn else False),
            ) for i in range(num_layers)
        ])

        if ln_pre:
            self.ln_pre = nn.LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):

        N, _, T, _, _ = x.shape

        def _convert_to_2d(x: torch.Tensor) -> torch.Tensor:
            """(N, C, T, H, W) -> (N x T, C, H, W)"""
            x = x.permute((0, 2, 1, 3, 4))
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            return x

        x = _convert_to_2d(x)
        x = self.conv1(x)
        x = x.flatten(2).permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)

        x = x.view(N, T, -1, self.feature_dim)  # (N, T, L, C)

        all_features = []
        for blk in self.blocks:
            x = blk(x).view(N, T, -1, self.feature_dim)
            x_duplicate = x.clone()
            all_features.append(x_duplicate)
        return all_features


vit_presets = {
    'ViT-B/32':
    dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(32, 32),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
    ),
    'ViT-B/16':
    dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
    ),
    'ViT-L/14':
    dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
    ),
}


@BACKBONES.register_module()
class EVL(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = '',
        backbone_mode: str = 'finetune_temporal',
        decoder_num_layers: int = 4,
        layers_with_temporal_before_attn: list = [8, 9, 10, 11],
        layers_with_temporal_before_ffn: list = [],
    ):
        super().__init__()
        self.num_frames = num_frames
        if layers_with_temporal_before_attn is None:
            layers_with_temporal_before_attn = []
        if layers_with_temporal_before_ffn is None:
            layers_with_temporal_before_ffn = []
        self._create_backbone(backbone_name, backbone_type, backbone_path,
                              backbone_mode, layers_with_temporal_before_attn,
                              layers_with_temporal_before_ffn)
        self.decoder_num_layers = decoder_num_layers

    def init_weights(self):
        pass

    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
        layers_with_temporal_before_attn: list,
        layers_with_temporal_before_ffn: list,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer(
            **vit_presets[backbone_name],
            ln_pre=('clip' in backbone_type),
            layers_with_temporal_before_attn=layers_with_temporal_before_attn,
            layers_with_temporal_before_ffn=layers_with_temporal_before_ffn)

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

        return vit_presets[backbone_name]

    def forward(self, x: torch.Tensor):
        x = x.squeeze()
        NT, C, H, W = x.shape
        T = self.num_frames
        N = NT // T
        # (NT, C, H, W) -> (N, C, T, H, W)
        x = x.reshape(N, T, C, H, W).permute(0, 2, 1, 3, 4)

        features = self.backbone(x)[-self.decoder_num_layers:]
        # a list of (N, T, L, C)
        # (N, T, 197, 768) for ViT-B/16
        return features
