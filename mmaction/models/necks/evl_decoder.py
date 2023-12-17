from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

from ..builder import NECKS


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
        attn_map = torch.mean(aff, dim=-1)  # mean across head dim
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        attn_out = self.out_proj(mix.flatten(-2))

        return attn_out, attn_map


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        num_self_attn_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        enable_batch_self_attn: bool = False,
    ):
        super().__init__()

        self.enable_batch_self_attn = enable_batch_self_attn
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
        self.norm3 = nn.LayerNorm(in_feature_dim)
        if self.enable_batch_self_attn:
            self.batch_attn = nn.MultiheadAttention(
                in_feature_dim,
                num_heads=num_self_attn_heads,
                dropout=mlp_dropout,
                batch_first=False)
            self.norm4 = nn.LayerNorm(in_feature_dim)
            self.norm5 = nn.LayerNorm(in_feature_dim)
            self.mlp2 = nn.Sequential(
                OrderedDict([
                    ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
                    ('act', act()),
                    ('dropout', nn.Dropout(mlp_dropout)),
                    ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
                ]))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_norm = self.norm3(y)

        attn_out, attn_map = self.attn(self.norm1(x), y_norm, y_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))

        if self.enable_batch_self_attn:
            x = x + self.batch_attn(self.norm4(x), self.norm4(x), x)[0]
            x = x + self.mlp2(self.norm5(x))

        return x, attn_map


@NECKS.register_module()
class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        num_layers: int = 4,
        layers_with_batch_self_attn: list = [0, 1, 2, 3],
        origin_in_feature_dim: list = None,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        num_self_attn_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_in_frame_decoder: bool = False,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_in_frame_decoder = enable_in_frame_decoder
        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.layers_with_batch_self_attn = layers_with_batch_self_attn if (
            layers_with_batch_self_attn is not None) else []
        self.num_layers = num_layers
        self.in_feature_dim = in_feature_dim

        # shift feature dimension to in_feature_dim
        if origin_in_feature_dim is not None:
            assert (isinstance(origin_in_feature_dim, list)
                    and len(origin_in_feature_dim) == num_layers)
            self.dim_shift_layers = nn.ModuleList([
                nn.Linear(origin_in_feature_dim[i], in_feature_dim)
                for i in range(num_layers)
            ])
        else:
            self.dim_shift_layers = nn.ModuleList(
                [nn.Identity() for _ in range(num_layers)])

        if self.enable_in_frame_decoder:
            self.in_frame_decoder = nn.ModuleList([
                TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads,
                                        num_self_attn_heads, mlp_factor,
                                        mlp_dropout) for _ in range(num_layers)
            ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                in_feature_dim,
                qkv_dim,
                num_heads,
                num_self_attn_heads,
                mlp_factor,
                mlp_dropout,
                enable_batch_self_attn=(i in self.layers_with_batch_self_attn))
            for i in range(num_layers)
        ])

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList([
                nn.Conv1d(
                    in_feature_dim,
                    in_feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=in_feature_dim) for _ in range(num_layers)
            ])
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList([
                nn.Parameter(torch.zeros([num_frames, in_feature_dim]))
                for _ in range(num_layers)
            ])

        self.cls_token = nn.Parameter(
            torch.zeros([1, 1, in_feature_dim]))

        self.layer_norm = nn.LayerNorm(in_feature_dim)

    def init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        # nn.init.normal_(self.dim_shift_layers, std=0.02)

    def forward(self, in_features: List[torch.Tensor]):

        N = in_features[0].shape[0]
        assert len(in_features) == self.num_layers
        x = self.cls_token.repeat(N, 1, 1)

        # attn_scores = torch.zeros([])
        attn_maps = []
        for i in range(self.num_layers):
            in_features[i] = self.dim_shift_layers[i](in_features[i])
            frame_features = in_features[i]
            N, T, L, C = frame_features.size()

            if self.enable_in_frame_decoder:
                frame_features = torch.cat([
                    self.in_frame_decoder[i](
                        frame_features[:, t, :, :],
                        frame_features[:, t, :, :])[0].unsqueeze(1)
                    for t in range(T)
                ],
                                           dim=1)

            if self.enable_temporal_conv:
                feat = in_features[i]
                feat = feat.permute(0, 2, 3,
                                    1).contiguous().flatten(0,
                                                            1)  # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C,
                                 T).permute(0, 3, 1,
                                            2).contiguous()  # N, T, L, C
                frame_features += feat

            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)

            frame_features = frame_features.flatten(1, 2)  # N, T * L, C

            x, attn_map = self.decoder_layers[i](x, frame_features)
            attn_maps.append(attn_map)

            # attn_score = attn_score.view(N, 1, T, L).sum(
            #     axis=-1, keepdim=False)

            # if i == 0:
            #     attn_scores = attn_score
            # else:
            #     attn_scores += attn_score

        # save attn_maps for visualization
        # torch.save(attn_maps, 'attn_maps.pth')

        x = self.layer_norm(x)
        # ret_dict = {'feats': x, 'aux_feats': attn_scores}
        ret_dict = {'feats': x}
        return ret_dict
