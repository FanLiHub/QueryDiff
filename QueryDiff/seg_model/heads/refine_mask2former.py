from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


@MODELS.register_module()
class RefineMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat: bool = False, **kwargs):
        super().__init__(**kwargs)

        feat_channels = kwargs["feat_channels"]

        del self.query_embed

        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

        self.query_proj = nn.Linear(feat_channels, feat_channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_channels, num_heads=8, batch_first=True
        )

        dilation_list = [1, 2, 3, 4]
        self.fuse_convs = nn.ModuleList([
            DepthwiseSeparableConv(
                in_channels=feat_channels * 4,
                out_channels=feat_channels,
                kernel_size=3,
                dilation=d
            )
            for d in dilation_list
        ])

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:

        x, query_embed = x

        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)

        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)

        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        orig_sizes = [mask_features.shape[-2:]] + [
            feat.shape[-2:] for feat in multi_scale_memorys
        ]

        q_embed = self.query_proj(query_embed)

        all_feats = [mask_features] + list(multi_scale_memorys)

        refined = []
        for feat in all_feats:
            B, C, H, W = feat.shape
            seq = feat.flatten(2).permute(0, 2, 1)
            ca, _ = self.cross_attn(
                query=seq, key=q_embed, value=q_embed
            )
            ca = ca.permute(0, 2, 1).reshape(B, C, H, W)
            refined.append(ca)

        target_h, target_w = refined[0].shape[-2:]
        refined_up = []
        for f in refined:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(
                    f, size=(target_h, target_w),
                    mode="bilinear", align_corners=False
                )
            refined_up.append(f)

        fused_in = torch.cat(refined_up, dim=1)

        new_feats = [conv(fused_in) for conv in self.fuse_convs]

        mask_features = F.interpolate(
            new_feats[0],
            size=orig_sizes[0],
            mode="bilinear",
            align_corners=False,
        )

        restored_multi_scale = []
        for i in range(self.num_transformer_feat_level):
            feat_i = F.interpolate(
                new_feats[i + 1],
                size=orig_sizes[i + 1],
                mode="bilinear",
                align_corners=False,
            )
            restored_multi_scale.append(feat_i)

        multi_scale_memorys = restored_multi_scale

        decoder_inputs = []
        decoder_positional_encodings = []

        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)

            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []

        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                key_padding_mask=None,
            )

            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list
