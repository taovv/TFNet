import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from mmcv.cnn import (Conv2d, Linear, build_activation_layer, build_norm_layer)
from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models.losses import accuracy
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.deeplabv3.decoder import SeparableConv2d
from .base import SignalBaseSegmentor
from mmseg.models.builder import build_backbone, SEGMENTORS, build_head
from torch.nn import MultiheadAttention


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(self.keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Implements encoder block with residual connection.
    Args:
        dim (int): The feature dimension.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Drop rate for mlp output weights. Default: 0.
        attn_drop (float): Drop rate for attention output weights.
            Default: 0.
        proj_drop (float): Drop rate for attn layer output weights.
            Default: 0.
        drop_path (float): Drop rate for paths of model.
            Default: 0.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN', requires_grad=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 with_cp=False):
        super(Block, self).__init__()
        self.with_cp = with_cp
        _, self.norm1 = build_norm_layer(norm_cfg, dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop,
                              proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

    def forward(self, x):
        out = x + self.drop_path(self.attn(self.norm1(x)))
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out


class SEBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 mlp_ratio=4,
                 reduction=16,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6)):
        super(SEBlock, self).__init__()
        _, self.norm1 = build_norm_layer(norm_cfg, dim)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(dim, dim // reduction, groups=num_heads, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(dim // reduction, dim, groups=num_heads, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        _, self.norm2 = build_norm_layer(norm_cfg, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

    def forward(self, x):
        out = self.norm1(x)
        out = out.transpose(1, 2)
        out = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(out)))))
        out = out.transpose(1, 2)
        out = x + self.drop_path(out * x)
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out


class PatchEmbed(nn.Module):

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=512):
        super(PatchEmbed, self).__init__()
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        elif isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            raise TypeError('img_size must be type of int or tuple')
        h, w = self.img_size
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PosCNN(nn.Module):
    def __init__(self, in_channels, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class TransBlock(nn.Module):

    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_channels=512,
                 out_channels=256,
                 num_head_msa=4,
                 num_head_mca=4,
                 mlp_ratio=4,
                 reduction=16,
                 position_embed='condition',  # 'abs' or 'condition'
                 drop_rate=0.,
                 interpolate_mode='bicubic',
                 **block_args
                 ):
        super(TransBlock, self).__init__()

        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.interpolate_mode = interpolate_mode
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                      embed_dim=out_channels)
        self.position_embed = position_embed
        if position_embed == 'condition':
            self.pos_embed = PosCNN(out_channels, out_channels)
        else:
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_channels))
        self.transformer_block = Block(dim=out_channels, num_heads=num_head_msa, mlp_ratio=mlp_ratio, **block_args)
        self.se_block = SEBlock(dim=out_channels, num_heads=num_head_mca, mlp_ratio=mlp_ratio, reduction=reduction)

    def _pos_embeding(self, img, patched_img, pos_embed):
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError('Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape))

            pos_embed = self.resize_pos_embed(pos_embed, img.shape[2:],
                                              (pos_h, pos_w), self.patch_size,
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, patch_size, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        input_h, input_w = input_shpae
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight,
            size=[input_h // patch_size, input_w // patch_size],
            align_corners=False,
            mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.patch_embed(x)
        if self.position_embed == 'condition':
            pos_embed = self.pos_embed(out, h // self.patch_size, w // self.patch_size)
            out = out + pos_embed
        else:
            out = self._pos_embeding(x, out, self.pos_embed)
        out = self.transformer_block(out)
        out = self.se_block(out)
        out = out.transpose(1, 2).view(b, -1, h // self.patch_size, w // self.patch_size)
        return out


@SEGMENTORS.register_module()
class TFNet(SignalBaseSegmentor):

    def __init__(
            self,
            backbone=None,
            pretrained=None,
            feat_sizes: tuple = (32, 16, 8),
            patch_sizes: tuple = (4, 2, 1),
            tf_out_channels: tuple = (256, 256, 256),
            decoder_channels: tuple = (256, 256),
            low_feat_index: int = -4,
            low_feat_out_channels: int = 48,
            first_up: int = 8,
            second_up: int = 4,
            msa_heads: tuple = (4, 4, 4),
            mca_heads: tuple = (4, 4, 4),
            position_embed: str = 'condition',
            classes: int = 2,
            activation: Optional[str] = None,
            auxiliary_head=None,
            loss=dict(
                type='MyLoss',
                kl_weight=0.1,
                bce_weight=1.0),
            train_cfg=dict(),
            test_cfg=dict()
    ):
        super(TFNet, self).__init__(loss, train_cfg, test_cfg)
        self.backbone = build_backbone(backbone)
        self._init_auxiliary_head(auxiliary_head)

        self.low_feat_index = low_feat_index

        features_channels = self.backbone.out_channels

        self.tb1 = TransBlock(img_size=feat_sizes[0], patch_size=patch_sizes[0], in_channels=features_channels[-3],
                              out_channels=tf_out_channels[0], num_head_msa=msa_heads[0], num_head_mca=mca_heads[0],
                              position_embed=position_embed)
        self.tb2 = TransBlock(img_size=feat_sizes[1], patch_size=patch_sizes[1], in_channels=features_channels[-2],
                              out_channels=tf_out_channels[1], num_head_msa=msa_heads[1], num_head_mca=mca_heads[1],
                              position_embed=position_embed)
        self.tb3 = TransBlock(img_size=feat_sizes[2], patch_size=patch_sizes[2], in_channels=features_channels[-1],
                              out_channels=tf_out_channels[2], num_head_msa=msa_heads[2], num_head_mca=mca_heads[2],
                              position_embed=position_embed)

        self.project = nn.Sequential(
            nn.Conv2d(sum(tf_out_channels), decoder_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.dsv = nn.Sequential(
            SeparableConv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(features_channels[low_feat_index], low_feat_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_feat_out_channels),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            SeparableConv2d(
                low_feat_out_channels + decoder_channels[0],
                decoder_channels[1],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU()
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=first_up)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=16)

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[1],
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=second_up,
        )

        self.init_weights()

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def init_weights(self):
        init.initialize_head(self.segmentation_head)
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def forward_train(self, imgs, img_metas, gt_semantic_seg):
        losses = dict()
        seg_logits, aux_seg_logits, x = self._forward(imgs)
        seg_losses = self.losses_(seg_logits, aux_seg_logits, gt_semantic_seg)
        losses.update(add_prefix(seg_losses, 'seg'))
        return losses

    def losses_(self, seg_logits, aux_seg_logits, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze(1)
        loss['loss'] = 0.
        for l in self.loss:
            if self.with_auxiliary_head:
                loss['loss'] += l(seg_logits, aux_seg_logits, seg_label)
            else:
                loss['loss'] += l(seg_logits, seg_label, weight=None, ignore_index=self.ignore_index)
        loss['acc'] = accuracy(seg_logits, seg_label)
        return loss

    def encode_decode(self, img, img_metas):
        out, _, _ = self._forward(img)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _forward(self, imgs):
        features = self.backbone(imgs)

        if self.with_auxiliary_head:
            masks_aux_ = self.auxiliary_head.forward(features)
            masks_aux = self.up2(masks_aux_)
        else:
            masks_aux = None
        x1 = self.tb1(features[-3])
        x2 = self.tb2(features[-2])
        x3 = self.tb3(features[-1])
        x = torch.cat([x3, x2, x1], dim=1)
        x = self.dsv(self.project(x))
        x = self.up(x)
        high_res_features = self.block1(features[self.low_feat_index])
        concat_features = torch.cat([x, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        masks = self.segmentation_head(fused_features)
        return masks, masks_aux, features
