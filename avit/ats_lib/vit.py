""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats_lib.utils import batch_index_select

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # patch models (my experiments)
    "vit_small_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth",
    ),
    # patch models (weights ported from official Google JAX impl)
    "vit_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_base_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch32_224": _cfg(
        url="",  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    "vit_large_patch32_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
    ),
    # patch models, imagenet21k (weights ported from official Google JAX impl)
    "vit_base_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_base_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch16_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_large_patch32_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "vit_huge_patch14_224_in21k": _cfg(
        hf_hub="timm/vit_huge_patch14_224_in21k",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    # hybrid models (weights ported from official Google JAX impl)
    "vit_base_resnet50_224_in21k": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=0.9,
        first_conv="patch_embed.backbone.stem.conv",
    ),
    "vit_base_resnet50_384": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth",
        input_size=(3, 384, 384),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=1.0,
        first_conv="patch_embed.backbone.stem.conv",
    ),
    # hybrid models (my experiments)
    "vit_small_resnet26d_224": _cfg(),
    "vit_small_resnet50d_s3_224": _cfg(),
    "vit_base_resnet26d_224": _cfg(),
    "vit_base_resnet50d_224": _cfg(),
    # deit models (FB weights)
    "vit_deit_tiny_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    ),
    "vit_deit_small_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
    ),
    "vit_deit_base_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
    ),
    "vit_deit_base_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
    "vit_deit_tiny_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
    ),
    "vit_deit_small_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth"
    ),
    "vit_deit_base_distilled_patch16_224": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
    ),
    "vit_deit_base_distilled_patch16_384": _cfg(
        url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
        input_size=(3, 384, 384),
        crop_pct=1.0,
    ),
}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        if policy is not None:
            qkv = qkv * policy.unsqueeze(0).unsqueeze(2)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if policy is not None:
            x = self.proj(x) * policy
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, policy=None):
        x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
        if policy is not None:
            x = x * policy
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if policy is not None:
            x = x * policy
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.in_channels = in_chans
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=(stride, stride)
        )

    def num_patches(self):
        num_patches = self.forward(
            torch.zeros((1, self.in_channels, self.img_size[0], self.img_size[1]))
        ).shape[1]
        return num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PredictorLG(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, : C // 2]
        global_x = (x[:, :, C // 2 :] * policy).sum(dim=1, keepdim=True) / torch.sum(
            policy, dim=1, keepdim=True
        )
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)
        return self.out_conv(x)


class VisionTransformerDiffPruning(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        pruning_loc=None,
        token_ratio=None,
        distill=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print("## diff vit pruning method")
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        predictor_list = [PredictorLG(embed_dim) for _ in range(len(pruning_loc))]

        self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        p_count = 0
        out_pred_prob = []
        init_n = 14 * 14
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n + 1, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                spatial_x = x[:, 1:]
                pred_score = self.score_predictor[p_count](
                    spatial_x, prev_decision
                ).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = (
                        F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1]
                        * prev_decision
                    )
                    out_pred_prob.append(hard_keep_decision.reshape(B, init_n))
                    cls_policy = torch.ones(
                        B,
                        1,
                        1,
                        dtype=hard_keep_decision.dtype,
                        device=hard_keep_decision.device,
                    )
                    policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:, :, 0]
                    num_keep_node = int(init_n * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[
                        :, :num_keep_node
                    ]
                    cls_policy = torch.zeros(
                        B, 1, dtype=keep_policy.dtype, device=keep_policy.device
                    )
                    now_policy = torch.cat([cls_policy, keep_policy + 1], dim=1)
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = blk(x)
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = self.norm(x)
        features = x[:, 1:]
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob
            else:
                return x, out_pred_prob
        else:
            return x

import time
import pdb


from collections import OrderedDict

from torch import Tensor

from ats_lib.vit import Attention, Block, Mlp
from ats_lib.lvvit import GroupLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
import numpy as np


class QuerySelector(Attention):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        drop_path=0.0,
    ):
        super(QuerySelector, self).__init__(
            dim_in, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        # self.norm1 = norm_layer(dim_in)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        #self.score_biases = nn.Parameter(torch.zeros(1, 197))  # [1 x T]
        self.bin_sizes = nn.Parameter(torch.rand(1, 103))  # [1 x n-1]

        ys = np.linspace(0, 1.0, 196)
        self.ys = nn.Parameter(torch.tensor(ys).float().unsqueeze(dim=0))


    def mask_generation(
        self, bin_sizes: Tensor, bin_starts: Tensor, num_tokens: int
    ) -> Tensor:
        """

        :param bin_sizes: [1 x n-1]
        :param bin_starts:
        :param num_tokens:
        :return:
        """
        _, k = bin_sizes.shape
        l_s = 10.0
        t = torch.arange(num_tokens).to(bin_sizes.device)
        l_c = bin_starts + bin_sizes / 2.0
        l_c = l_c * num_tokens
        l_w = bin_sizes * num_tokens
        l_c = l_c.unsqueeze(-1).expand(1, k, num_tokens)  # [1 x n-1 x num_tokens]
        l_w = l_w.unsqueeze(-1).expand(1, k, num_tokens)  # [1 x n-1 x num_tokens]
        masks = 1 / (
            (torch.exp(l_s * (t - l_c - l_w)) + 1)
            * (torch.exp(l_s * (-t + l_c - l_w)) + 1)
        )  # [1 x n-1 x num_tokens]
        return masks

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    @staticmethod
    def find_unique_indices(tokens_to_pick_ind):
        unique_indices = np.unique(tokens_to_pick_ind)
        num_unique_indices = len(unique_indices)
        indices = np.zeros(tokens_to_pick_ind.shape[0])
        indices[:num_unique_indices] = unique_indices
        #indices[num_unique_indices : tokens_to_pick_ind.shape[0]] = (
        #    unique_indices[-1] + 1
        #)
        indices[num_unique_indices: tokens_to_pick_ind.shape[0]] = tokens_to_pick_ind.shape[0]
        return indices

    @staticmethod
    def create_ys(norm_cdf, n):
        ys_start_ind = (norm_cdf != 0).astype(int).argmax()
        ys_start = norm_cdf[ys_start_ind]
        ys = np.linspace(ys_start, 1.0, n - 1)
        # ys = np.random.uniform(ys_start, 1.0, n - 1)
        return ys


    def forward(self, x, n, raw_x, policy):

        # end = time.time()

        B, N, C = x.shape
        if type(N) is Tensor:
            N = N.cpu().item()
        #n = 0.9
        if n <= 1.0:
            n = n * N
            if n < 8:
                n = 8
        n = round(n)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # qkv = qkv * policy.unsqueeze(0).unsqueeze(2)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_no_softmax.softmax(dim=-1)
        # attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]

        # batch_time = time.time() - end
        # print(batch_time)

        if True:
            v_norm_2 = torch.linalg.norm(
                v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
            )  # [B x T]
            selection_score = attn[:, :, 0].sum(dim=1)  # [B x T]
            selection_score = selection_score * v_norm_2  # [B x T]
            # selection_score = attn[:, :, :, :].sum(dim=(1, 2))  # [B x T]
            selection_score = selection_score[:, 1:]  # [B x T-1]
            # selection_score = (
            #    selection_score.detach() + self.score_biases[:, 1:]
            # )  # [B x T-1]
            # selection_score = F.softmax(selection_score, dim=1)  # [B x T-1]
            selection_score = selection_score / selection_score.sum(
                dim=1, keepdim=True
            )  # [B x T-1]
            sorted_scores, sorted_ind = torch.sort(
                selection_score, descending=False, dim=1
            )
            # selection_score[:, sorted_ind] = (
            #        selection_score[:, sorted_ind].detach() + self.score_biases[:, 1:]
            # )  # [B x T-1]
            # selection_score = F.softmax(selection_score/1.0, dim=1)  # [B x T-1]
            # sorted_scores = selection_score
            if True:
                cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]
                norm_cdf = (
                    (cdf - cdf.min(dim=1)[0].unsqueeze(dim=1))
                    / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)
                ).unsqueeze(
                    dim=1
                )  # [B x 1 x T-1]

            # norm_score = (selection_score - selection_score.min()) / (selection_score.max() - selection_score.min())
            #ys_start_ind = (norm_cdf != 0).int().argmax(dim=2)
            #ys_start = norm_cdf[ys_start_ind]
            """
            ys = (
                torch.linspace(1/(n-1), 1.0, n - 1)
                .unsqueeze(dim=1)
                .to(selection_score.device)
            )
            """
            # pdb.set_trace()
            # FIX THIS
            # ys = np.apply_along_axis(self.create_ys, 2, norm_cdf.detach().cpu().numpy(), n)
            # ys = torch.tensor(ys).to(selection_score.device)

            ys = self.ys.detach()
            ys = ys[:,:norm_cdf.shape[2]]
            ys = ys.unsqueeze(0).expand(B, -1, -1)
            # FIX THIS
            n = norm_cdf.shape[2]+1
            """
            expanded_ys = torch.Tensor.expand(
                torch.Tensor.expand(ys, (-1, N - 1)).unsqueeze(2), (n - 1, N - 1, B)
            ).permute(
                2, 0, 1
            )  # [B x n-1 x T-1]
            """
            ys = ys.permute(0, 2, 1).expand(B, norm_cdf.shape[2], norm_cdf.shape[2])
            tokens_to_pick_ind = torch.min(torch.abs(ys - norm_cdf), dim=2)[
                1
            ]  # [B x n-1]



            raw_x_tmp = torch.gather(
                raw_x[:, 1:], 1, sorted_ind.unsqueeze(-1).expand(B, N - 1, C)
            )
            # raw_x_tmp = torch.cat(
            #     [raw_x_tmp, torch.zeros_like(raw_x_tmp[:, 0:1, :])], dim=1
            # )  # [B x n x C]



            # FIX THIS
            # indices = (
            #     torch.tensor(
            #         np.apply_along_axis(
            #             self.find_unique_indices, 1, tokens_to_pick_ind.cpu().numpy()
            #         )
            #     )
            #     .long()
            #     .to(x.device)
            # )
            indices = tokens_to_pick_ind
            # FIX THIS

            # pdb.set_trace()
            raw_x_tmp = torch.gather(
                raw_x_tmp, 1, indices.unsqueeze(2).expand(B, n - 1, C)
            )


            raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)
            selected_x = raw_x_tmp


        # end = time.time()
        if False:  # top K
            v_norm_2 = torch.linalg.norm(
                v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
            )  # [B x T]
            selection_score = attn[:, :, 0].sum(dim=1)  # [B x T]
            selection_score = selection_score * v_norm_2  # [B x T]
            # selection_score = attn[:, :, :, :].sum(dim=(1, 2))  # [B x T]
            selection_score = selection_score[:, 0:]  # [B x T-1]
            # selection_score = (
            #    selection_score.detach() + self.score_biases[:, 1:]
            # )  # [B x T-1]
            # selection_score = F.softmax(selection_score, dim=1)  # [B x T-1]
            selection_score = selection_score / selection_score.sum(
                dim=1, keepdim=True
            )  # [B x T-1]

            top_scores, top_tokens = torch.topk(selection_score[:, :], k=n)  # [B x n]
            policy = torch.ones_like(top_tokens).unsqueeze(-1)
            top_tokens_expanded = (
                top_tokens.unsqueeze(dim=1)
                    .unsqueeze(dim=-1)
                    .expand(B, self.num_heads, n, N)
            )
            tokens_selected = torch.gather(attn, 2, top_tokens_expanded)
            selected_x = torch.gather(raw_x, 1, top_tokens.unsqueeze(-1).expand(B, n, C))
            attn = tokens_selected

        # -----------------------------

        x = self.proj_drop(x)
        #top_tokens = tokens_to_pick_ind

        # batch_time = time.time() - end
        # print(batch_time)
        #

        return x, selected_x, None, policy

class TokenSampling(Attention):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        representation_size=None,
        num_classes=1000,
        drop_path=0,
    ):
        super(TokenSampling, self).__init__(
            dim_in, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )

        self.proj_q_selector = QuerySelector(
            dim_in=dim_in,
            dim_out=dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
        )

        self.norm = norm_layer(dim_in)

    def forward(self, x, n, raw_x, policy):
        B, N, C = x.shape
        x_out = x

        x, q_tokens, masks, policy = self.proj_q_selector(x, n, raw_x, policy)


        selected_tokens = q_tokens

        return x, selected_tokens, masks, policy



class ATSBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = TokenSampling(
            dim_in=dim,
            dim_out=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, n=197, policy=None):
        x_out, top_tokens, masks, policy = self.attn(self.norm1(x), n, x, policy)
        # x_out = x_out/self.skip_lam
        if x_out.shape == x.shape and False:
            x = x + self.drop_path(x_out)
        else:
            B, N, C = x_out.shape

            selected_x = (
                top_tokens  # torch.gather(x, 1, top_tokens_expanded)  # [B x N x C]
            )

            x = selected_x + self.drop_path(x_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        return x, top_tokens, None, policy






class VisionTransformerTeacher(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        print("## diff vit pruning method")
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.ats_block = ATSBlock(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[0],
                            norm_layer=norm_layer,
                        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        # end = time.time()

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # end = time.time()
        for i, blk in enumerate(self.blocks):
            if i == 3:

                # end = time.time()
                # blk(x)
                # batch_time = time.time() - end
                # print(batch_time)
                # end = time.time()
                self.ats_block(x)
                # batch_time = time.time() - end
                # print(batch_time)
                # x = x[:,:108,:]
            # if i == 4:
            #     self.ats_block(x)
            #     x = x[:,:112,:]
            # if i == 5:
            #     self.ats_block(x)
            #     x = x[:,:100,:]
            # if i == 5:
            #     self.ats_block(x)
            #     x = x[:,:120,:]
            # if i == 6:
            #     self.ats_block(x)
            #     x = x[:,:110,:]
            # if i == 7:
            #     self.ats_block(x)
            #     x = x[:,:100,:]
            # if i == 8:
            #     self.ats_block(x)
            #     x = x[:,:87,:]
            # if i == 9:
            #     self.ats_block(x)
            #     x = x[:,:70,:]
            # if i == 10:
            #     self.ats_block(x)
            #     x = x[:,:55,:]
            else:
                x = blk(x)

        feature = self.norm(x)
        cls = feature[:, 0]
        tokens = feature  # feature[:, 1:]
        cls = self.pre_logits(cls)
        cls = self.head(cls)
        # batch_time = time.time() - end
        # print(batch_time)
        return cls, tokens


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, integrate_attn, stages):
    out_dict = {}
    if "model" in state_dict:
        # For DeiT models
        state_dict = state_dict["model"]

    if integrate_attn:
        for stage in stages:
            if stage != -1:
                out_dict[
                    "blocks.{}.attn.proj_q_selector.qkv.weight".format(stage)
                ] = state_dict["blocks.{}.attn.qkv.weight".format(stage)]
                out_dict[
                    "blocks.{}.attn.proj_q_selector.qkv.bias".format(stage)
                ] = state_dict["blocks.{}.attn.qkv.bias".format(stage)]
                out_dict[
                    "blocks.{}.attn.proj_q_selector.proj.weight".format(stage)
                ] = state_dict["blocks.{}.attn.proj.weight".format(stage)]
                out_dict[
                    "blocks.{}.attn.proj_q_selector.proj.bias".format(stage)
                ] = state_dict["blocks.{}.attn.proj.bias".format(stage)]

                """
                out_dict["blocks.{}.attn.proj_q_selector.proj_q.weight".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.weight".format(stage)
                ].reshape(3, 384, 384)[0]
                out_dict["blocks.{}.attn.proj_q_selector.proj_k.weight".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.weight".format(stage)
                ].reshape(3, 384, 384)[1]
                out_dict["blocks.{}.attn.proj_q_selector.proj_v.weight".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.weight".format(stage)
                ].reshape(3, 384, 384)[2]
                out_dict["blocks.{}.attn.proj_q_selector.proj_q.bias".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.bias".format(stage)
                ].reshape(3, 384)[0]
                out_dict["blocks.{}.attn.proj_q_selector.proj_k.bias".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.bias".format(stage)
                ].reshape(3, 384)[1]
                out_dict["blocks.{}.attn.proj_q_selector.proj_v.bias".format(stage)] = state_dict[
                    "blocks.{}.attn.qkv.bias".format(stage)
                ].reshape(3, 384)[2]
                out_dict["blocks.{}.attn.proj_q_selector.proj.weight".format(stage)] = state_dict[
                    "blocks.{}.attn.proj.weight".format(stage)
                ]
                out_dict["blocks.{}.attn.proj_q_selector.proj.bias".format(stage)] = state_dict[
                    "blocks.{}.attn.proj.bias".format(stage)
                ]
                out_dict["blocks.{}.attn.proj_q_selector.norm1.weight".format(stage)] = state_dict[
                    "blocks.{}.norm1.weight".format(stage)
                ]
                out_dict["blocks.{}.attn.proj_q_selector.norm1.bias".format(stage)] = state_dict[
                    "blocks.{}.norm1.bias".format(stage)
                ]
                if True:
                    out_dict["blocks.{}.attn.proj_q.weight".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.weight".format(stage)
                    ].reshape(3, 384, 384)[0]
                    out_dict["blocks.{}.attn.proj_k.weight".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.weight".format(stage)
                    ].reshape(3, 384, 384)[1]
                    out_dict["blocks.{}.attn.proj_v.weight".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.weight".format(stage)
                    ].reshape(3, 384, 384)[2]
                    out_dict["blocks.{}.attn.proj_q.bias".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.bias".format(stage)
                    ].reshape(3, 384)[0]
                    out_dict["blocks.{}.attn.proj_k.bias".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.bias".format(stage)
                    ].reshape(3, 384)[1]
                    out_dict["blocks.{}.attn.proj_v.bias".format(stage)] = state_dict[
                        "blocks.{}.attn.qkv.bias".format(stage)
                    ].reshape(3, 384)[2]
                    out_dict["blocks.{}.attn.proj.weight".format(stage)] = state_dict[
                        "blocks.{}.attn.proj.weight".format(stage)
                    ]
                    out_dict["blocks.{}.attn.proj.bias".format(stage)] = state_dict[
                        "blocks.{}.attn.proj.bias".format(stage)
                    ]
                    out_dict["blocks.{}.attn.proj_1.weight".format(stage)] = state_dict[
                        "blocks.{}.attn.proj.weight".format(stage)
                    ]
                    out_dict["blocks.{}.attn.proj_1.bias".format(stage)] = state_dict[
                        "blocks.{}.attn.proj.bias".format(stage)
                    ]
                    out_dict["blocks.{}.attn.norm1.weight".format(stage)] = state_dict[
                        "blocks.{}.norm1.weight".format(stage)
                    ]
                    out_dict["blocks.{}.attn.norm1.bias".format(stage)] = state_dict[
                        "blocks.{}.norm1.bias".format(stage)
                    ]
                if False:
                    for i in range(1, 3):
                        out_dict["blocks.{}.attn_{}.proj_q.proj_q.weight".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.weight".format(stage)
                        ].reshape(3, 384, 384)[0]
                        out_dict["blocks.{}.attn_{}.proj_q.proj_k.weight".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.weight".format(stage)
                        ].reshape(3, 384, 384)[1]
                        out_dict["blocks.{}.attn_{}.proj_q.proj_v.weight".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.weight".format(stage)
                        ].reshape(3, 384, 384)[2]
                        out_dict["blocks.{}.attn_{}.proj_q.proj_q.bias".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.bias".format(stage)
                        ].reshape(3, 384)[0]
                        out_dict["blocks.{}.attn_{}.proj_q.proj_k.bias".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.bias".format(stage)
                        ].reshape(3, 384)[1]
                        out_dict["blocks.{}.attn_{}.proj_q.proj_v.bias".format(stage, i)] = state_dict[
                            "blocks.{}.attn.qkv.bias".format(stage)
                        ].reshape(3, 384)[2]
                        out_dict["blocks.{}.attn_{}.proj_q.proj.weight".format(stage, i)] = state_dict[
                            "blocks.{}.attn.proj.weight".format(stage)
                        ]
                        out_dict["blocks.{}.attn_{}.proj_q.proj.bias".format(stage, i)] = state_dict[
                            "blocks.{}.attn.proj.bias".format(stage)
                        ]

                del state_dict["blocks.{}.attn.qkv.bias".format(stage)]
                del state_dict["blocks.{}.attn.qkv.weight".format(stage)]
                del state_dict["blocks.{}.attn.proj.weight".format(stage)]
                del state_dict["blocks.{}.attn.proj.bias".format(stage)]
                """

    for k, v in state_dict.items():
        if not integrate_attn and ("blocks" in k):
            block_num = int(k.split(".")[1])
            if block_num >= 7:
                k = "blocks.{}.".format(block_num + 1) + ".".join(k.split(".")[2:])

        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v

    return out_dict
