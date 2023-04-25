import pdb
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from ats_lib.vit import HybridEmbed, PatchEmbed, Block
from ats_lib.TokenSampler import TokenSampling, ATSBlock


class DynamicVisionTransformer(nn.Module):
    """
    Our VisionTransformer
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
        integrate_attn=False,
        stages=[6],
        num_tokens=[8],
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

        print("Dynamic ViT ...")
        self.num_classes = num_classes
        self.integrate_attn = integrate_attn
        self.num_tokens = num_tokens
        self.stages = stages
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
                stride=patch_size,
            )
        num_patches = self.patch_embed.num_patches()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        block_list = []
        for i in range(depth):
            if integrate_attn:
                if i not in stages:
                    block_list += [
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
                    ]
                elif i in stages:
                    block_list += [
                        ATSBlock(
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
                    ]
            else:
                block_list += [
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
                ]
                if i in stages:
                    block_list += [
                        TokenSampling(
                            dim_in=embed_dim,
                            dim_out=embed_dim,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop_rate,
                        )
                    ]

        self.blocks = nn.ModuleList(block_list)
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

        # self.adv_training = False
        # self.adv_patch = nn.Parameter(torch.rand(16, 16, 3)-0.5)

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

        B, C, W, H = x.shape

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        ats_cls = []
        top_tokens = None
        init_n = self.pos_embed.shape[1]
        policies = []
        attn = None
        cls_attn_list = []
        qk_list = []
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, ATSBlock) or isinstance(blk, TokenSampling):
                if self.integrate_attn:
                    idx = self.stages.index(i)
                else:
                    idx = self.stages.index(i - 1)
                # x, attn, blk_cls = blk(x, self.num_tokens[idx])
                x, top_tokens, blk_cls, policy, cls_attn, qk = blk(x, self.num_tokens[idx], policy)
                cls_attn_list.append(cls_attn)
                qk_list.append(qk)

                policies.append(policy)
                attn = top_tokens
                ats_cls.append(blk_cls)
            else:
                x = blk(x, policy)

        feature = self.norm(x)
        if policy is not None:
            feature = feature * policy
        cls = feature[:, 0]
        tokens = feature[:, 1:]
        cls = self.pre_logits(cls)
        cls = self.head(cls)
        return cls, tokens, attn, ats_cls, policies, cls_attn_list, qk_list
