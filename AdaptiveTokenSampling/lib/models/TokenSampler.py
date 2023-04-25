import pdb
from collections import OrderedDict

from torch import Tensor

from vit import Attention, Block, Mlp
from lvvit import GroupLinear
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
import numpy as np




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
        drop_path=0.0,
    ):
        super(TokenSampling, self).__init__(
            dim_in, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        # self.norm1 = norm_layer(dim_in)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        #self.score_biases = nn.Parameter(torch.zeros(1, 197))  # [1 x T]
        #self.bin_sizes = nn.Parameter(torch.rand(1, 103))  # [1 x n-1]

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
    def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        Sample uniformly from y-axis.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
            torch.range(0, n_tokens - 2, device=normalized_cdf.device)
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

    def inverse_transform_sampling(
        self,
        sorted_scores: Tensor,
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
    ) -> (Tensor, Tensor):
        """
        Sample tokens based on their significance scores.
        """
        B, N, C = raw_x.shape

        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 1)
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 1, N),
        )  # [B x h x T-1 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS token.
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=N - 1
        )[:, : N - 1]

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(B, self.num_heads, n_tokens - 1, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        )

        attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp

        sampler = torch.nonzero(policy)

        return selected_x, attn, policy, sampler


    def forward(self, x, n, raw_x, policy):
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
        qkv = qkv * policy.unsqueeze(0).unsqueeze(2)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        # cls_attn = attn_no_softmax[:, :, 0]
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T]
        cls_attn = attn[:, :, 0]
        if True:
            v_norm_2 = torch.linalg.norm(
                v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
            )  # [B x T]
            selection_score = attn[:, :, 0].sum(dim=1)  # [B x T]



            selection_score = selection_score * v_norm_2  # [B x T]

            selection_score = selection_score[:, 1:]  # [B x T-1]

            selection_score = selection_score / selection_score.sum(
                dim=1, keepdim=True
            )  # [B x T-1]

            # selection_score = torch.ones_like(selection_score).to(selection_score.device)/selection_score.shape[1]
            # if adv_training:
            #     return selection_score

            sorted_scores, sorted_ind = torch.sort(
                selection_score, descending=False, dim=1
            )

            selected_x, attn, policy, sampler = self.inverse_transform_sampling(
                sorted_scores, sorted_ind, attn, n, raw_x, n
            )







        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[2], C)




        # -----------------------------

        x = self.proj(x) * policy
        x = self.proj_drop(x)

        return x, selected_x, None, policy, cls_attn, [q,k]







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
            proj_drop=0.0,
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

    def forward(self, x, n, policy=None):

        x_out, top_tokens, masks, policy, cls_attn, qk = self.attn(self.norm1(x), n, x, policy)
        selected_x = (
            top_tokens  # torch.gather(x, 1, top_tokens_expanded)  # [B x N x C]
        )
        x = selected_x + self.drop_path(x_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x * policy

        return x, top_tokens, None, policy, cls_attn, qk













