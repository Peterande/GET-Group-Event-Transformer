# ==============================================================================
# GET: Group Event Transformer.
# Copyright (c) 2023 The Group Event Transformer Authors.
# Licensed under The MIT License.
# Written by Yansong Peng.
# ==============================================================================
# GET   [Stage 1 → Stage 2 ... → Stage N]
# Stage [GTE (1st. stage only) → Blocks → Head (last stage only)]
# Block [EDSA → GTA (last block only)]
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.utils.checkpoint as checkpoint

from torch import Tensor, Size
from typing import Union, List
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import create_conv2d, create_pool2d
from utils.utils import load_checkpoint


_shape_t = Union[int, List[int], Size]
    
    
def custom_normalize(input, p=2, dim=1, eps=1e-12, out=None):
    if out is None:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return input / (denom + eps)
    else:
        denom = input.norm(p, dim, keepdim=True).expand_as(input)
        return torch.div(input, denom + eps, out=out)


class LayerNormFP32(nn.LayerNorm):
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormFP32, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)


class LinearFP32(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearFP32, self).__init__(in_features, out_features, bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input.float(), self.weight.float(),
                        self.bias.float() if self.bias is not None else None)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, pad_type='same', kernel_size=3):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad_type, bias=True)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        x = self.conv(x)
        return x  # (B, 2C, H, W)


class Pool(nn.Module):
    def __init__(self, pad_type='same', kernel_size=3):
        super().__init__()
        self.pool = create_pool2d('max', kernel_size=kernel_size, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1,
                 norm_layer=None, mlpfp32=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x):
        x = self.fc1(x)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32))
            x = self.drop.float()(x)
        else:
            x = self.fc2(x)
            x = self.drop(x)
        return x


class Group_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1,
                 norm_layer=None, mlpfp32=False, group_num=12):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlpfp32 = mlpfp32

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), groups=group_num)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1, 1), groups=group_num)
        self.drop = nn.Dropout(drop)
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x):
        need_reset_shape = False
        if x.dim() != 4:
            need_reset_shape = True
            hw = int(x.shape[1] ** 0.5)
            x = x.view(x.shape[0], hw, hw, -1)
        x = self.fc1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        if self.norm:
            x = self.norm(x)
        x = self.act(x)
        if self.mlpfp32:
            x = self.fc2.float()(x.type(torch.float32).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = self.drop.float()(x)
        else:
            x = self.fc2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
            x = self.drop(x)
        if need_reset_shape:
            x = x.view(x.shape[0], -1, x.shape[-1])
        return x
    

class GTA(nn.Module):
    """ Group Token Aggregation Layer.
    Including Overlapping Group Convolution and Max Pooling.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        group_num (int): Number of groups.
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, group_num=12, embed_dim=48):
        super().__init__()
        self.dim = dim
        mul = dim // embed_dim
        self.channelexpand = OGConv(dim, 2 * dim, norm_layer=norm_layer, group_num=group_num // mul)
        self.pool = Pool()

    def forward(self, x, H, W):
        """
        x is expected to have shape (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # convpool
        x = self.channelexpand(x)
        x = self.pool(x)

        x = x.permute(0, 2, 3, 1).reshape(B, -1, 2*C)
        return x  # (B, H//2*W//2, 2C)
    
    
class OGConv(nn.Module):
    def __init__(self, in_chans, embed_dim, kernel=3, norm_layer=None, group_num=None):
        super().__init__()
        
        self.in_chans = kernel * in_chans // group_num
        self.step = (kernel - 1) * in_chans // group_num
        self.new_group_num = group_num // 2
        self.padding = ((1 - 1) + 1 * (3 - 1)) // 2
        self.conv = nn.Conv2d(
            in_channels=self.new_group_num * self.in_chans, out_channels=embed_dim,
            kernel_size=(3, 3), groups=self.new_group_num, padding=self.padding)
        self.norm = norm_layer(embed_dim)

    def forward(self, x):  # (B C H W)
        """
        Overlapping group convolution layer.
        """
        B, C, H, W = x.shape
        pad_c = ((self.new_group_num - 1) * (C // self.new_group_num) + self.in_chans - C)
        x = torch.cat([x[:, :pad_c], x], dim=1)
        x_grouped = x.unfold(1, self.in_chans, self.step).permute(0, 1, 4, 2, 3)
        x_grouped = x_grouped.reshape(B, self.new_group_num * self.in_chans, H, W)
        out = self.conv(x_grouped)
        y = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return y  # (B C H W) 
    

class GTE(nn.Module):
    def __init__(self, in_chans, hidden_dim, embed_dim, norm_layer=None, group_num=None, kernel_size=(3, 3)):
        super().__init__()

        self.in_chans = in_chans
        padding = ((1 - 1) + 1 * (3 - 1)) // 2
        self.conv = nn.Conv2d(in_chans, hidden_dim // 4, kernel_size=kernel_size, groups=group_num, padding=padding)
        self.mlp = Group_Mlp(hidden_dim // 4, hidden_dim, embed_dim, norm_layer=norm_layer, group_num=1)
        
    def forward(self, x):  # (B C H W)
        """
        Embed 'Group Tokens' using group convolution and MLP.
        """
        x = self.conv(x)
        y = x.permute(0, 2, 3, 1).contiguous()
        y = self.mlp(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y  # (B C H W)
    

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., rpe_hidden_dim=512):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # mlp to generate table of relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                     nn.ReLU(inplace=True),
                                     LinearFP32(rpe_hidden_dim, num_heads, bias=False))
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))\
                                .permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                    torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # log8
        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = custom_normalize(q.float(), dim=-1, eps=5e-5)
        k = custom_normalize(k.float(), dim=-1, eps=5e-5)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale.float()

        # relative_position_bias_table: 2*Wh-1 * 2*Ww-1, nH
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = attn.type_as(x)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (num_windows*B, N, C)


class GroupAttention(nn.Module):
    """ Group based W-MSA module with relative group bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, attn_drop=0., proj_drop=0., rpe_hidden_dim=512, group_num=12, embed_dim=48):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads_i = window_size[0]
        
        # get relative_coords_table
        self.group = group_num // (dim // embed_dim)  # Current group number.
        self.dpg = dim // self.group  # Dimension per group.
        
        self.cpb_mlp_i = nn.Sequential(nn.Linear(2, rpe_hidden_dim, bias=True),
                                    nn.ReLU(inplace=True),
                                    LinearFP32(rpe_hidden_dim, self.num_heads_i, bias=False))
        relative_coords_dpg = torch.arange(-(self.dpg - 1), self.dpg, dtype=torch.float32)
        relative_coords_group = torch.arange(-(self.group - 1), self.group, dtype=torch.float32)
        relative_coords_table_i = torch.stack(torch.meshgrid([relative_coords_dpg, relative_coords_group])) \
            .permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*dpg-1, 2*group-1, 2
        relative_coords_table_i[:, :, :, 0] = relative_coords_table_i[:, :, :, 0] / (self.dpg - 1) if self.dpg > 1 else 0
        relative_coords_table_i[:, :, :, 1] = relative_coords_table_i[:, :, :, 1] / (self.group - 1) if self.group > 1 else 0
        relative_coords_table_i *= 8  # normalize to -8, 8
        relative_coords_table_i = torch.sign(relative_coords_table_i) * torch.log2(
            torch.abs(relative_coords_table_i) + 1.0) / np.log2(8)  # log8
        self.register_buffer("relative_coords_table_i", relative_coords_table_i)

        # get pair-wise relative position index for each group.
        coords_dpg = torch.arange(self.dpg)
        coords_div = torch.arange(self.group)
        coords_i = torch.stack(torch.meshgrid([coords_dpg, coords_div]))  # 2, dpg, group
        coords_flatten_i = torch.flatten(coords_i, 1)  # 2, C
        relative_coords_i = coords_flatten_i[:, :, None] - coords_flatten_i[:, None, :]  # 2, C, C
        relative_coords_i = relative_coords_i.permute(1, 2, 0).contiguous()  # C, C, 2
        relative_coords_i[:, :, 0] += self.dpg - 1  # shift to start from 0
        relative_coords_i[:, :, 1] += self.group - 1
        relative_coords_i[:, :, 0] *= 2 * self.group - 1
        relative_position_index_i = relative_coords_i.sum(-1)  # C, C
        self.register_buffer("relative_position_index_i", relative_position_index_i)

        self.qkv_i = nn.Linear(self.window_size[0] * self.window_size[1],
                               self.window_size[0] * self.window_size[1] * 3, bias=False)

        self.q_i_bias = nn.Parameter(torch.zeros(self.window_size[0] * self.window_size[1]))
        self.v_i_bias = nn.Parameter(torch.zeros(self.window_size[0] * self.window_size[1]))

        self.attn_drop_i = nn.Dropout(attn_drop)
        self.proj_i = nn.Linear(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1])
        self.proj_drop_i = nn.Dropout(proj_drop)
        self.softmax_i = nn.Softmax(dim=-1)


    def forward(self, x_i):
        """
        Args:
            x_i: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x_i.shape

        qkv_i_bias = None
        if self.q_i_bias is not None:
            qkv_i_bias = torch.cat(
                (self.q_i_bias, torch.zeros_like(self.v_i_bias, requires_grad=False), self.v_i_bias))
        qkv_i = F.linear(input=x_i.permute(0, 2, 1), weight=self.qkv_i.weight, bias=qkv_i_bias)
        qkv_i = qkv_i.reshape(B_, C, 3, self.num_heads_i, -1).permute(2, 0, 3, 1, 4)
        # qkv_i = self.qkv_i(x).reshape(B_, C, 3, self.num_heads, N // self.num_heads).permute(2, 0, 3, 1, 4)
        q_i, k_i, v_i = qkv_i[0], qkv_i[1], qkv_i[2]
        q_i = custom_normalize(q_i.float(), dim=-1, eps=5e-5)
        k_i = custom_normalize(k_i.float(), dim=-1, eps=5e-5)
        #logit_scale_i = torch.clamp(self.logit_scale_i, max=torch.log(torch.tensor(1. / 0.01, device=x_i.device))).exp()
        attn_i = (q_i @ k_i.transpose(-2, -1)) #* logit_scale_i.float()
        
        relative_position_bias_table_i = self.cpb_mlp_i(self.relative_coords_table_i).view(-1, self.num_heads_i)
        relative_position_bias_i = relative_position_bias_table_i[self.relative_position_index_i.view(-1)].view(
            C, C, -1)  # C,C,nH
        relative_position_bias_i = relative_position_bias_i.permute(2, 0, 1).contiguous()  # nH, C, C
        relative_position_bias_i = 16 * torch.sigmoid(relative_position_bias_i)
        attn_i = attn_i + relative_position_bias_i.unsqueeze(0)

        attn_i = self.softmax_i(attn_i)
        attn_i = attn_i.type_as(x_i)
        attn_i = self.attn_drop_i(attn_i)
        x_i = (attn_i @ v_i).transpose(1, 2).reshape(B_, C, N)

        x_i = self.proj_i(x_i)
        x_i = self.proj_drop_i(x_i)
        return x_i.transpose(1, 2)  # (num_windows*B, N, C)
    

class EDSABlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 use_mlp_norm=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rpe_hidden_dim=512, group_num=12, embed_dim=48):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = 0  # Shift is not necessary here.
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm = use_mlp_norm
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        
        self.attn_spatial = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop,
            rpe_hidden_dim=rpe_hidden_dim)
        self.attn_temporal = GroupAttention(
            dim, window_size=to_2tuple(self.window_size), attn_drop=attn_drop, proj_drop=drop, rpe_hidden_dim=rpe_hidden_dim, 
            group_num=group_num, embed_dim=embed_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Group_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_layer=norm_layer if self.use_mlp_norm else None, group_num=1)
        
        self.H = None
        self.W = None

    def forward(self, x, x_i, mask_matrix):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, with L = {L}, H = {H}, W = {W}"

        # dual shortcuts
        shortcut = x.clone()
        shortcut_i = x_i.clone()
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # WSA and G-WSA
        attn_windows = self.attn_spatial(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        attn_windows_i = self.attn_temporal(attn_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        attn_windows_i = attn_windows_i.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        shifted_x_i = window_reverse(attn_windows_i, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_i = torch.roll(shifted_x_i, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            x_i = shifted_x_i

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            x_i = x_i[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x_i = x_i.view(B, H * W, C)

        # FFN
        x = self.norm1(x)
        x_i = self.norm2(x_i)

        # dual residual connections
        x = shortcut + self.drop_path(x)
        x_i = shortcut_i + self.drop_path(x_i)
        shortcut = x.clone()
        
        x = self.mlp(x + x_i)
        x = self.norm3(x)
        x = shortcut + self.drop_path(x)
        return x, x_i


class BasicLayer(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 checkpoint_blocks=255,
                 init_values=None,
                 use_mlp_norm=False,
                 use_shift=True,
                 rpe_hidden_dim=512,
                 group_num=12,
                 embed_dim=48):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.checkpoint_blocks = checkpoint_blocks
        self.init_values = init_values if init_values is not None else 0.0

        # build blocks
        self.blocks = nn.ModuleList([
                EDSABlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) or (not use_shift) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_mlp_norm=use_mlp_norm,
                    rpe_hidden_dim=rpe_hidden_dim,
                    group_num=group_num, 
                    embed_dim=embed_dim)
                for i in range(depth)])

        # downsample layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, group_num=group_num, embed_dim=embed_dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        x_i = x
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint and self.training:
                x, x_i = checkpoint.checkpoint(blk, x, x_i, attn_mask)
            else:
                x, x_i = blk(x, x_i, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def _init_block_norm_weights(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, self.init_values)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, self.init_values)


class GET(nn.Module):

    def __init__(self,
                 patch_size=4,
                 num_classes=10,
                 embed_dim=48,
                 depths=[2, 2, 8],
                 num_heads=[3, 6, 12],
                 window_size=8,
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1,
                 norm_layer=partial(LayerNormFP32, eps=1e-6),
                 use_checkpoint=False,
                 init_values=1e-5,
                 use_mlp_norm_layers=[],
                 rpe_hidden_dim=512,
                 frozen_stages=-1,
                 use_shift=True,
                 embed_split=24,
                 group_num=12):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.mlp_ratio = mlp_ratio
        self.use_mlp_norm_layers = use_mlp_norm_layers
        self.rpe_hidden_dim = rpe_hidden_dim

        if isinstance(window_size, list):
            pass
        elif isinstance(window_size, int):
            window_size = [window_size] * self.num_layers
        else:
            raise TypeError("We only support list or int for window size")

        if isinstance(use_shift, list):
            pass
        elif isinstance(use_shift, bool):
            use_shift = [use_shift] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_shift")

        if isinstance(use_checkpoint, list):
            pass
        elif isinstance(use_checkpoint, bool):
            use_checkpoint = [use_checkpoint] * self.num_layers
        else:
            raise TypeError("We only support list or bool for use_checkpoint")

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        num_features = []
        for i_layer in range(self.num_layers):
            cur_dim = int(embed_dim * 2 ** i_layer)
            num_features.append(cur_dim)
            if i_layer <= self.num_layers - 2:
                cur_downsample_layer = GTA
            else:
                cur_downsample_layer = None
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=cur_downsample_layer,
                use_checkpoint=use_checkpoint[i_layer],
                init_values=init_values,
                use_mlp_norm=True if i_layer in use_mlp_norm_layers else False,
                use_shift=use_shift[i_layer],
                rpe_hidden_dim=self.rpe_hidden_dim,
                group_num=group_num,
                embed_dim=embed_dim
            )
            self.layers.append(layer)

        self.num_features = num_features
        
        self.norm = norm_layer(num_features[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_features[-1], num_classes) if num_classes > 0 else nn.Identity()
        self._freeze_stages()
        
        # init group token embedding layer
        self.num_features = num_features
        input_dim = 2 * embed_split * int(patch_size ** 2)
        hidden_dim = int(group_num * 64 / (group_num / 12))
        kernel_size = (3, 3) if patch_size == 4 else (7, 7)
        self.channel_embed = GTE(input_dim, hidden_dim, embed_dim, 
                                          norm_layer=norm_layer, group_num=group_num, kernel_size=kernel_size)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        for bly in self.layers:
            bly._init_block_norm_weights()

        if isinstance(pretrained, str):
            # logger = get_root_logger()
            logger = None
            load_checkpoint(self, pretrained, strict=False, map_location='cpu', logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.channel_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        layers_out = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer.use_checkpoint = False if i==0 else layer.use_checkpoint  # disable checkpoint for the first layer
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
        layers_out.append(x_out)

        x_last = self.norm(layers_out[-1])  # (B, L, C)

        pool = self.avgpool(x_last.transpose(1, 2))  # (B, C, 1)
        outs = self.head(torch.flatten(pool, 1))
        return outs


    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(GET, self).train(mode)
        self._freeze_stages()
