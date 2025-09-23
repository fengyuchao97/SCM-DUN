import math
import warnings

import numbers

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

from einops import rearrange
from torch import einsum

from box import Box
from fvcore.nn import FlopCountAnalysis

from csi.data import shift_batch, shift_back_batch, gen_meas_torch_batch
# import swattention
from timm.models.layers import DropPath, trunc_normal_, drop_path
from mamba_ssm import Mamba, Mamba_rope, Mamba2
import numpy as np
import pandas as pd
from einops import repeat
from csi.Hilbert.Hilbert3d import Hilbert3d
from csi.architectures.MambaNO.CNOModule import LiftProjectBlock

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LocalMSA(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """
    def __init__(self, 
                 dim, 
                 num_heads, 
                 window_size, 
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(torch.Tensor(1, num_heads, window_size[0]*window_size[1], window_size[0]*window_size[1]))
        trunc_normal_(self.pos_emb)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                              b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0], w=w // self.window_size[1],
                            b0=self.window_size[0])
        out = self.project_out(out)
        
        return out
    
class NonLocalMSA(nn.Module):
    """
    The Non-Local MSA divides the input into N × N non-overlapping windows, treating each window as a token, and computes self-attention across the windows.
    """
    def __init__(self, 
                 dim, 
                 num_heads, 
                 window_num 
    ):
        super().__init__()
        self.dim = dim
        self.window_num = window_num
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5


        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)


        self.pos_emb = nn.Parameter(torch.Tensor(1, num_heads, window_num[0]*window_num[1], window_num[0]*window_num[1]))
        trunc_normal_(self.pos_emb)


    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1)-> b (h w) (b0 b1 c)',
                                              h=self.window_num[0], w=self.window_num[1]), (q, k, v))
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        head_dim = ((h // self.window_num[0]) * (w // self.window_num[1]) * c) / self.num_heads 
        scale = head_dim ** -0.5

        q *= scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
       
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b (h w) (b0 b1 c) -> (b h w) (b0 b1) c', h=self.window_num[0], b0=h // self.window_num[0], b1=w // self.window_num[1])

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=self.window_num[0], w= self.window_num[1],
                            b0=h//self.window_num[0])
        out = self.project_out(out)
        
        
        return out
    

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        out = self.net(x)
        return out
    

## Gated-Dconv Feed-Forward Network (GDFN)
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, 
                 dim, 
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x
    

def FFN_FN(
    cfg,
    ffn_name,
    dim
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
                dim, 
                ffn_expansion_factor=cfg.MODEL.DENOISER.SCMAMBA.FFN_EXPAND, 
            )
    elif ffn_name == "FeedForward":
        return FeedForward(dim = dim)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)
    
class MambaLayer_temp(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3):
        super().__init__()
        print(f"MambaLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,1,2]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            assert d_model == self.dim
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]
            x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
            x_norm=self.norm(x_flat)
            # x_norm = self.norm(x_flat)
            # torch.cuda.empty_cache()
            x_mamba = self.mamba(x_norm)
            # x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
            # x_mamba = x_mamba + x_mamba_b
            out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=three_d_tensors_out
                    elif mode == 1:
                        tensor_out=three_d_tensors_out.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=three_d_tensors_out.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)


            mode_list.append(reconstructed_x_out)

        return out

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3):
        super().__init__()
        print(f"Transformer: dim: {dim}")
        print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_m)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.mamba2 = Mamba(
                d_model=dim_m, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,1,2]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            # assert d_model == self.dim
            d_model=C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]
            x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
            if mode == 0:    
                x_norm=self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)   
            elif mode == 1 or mode == 2:
                x_norm=self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)  


            # x_mamba_b = self.mamba_b(x_norm.flip(1)).flip(1)
            # x_mamba = x_mamba + x_mamba_b
            out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=tensor_out_last
                    elif mode == 1:
                        tensor_out=tensor_out_last.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=tensor_out_last.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)
        
        mean = (mode_list[0] + mode_list[1] + mode_list[2])/3


        return mean

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer_hilbert(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, drop_path=0., channel_token = False, mode_temp=3):
        super().__init__()
        print(f"Transformer: dim: {dim}")
        print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        self.norm = nn.LayerNorm(64)
        self.mamba = Mamba(
                d_model=64, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_patch_token(self, x, hilbert_curve1=None, r=False):
        x_perm = x  # x_perm 的形状为 (B, C, H, W)
        hilbert_curve = hilbert_curve1

        B, C, H, W = x_perm.shape
        channel_division = 4  # 通道划分数
        num_partitions = 4    # 分块数

        assert H % num_partitions == 0 and W % num_partitions == 0, "H and W must be divisible by num_partitions"
        assert C % channel_division == 0, "C must be divisible by channel_division"

        # 分割 C 维度
        x_C_splits = torch.split(x_perm, C // channel_division, dim=1)  # 列表，元素形状为 (B, C//channel_division, H, W)

        # 分割 H 和 W 维度
        x_patches = []
        for x_c in x_C_splits:
            x_H_splits = torch.split(x_c, H // num_partitions, dim=2)
            for x_h in x_H_splits:
                x_W_splits = torch.split(x_h, W // num_partitions, dim=3)
                for x_w in x_W_splits:
                    x_patches.append(x_w)  # 每个 x_w 的形状为 (B, C//channel_division, H//num_partitions, W//num_partitions)

        # 将分块后的张量堆叠在新的维度上
        x_patches = torch.stack(x_patches, dim=1)  # 形状：(B, T, d_model, h, w)
        T = x_patches.shape[1]  # T = channel_division * num_partitions * num_partitions

        B, T, d_model, h, w = x_patches.shape
        n_tokens_per_patch = h * w
        n_tokens = d_model * n_tokens_per_patch  # n_tokens = d_model * n_tokens_per_patch

        # 展平并调整维度
        x_patches_flat = x_patches.view(B, T, d_model, n_tokens_per_patch)  # 形状：(B, T, d_model, n_tokens_per_patch)
        x_patches_flat = x_patches_flat.permute(0, 3, 2, 1).contiguous()    # 形状：(B, n_tokens_per_patch, d_model, T)
        x_hil = x_patches_flat.view(B, n_tokens, T)                         # 形状：(B, n_tokens, T)

        # 应用 Hilbert 曲线索引（如果需要）
        if hilbert_curve is not None:
            x_hil = x_hil.index_select(dim=1, index=hilbert_curve)
        else:
            x_hil = x_hil

        # 通过 Mamba 模块处理
        x_mamba = x_hil + self.drop_path(self.mamba(self.norm(x_hil)))  # x_mamba 形状：(B, n_tokens, T)

        # 反向 Hilbert 曲线索引
        if hilbert_curve is not None:
            sum_out = torch.zeros_like(x_mamba)
            hilbert_curve_re = hilbert_curve.unsqueeze(0).unsqueeze(-1).expand(B, -1, T)
            sum_out.scatter_add_(dim=1, index=hilbert_curve_re, src=x_mamba)
        else:
            sum_out = x_mamba

        # 恢复到分块形状
        x_patches_reconstructed = sum_out.view(B, n_tokens_per_patch, d_model, T)
        x_patches_reconstructed = x_patches_reconstructed.permute(0, 3, 2, 1).contiguous()  # 形状：(B, T, d_model, n_tokens_per_patch)
        x_patches_reconstructed = x_patches_reconstructed.view(B, T, d_model, h, w)         # 形状：(B, T, d_model, h, w)

        # 重建原始张量
        idx = 0
        x_reconstructed_C_splits = []
        for c in range(channel_division):
            c_patches = []
            for h_idx in range(num_partitions):
                h_patches = []
                for w_idx in range(num_partitions):
                    patch = x_patches_reconstructed[:, idx, :, :, :]  # 形状：(B, d_model, h, w)
                    h_patches.append(patch)
                    idx += 1
                h_concat = torch.cat(h_patches, dim=3)  # 在 W 维度拼接
                c_patches.append(h_concat)
            c_concat = torch.cat(c_patches, dim=2)  # 在 H 维度拼接
            x_reconstructed_C_splits.append(c_concat)
        x_reconstructed = torch.cat(x_reconstructed_C_splits, dim=1)  # 在 C 维度拼接

        x_out = x_reconstructed  # 形状：(B, C, H, W)

        return x_out

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x, hilbert_curve1=None, r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, hilbert_curve1, r=r)

        return out

def SoftShrink_GPU(S, tau):
    # S: [num_patches, k], dtype: torch.float32
    return torch.sign(S) * torch.clamp(torch.abs(S) - tau, min=0.0)

def SVDShrink_Fast_GPU(X, tau=0.1, patch_size=8):
    # X: [N, C, H, W]
    N, C, H, W = X.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H 和 W 必须能被 patch_size 整除"
    
    # 对输入进行二维 FFT
    D = torch.fft.fft2(X, dim=(-2, -1))
    D = D.to(torch.complex64)
    
    # 将图像划分为小块，并重塑为批量形式
    D_patches = D.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    D_patches = D_patches.contiguous().view(N * C * (H // patch_size) * (W // patch_size), patch_size, patch_size)
    # 现在 D_patches 的形状是 [num_patches, patch_size, patch_size]
    
    # 对所有小块批量执行 SVD
    U, S, Vh = torch.linalg.svd(D_patches, full_matrices=False)
    # U: [num_patches, patch_size, patch_size]
    # S: [num_patches, patch_size]
    # Vh: [num_patches, patch_size, patch_size]
    
    # 对奇异值进行软阈值收缩
    S_shrink = SoftShrink_GPU(S, tau)
    
    # 将收缩后的奇异值转换为对角矩阵
    S_shrink_diag = torch.diag_embed(S_shrink).to(U.dtype)
    # S_shrink_diag: [num_patches, patch_size, patch_size]
    
    # 重构小块
    D_reconstructed_patches = torch.matmul(U, torch.matmul(S_shrink_diag, Vh))
    # D_reconstructed_patches: [num_patches, patch_size, patch_size]
    
    # 将重构的小块还原为原始形状
    D_reconstructed = D_reconstructed_patches.view(N, C, H // patch_size, W // patch_size, patch_size, patch_size)
    D_reconstructed = D_reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous()
    D_reconstructed = D_reconstructed.view(N, C, H, W)
    
    # 执行二维逆 FFT，恢复时域数据
    X_reconstructed = torch.fft.ifft2(D_reconstructed, dim=(-2, -1)).real
    return X_reconstructed


from mmcv.cnn import build_norm_layer

class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, bias=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=bias))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

from mmcv.cnn.bricks import ConvModule

class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias

        self.dwconv1_1 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                   groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.dwconv1_2 = Conv2d_BN(dim, dim, ks=3, stride=1, pad=1, dilation=1,
                                   groups=dim, norm_cfg=dict(type='BN2d', requires_grad=True))
        self.act = nn.ReLU6()

        self.weight1 = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))

        # 将 BatchNorm2d 替换为 LayerNorm 或者直接移除归一化层
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1),
        )

        if bias:
            self.bias1 = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias1 = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight1, std=0.02)
        if self.bias_type is not None:
            nn.init.trunc_normal_(self.bias1, std=0.02)

    def forward(self, x1):
        B, C, H, W = x1.shape
        x_fuse1 = self.act(self.dwconv1_1(x1)) * self.dwconv1_2(x1)

        # 处理权重
        pooled_x = self.pool(x_fuse1)
        scale1 = self.proj1(pooled_x).reshape(B, self.num_groups, C, self.K, self.K)
        scale1 = torch.softmax(scale1, dim=1)
        weight1 = scale1 * self.weight1.unsqueeze(0)
        weight1 = torch.sum(weight1, dim=1)
        weight1 = weight1.reshape(-1, 1, self.K, self.K)

        # 处理偏置
        if self.bias_type is not None:
            mean_x = torch.mean(x_fuse1, dim=[-2, -1], keepdim=True)
            scale1_bias = self.proj1(mean_x).reshape(B, self.num_groups, C)
            scale1_bias = torch.softmax(scale1_bias, dim=1)
            bias1 = scale1_bias * self.bias1.unsqueeze(0)
            bias1 = torch.sum(bias1, dim=1).flatten(0)
        else:
            bias1 = None

        x_out1 = F.conv2d(x_fuse1.reshape(1, -1, H, W),
                          weight=weight1,
                          padding=self.K // 2,
                          groups=B * C,
                          bias=bias1)

        return x_out1.reshape(B, C, H, W)
    
class Mamba_global(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, drop_path=0., channel_token = False, mode_temp=3, num_tokens=1):
        super(Mamba_global, self).__init__()

        self.global_token = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.dim = dim
        self.Mamba_c = Mamba(
            d_model=dim,    # Model dimension d_model
            d_state=16,     # SSM state expansion factor (Mamba: d_state=16)
            d_conv=4,       # Local convolution width
            expand=2,       # Block expansion factor
            # bimamba= True,
        )

    def forward(self, x):
        B, C, H, W = x.shape    
        x = x.flatten(2).transpose(1, 2)  
        global_token = self.global_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((global_token, x), dim=1)   
        x = self.Mamba_c(x)
        x = x[:, -H*W:, :]  
        out = x.view(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()  
        return out

class Mamba_3dglobal(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, drop_path=0., channel_token = False, mode_temp=3, num_tokens=1, upscale_factor=2):
        super(Mamba_3dglobal, self).__init__()

        self.dim = dim
        self.upscale_factor = upscale_factor
        self.upscale_factor_pow = upscale_factor ** 2
        self.global_token = nn.Parameter(torch.zeros(1, num_tokens, dim//self.upscale_factor_pow))

        self.Mamba_c = Mamba_rope(
            d_model=dim//self.upscale_factor_pow,    # Model dimension d_model
            d_state=16,     # SSM state expansion factor (Mamba: d_state=16)
            d_conv=4,       # Local convolution width
            expand=2,       # Block expansion factor
            # bimamba= True,
            # use_fast_path=False,
        )

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.pixelunshuffle = nn.PixelUnshuffle(self.upscale_factor)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.pixelshuffle(x)    # B, C, H, W -> B, C//4, 2H, 2W

        x = x.flatten(2).transpose(1, 2)  
        global_token = self.global_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((global_token, x), dim=1)   
        x = self.Mamba_c(x)
        # x = self.Mamba_c(x, H*self.upscale_factor, W*self.upscale_factor)
        x = x[:, -H*W*self.upscale_factor_pow:, :]  
        out = x.view(B, H*self.upscale_factor, W*self.upscale_factor, self.dim//self.upscale_factor_pow).permute(0, 3, 1, 2).contiguous()  

        out = self.pixelunshuffle(out)
        return out
    
class MambaLayer_local(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3, num_tokens=1, local_numbers=4, column_first=False):
        super().__init__()
        print(f"Transformer: dim: {dim}")
        print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_m)
        self.mamba = Mamba_rope(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.mamba2 = Mamba_rope(
                d_model=dim_m, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

        self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m))
        self.local_numbers = local_numbers
        self.column_first = column_first

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,1,2]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            # assert d_model == self.dim
            d_model=C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]

            if self.local_numbers > 4 and mode !=0:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//4, 4, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 5, 3, 6, 4).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//4, 4, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
            else:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 5, 3, 6, 4).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)

            if mode == 0:    
                global_token = self.global_token1.expand(x.shape[0], -1, -1)    # 2 1 28
                x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)   
            elif mode == 1 or mode == 2:
                global_token = self.global_token2.expand(x.shape[0], -1, -1)    # 2 1 28
                x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)  

            x_mamba = x_mamba[:, -n_tokens:, :]  

            if self.local_numbers > 4 and mode !=0:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, W//self.local_numbers, H//4, self.local_numbers, 4).permute(0, 1, 2, 4, 6, 3, 5).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, H//4, W//self.local_numbers, 4, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, *img_dims).contiguous()
            else:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, W//self.local_numbers, H//self.local_numbers, self.local_numbers, self.local_numbers).permute(0, 1, 2, 4, 6, 3, 5).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, H//self.local_numbers, W//self.local_numbers, self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=tensor_out_last
                    elif mode == 1:
                        tensor_out=tensor_out_last.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=tensor_out_last.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)
        
        mean = (mode_list[0] + mode_list[1] + mode_list[2])/3

        return mean

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer_global(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3, num_tokens=1, use_fast_path=True):
        super().__init__()
        print(f"Transformer: dim: {dim}")
        print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim_m)
        self.mamba = Mamba( #Mamba_rope(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                use_fast_path=use_fast_path,
        )
        self.mamba2 = Mamba( #Mamba_rope(
                d_model=dim_m, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                use_fast_path=use_fast_path,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

        # self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim))
        # self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m))
        # self.conv = nn.Conv2d(dim*3, dim, 1, padding=0, bias=True)
        
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,2]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            # assert d_model == self.dim
            d_model=C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]

            x_flat = x.reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)

            if mode == 0:    
                # global_token = self.global_token1.expand(x.shape[0], -1, -1)    # 2 1 28
                # x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)   
            elif mode == 1 or mode == 2:
                # global_token = self.global_token2.expand(x.shape[0], -1, -1)    # 2 1 28
                # x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)  

            # x_mamba = x_mamba[:, -n_tokens:, :]  

            out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=tensor_out_last
                    elif mode == 1:
                        tensor_out=tensor_out_last.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=tensor_out_last.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)
        
        # mean = (mode_list[0] + mode_list[1])/2
        # mean = (mode_list[0] + mode_list[1] + mode_list[2])/3

        # mean = self.conv(torch.cat([mode_list[0],mode_list[1],mode_list[2]],dim=1))
        
        # mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1]+self.alpha3*mode_list[2])/3
        
        mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1])/2
        return mean

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out
    
class MambaLayer_3dlocalblock(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3, num_tokens=1, channel_numbers=2, local_numbers=2, column_first=False, upscale_factor=2):
        super().__init__()
        self.dim = dim
        self.upscale_factor = upscale_factor
        self.upscale_factor_pow = upscale_factor ** 2
        self.channel_numbers = channel_numbers
        self.local_numbers = local_numbers

        if self.channel_numbers > self.local_numbers:
            self.blocks1 = self.channel_numbers * self.local_numbers
        else: self.blocks1 = self.local_numbers * self.local_numbers

        self.blocks2 = self.local_numbers * self.local_numbers

        self.norm = nn.LayerNorm(dim//self.upscale_factor_pow*self.blocks2)
        self.norm2 = nn.LayerNorm(dim_m*self.blocks1)
        self.mamba = Mamba_rope(
                d_model=dim//self.upscale_factor_pow*self.blocks2, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                # use_fast_path=False,
        )
        self.mamba2 = Mamba_rope(
                d_model=dim_m*self.blocks1, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                # use_fast_path=False,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

        self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim//self.upscale_factor_pow*self.blocks2))
        self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m*self.blocks1))
        self.column_first = column_first

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.pixelunshuffle = nn.PixelUnshuffle(self.upscale_factor)

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,1,2]
        # mode_temp = [0]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            if mode == 0:    
                x = self.pixelshuffle(reconstructed_x).unsqueeze(2) # B, C, H, W -> B, C//4, H*2, W*2
            else: x = reconstructed_x.unsqueeze(2)

            # x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            # assert d_model == self.dim
            d_model=C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]

            self.channel_numbers = 7
            
            if self.channel_numbers > self.local_numbers and mode !=0:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//self.channel_numbers, self.channel_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 6, 4, 2, 5, 3).reshape(B, d_model*self.blocks1, n_tokens//self.blocks1).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//self.channel_numbers, self.channel_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 4, 6, 2, 3, 5).reshape(B, d_model*self.blocks1, n_tokens//self.blocks1).contiguous().transpose(-1, -2)
            else:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 6, 4, 2, 5, 3).reshape(B, d_model*self.blocks2, n_tokens//self.blocks2).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 4, 6, 2, 3, 5).reshape(B, d_model*self.blocks2, n_tokens//self.blocks2).contiguous().transpose(-1, -2)

            if mode == 0:    
                global_token = self.global_token1.expand(x.shape[0], -1, -1)    # 2 1 28
                # print('fyc-global_token:',global_token.shape)
                # print('fyc-x_flat:',x_flat.shape)
                x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)   
            elif mode == 1 or mode == 2:
                global_token = self.global_token2.expand(x.shape[0], -1, -1)    # 2 1 28
                x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)  

            if mode == 0:
                x_mamba = x_mamba[:, -n_tokens//self.blocks2:, :]  
            else: x_mamba = x_mamba[:, -n_tokens//self.blocks1:, :]  

            # d_model*self.blocks, n_tokens//self.blocks
            if self.channel_numbers > self.local_numbers and mode !=0:
                if self.column_first: 
                    out = x_mamba.transpose(-1, -2).reshape(B, d_model, self.local_numbers, self.channel_numbers, T, W//self.local_numbers, H//self.channel_numbers).permute(0, 1, 4, 6, 3, 5, 2).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).reshape(B, d_model, self.channel_numbers, self.local_numbers, T, H//self.channel_numbers, W//self.local_numbers).permute(0, 1, 4, 5, 2, 6, 3).reshape(B, d_model, *img_dims).contiguous()
            else:
                if self.column_first: 
                    out = x_mamba.transpose(-1, -2).reshape(B, d_model, self.local_numbers, self.local_numbers, T, W//self.local_numbers, H//self.local_numbers).permute(0, 1, 4, 6, 3, 5, 2).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).reshape(B, d_model, self.local_numbers, self.local_numbers, T, H//self.local_numbers, W//self.local_numbers).permute(0, 1, 4, 5, 2, 6, 3).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            if mode == 0: 
                out = self.pixelunshuffle(out)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=tensor_out_last
                    elif mode == 1:
                        tensor_out=tensor_out_last.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=tensor_out_last.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)

        mean = (mode_list[0] + mode_list[1] + mode_list[2])/3
        # mean = (mode_list[0] + mode_list[1])/2
        # mean = mode_list[0]

        return mean

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer_3dlocal(nn.Module):
    def __init__(self, dim, dim_m, d_state = 16, d_conv = 4, expand = 2, channel_token = False, mode_temp=3, mode_dir=0, num_tokens=1, local_numbers=16, channel_numbers=7, column_first=False, upscale_factor=2):
        super().__init__()
        self.dim = dim
        self.mode_dir = mode_dir if isinstance(mode_dir, list) else [mode_dir]
        self.upscale_factor = upscale_factor
        self.upscale_factor_pow = upscale_factor ** 2
        self.norm = nn.LayerNorm(dim//self.upscale_factor_pow)
        self.norm2 = nn.LayerNorm(dim_m)
        self.mamba = Mamba(
                d_model=dim//self.upscale_factor_pow, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                use_fast_path=True,
        )
        self.mamba2 = Mamba(
                d_model=dim_m, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba= True,
                use_fast_path=True,
        )
        self.channel_token = channel_token ## whether to use channel as tokens
        self.mode_temp = mode_temp

        # self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim//self.upscale_factor_pow))
        # self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m))
        self.local_numbers = local_numbers
        self.channel_numbers = channel_numbers
        self.column_first = column_first

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.pixelunshuffle = nn.PixelUnshuffle(self.upscale_factor)
        # self.conv = nn.Conv2d(dim*3, dim, 1, padding=0, bias=True)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.alpha3 = nn.Parameter(torch.ones(1))

    def forward_patch_token(self, x, r = False):

        # B, C, H, W = x.shape
        mode_list = []
        mode_temp=[0,1]
        x_1=x

        for mode in mode_temp:
            
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                    if mode == 0:    
                        tensor=tensor
                    elif mode == 1:
                        tensor=tensor.permute(1, 0, 2)
                    elif mode == 2:
                        tensor=tensor.permute(2, 0, 1)
                    tensor_list.append(tensor)
        
        
            reconstructed_x = torch.stack(tensor_list, dim=0)

            if mode == 0:    
                x = self.pixelshuffle(reconstructed_x).unsqueeze(2) # B, C, H, W -> B, C//4, H*2, W*2
            else: x = reconstructed_x.unsqueeze(2)

            # x = reconstructed_x.unsqueeze(2)
            B, C, T, H, W = x.shape

            B, d_model = x.shape[:2]
            # assert d_model == self.dim
            d_model=C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]

            # self.channel_numbers = 7

            if mode !=0:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//self.channel_numbers, self.channel_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 5, 3, 6, 4).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//self.channel_numbers, self.channel_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
            else:
                if self.column_first:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 5, 3, 6, 4).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(B, d_model, T, H//self.local_numbers, self.local_numbers, W//self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, n_tokens).contiguous().transpose(-1, -2)

            if mode == 0:    
                # global_token = self.global_token1.expand(x.shape[0], -1, -1)    # 2 1 28
                # # print('fyc-global_token:',global_token.shape)
                # # print('fyc-x_flat:',x_flat.shape)
                # x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)   
            elif mode == 1 or mode == 2:
                # global_token = self.global_token2.expand(x.shape[0], -1, -1)    # 2 1 28
                # x_flat = torch.cat((global_token, x_flat), dim=1)    # 2 65537 28

                x_norm=self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)  

            # x_mamba = x_mamba[:, -n_tokens:, :]  

            if mode !=0:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, W//self.local_numbers, H//self.channel_numbers, self.local_numbers, self.channel_numbers).permute(0, 1, 2, 4, 6, 3, 5).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, H//self.channel_numbers, W//self.local_numbers, self.channel_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, *img_dims).contiguous()
            else:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, W//self.local_numbers, H//self.local_numbers, self.local_numbers, self.local_numbers).permute(0, 1, 2, 4, 6, 3, 5).reshape(B, d_model, *img_dims).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(B, d_model, T, H//self.local_numbers, W//self.local_numbers, self.local_numbers, self.local_numbers).permute(0, 1, 2, 3, 5, 4, 6).reshape(B, d_model, *img_dims).contiguous()

            out = out.squeeze(2)

            if mode == 0: 
                out = self.pixelunshuffle(out)

            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                    if mode == 0:    
                        tensor_out=tensor_out_last
                    elif mode == 1:
                        tensor_out=tensor_out_last.permute(1, 0, 2)
                    elif mode == 2:
                        tensor_out=tensor_out_last.permute(1, 2, 0)
                    tensor_list_out.append(tensor_out)
            
            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)

        # mean = (mode_list[0] + mode_list[1] + mode_list[2])/3
        # mean = (mode_list[0] + mode_list[1])/2
        # mean = mode_list[0]
        # mean = self.conv(torch.cat([mode_list[0],mode_list[1],mode_list[2]],dim=1))
        # return mean
        # mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1]+self.alpha3*mode_list[2])/3
        mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1])/2
        return mean
    
    def forward_patch_token_new(self, x, r=False):
        mode_list = []
        mode_temp = [0, 1]
        x_1 = x

        for mode in mode_temp:
            # 根据 mode 对 tensor 进行 permute
            three_d_tensors = torch.unbind(x_1, dim=0)
            tensor_list = []
            for i, tensor in enumerate(three_d_tensors):
                if mode == 0:
                    tensor = tensor
                elif mode == 1:
                    tensor = tensor.permute(1, 0, 2)
                elif mode == 2:
                    tensor = tensor.permute(2, 0, 1)
                tensor_list.append(tensor)

            reconstructed_x = torch.stack(tensor_list, dim=0)

            if mode == 0:
                x = self.pixelshuffle(reconstructed_x).unsqueeze(2)  # B, C, H, W -> B, C//4, H*2, W*2
            else:
                x = reconstructed_x.unsqueeze(2)

            B, C, T, H, W = x.shape
            d_model = C
            n_tokens = x.shape[2:].numel()
            img_dims = x.shape[2:]

            # 分块后转换到 B 维度
            if mode != 0:
                if self.column_first:
                    x_flat = x.view(
                        B, d_model, T, H // self.channel_numbers, self.channel_numbers,
                        W // self.local_numbers, self.local_numbers
                    ).permute(0, 4, 6, 1, 2, 3, 5).reshape(
                        B * self.channel_numbers * self.local_numbers, d_model, -1
                    ).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(
                        B, d_model, T, H // self.channel_numbers, self.channel_numbers,
                        W // self.local_numbers, self.local_numbers
                    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(
                        B * self.channel_numbers * self.local_numbers, d_model, -1
                    ).contiguous().transpose(-1, -2)
            else:
                if self.column_first:
                    x_flat = x.view(
                        B, d_model, T, H // self.local_numbers, self.local_numbers,
                        W // self.local_numbers, self.local_numbers
                    ).permute(0, 4, 6, 1, 2, 3, 5).reshape(
                        B * self.local_numbers * self.local_numbers, d_model, -1
                    ).contiguous().transpose(-1, -2)
                else:
                    x_flat = x.view(
                        B, d_model, T, H // self.local_numbers, self.local_numbers,
                        W // self.local_numbers, self.local_numbers
                    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(
                        B * self.local_numbers * self.local_numbers, d_model, -1
                    ).contiguous().transpose(-1, -2)

            # Mamba 模型处理
            if mode == 0:
                x_norm = self.norm(x_flat)
                x_mamba = self.mamba(x_norm)
            else:
                x_norm = self.norm2(x_flat)
                x_mamba = self.mamba2(x_norm)

            # 恢复维度
            if mode != 0:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(
                        B, self.channel_numbers, self.local_numbers, d_model, T,
                        H // self.channel_numbers, W // self.local_numbers
                    ).permute(0, 3, 4, 5, 1, 6, 2).reshape(B, d_model, T, H, W).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(
                        B, self.channel_numbers, self.local_numbers, d_model, T,
                        H // self.channel_numbers, W // self.local_numbers
                    ).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, d_model, T, H, W).contiguous()
            else:
                if self.column_first:
                    out = x_mamba.transpose(-1, -2).view(
                        B, self.local_numbers, self.local_numbers, d_model, T,
                        H // self.local_numbers, W // self.local_numbers
                    ).permute(0, 3, 4, 5, 1, 6, 2).reshape(B, d_model, T, H, W).contiguous()
                else:
                    out = x_mamba.transpose(-1, -2).view(
                        B, self.local_numbers, self.local_numbers, d_model, T,
                        H // self.local_numbers, W // self.local_numbers
                    ).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, d_model, T, H, W).contiguous()

            out = out.squeeze(2)

            if mode == 0:
                out = self.pixelunshuffle(out)

            # 恢复原始 permute 顺序
            three_d_tensors_out = torch.unbind(out, dim=0)
            tensor_list_out = []
            for i, tensor_out_last in enumerate(three_d_tensors_out):
                if mode == 0:
                    tensor_out = tensor_out_last
                elif mode == 1:
                    tensor_out = tensor_out_last.permute(1, 0, 2)
                elif mode == 2:
                    tensor_out = tensor_out_last.permute(1, 2, 0)
                tensor_list_out.append(tensor_out)

            reconstructed_x_out = torch.stack(tensor_list_out, dim=0)

            mode_list.append(reconstructed_x_out)

        # mean = self.conv(torch.cat([mode_list[0], mode_list[1], mode_list[2]], dim=1))
        # mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1]+self.alpha3*mode_list[2])/3
        mean = (self.alpha1*mode_list[0]+self.alpha2*mode_list[1])/2
        return mean

    def forward_channel_token(self, x, r= False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
 
        return out

    def forward(self, x,r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer_3dlocalchunk(nn.Module):
    def __init__(self, dim, dim_m, d_state=16, d_conv=4, expand=2, channel_token=False, mode_temp=3, num_tokens=1, local_numbers=8, channel_numbers=7, column_first=False, upscale_factor=2, use_fast_path=True):
        super().__init__()
        self.dim = dim
        self.upscale_factor = upscale_factor
        self.upscale_factor_pow = upscale_factor ** 2        
        self.local_numbers = local_numbers
        self.channel_numbers = channel_numbers
        self.cube = True
        self.norm = nn.LayerNorm(dim // self.upscale_factor_pow)
        self.norm2 = nn.LayerNorm(dim_m // self.local_numbers)
        self.mamba = Mamba( #Mamba_rope(
            d_model=dim // self.upscale_factor_pow,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=True,
        )
        self.mamba2 = Mamba( #Mamba_rope(
            d_model=dim_m // self.local_numbers,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=True,
        )
        self.channel_token = channel_token  # Whether to use channel as tokens
        self.mode_temp = mode_temp

        self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim // self.upscale_factor_pow))
        self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m // self.local_numbers))

        self.column_first = column_first

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.pixelunshuffle = nn.PixelUnshuffle(self.upscale_factor)

    def forward_patch_token(self, x, r=False):
        mode_list = []
        mode_temp = [0, 1, 2]
        x_1 = x

        B, C, H, W = x_1.shape
        chunk_size_h = H // self.local_numbers
        chunk_size_w = W // self.local_numbers
        # Ensure chunk_size_h and chunk_size_w are consistent to avoid misalignment
        if H % self.local_numbers != 0 or W % self.local_numbers != 0:
            raise ValueError("Height and Width must be divisible by local_numbers for consistent chunking.")

        for mode in mode_temp:
            
            # print('fyc-mode',mode,', x_1:',x_1.shape)
            # Use unfold to extract all chunks in a batch operation
            x_unfold = x_1.unfold(2, chunk_size_h, chunk_size_h).unfold(3, chunk_size_w, chunk_size_w)
            B_unfold, C_unfold, num_h, num_w, chunk_h, chunk_w = x_unfold.shape
            x_unfold = x_unfold.contiguous().view(B_unfold, C_unfold, num_h * num_w, chunk_h, chunk_w)
            x_unfold = x_unfold.permute(0, 2, 1, 3, 4).contiguous()  # B, num_chunks, C, chunk_h, chunk_w

            # print(f"Mode {mode} - After unfolding and permute: {x_unfold.shape}")

            if mode == 0:
            # if self.cube:
                x_unfold = x_unfold.view(-1, C, chunk_h, chunk_w)
                # print('fyc-mode',mode,', x_unfold:',x_unfold.shape)

                x_unfold = self.pixelshuffle(x_unfold)  # B * num_chunks, C//4, chunk_h*2, chunk_w*2
                _, C, chunk_h, chunk_w = x_unfold.shape
                num_chunks = x_unfold.shape[0] // B
                x_unfold = x_unfold.view(B, num_chunks, C, chunk_h, chunk_w)
                # print(f"Mode {mode} - After pixelshuffle: {x_unfold.shape}")

            if mode == 1:
                x_unfold = x_unfold.permute(0, 1, 3, 2, 4)  # Swap C and H to ensure consistent reshaping
            elif mode == 2:
                x_unfold = x_unfold.permute(0, 1, 4, 2, 3)  # Swap C and W to ensure consistent reshaping

            B_chunks, num_chunks, C, H_chunk, W_chunk = x_unfold.shape
            d_model = C
            n_tokens = H_chunk * W_chunk
            img_dims = (H_chunk, W_chunk)

            # Flatten the chunks for processing
            x_flat = x_unfold.reshape(B_chunks * num_chunks, d_model, -1).transpose(-1, -2)  # (B_chunks * num_chunks), n_tokens, d_model
            # print(f"Mode {mode} - After flattening: {x_flat.shape}")

            # Ensure global_token matches the dimension of x_flat before concatenation
            if mode == 0:
                global_token = self.global_token1.expand(B_chunks * num_chunks, -1, -1)
            else:
                global_token = self.global_token2.expand(B_chunks * num_chunks, -1, -1)

            x_flat = torch.cat((global_token, x_flat), dim=1)
            # print(f"Mode {mode} - After concatenating global token: {x_flat.shape}")

            if mode == 0:
                x_norm = self.norm(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba(x_norm)
            else:
                x_norm = self.norm2(x_flat)
                # torch.cuda.empty_cache()
                x_mamba = self.mamba2(x_norm)

            x_mamba = x_mamba[:, -n_tokens:, :]
            out = x_mamba.transpose(-1, -2).reshape(B_chunks, num_chunks, d_model, *img_dims).contiguous()
            # print(f"Mode {mode} - After Mamba and reshape: {out.shape}")

            if mode == 1:
                out = out.permute(0, 1, 3, 2, 4)  # Revert the permutation for consistency
            elif mode == 2:
                out = out.permute(0, 1, 3, 4, 2)  # Revert the permutation for consistency

            if mode == 0:
            # if self.cube:
                out = out.view(-1, d_model, chunk_h, chunk_w)  # Do not multiply by 2
                out = self.pixelunshuffle(out)
                B_new, C_new, H_new, W_new = out.shape
                # print(f"Mode {mode} - After pixelunshuffle: {out.shape}")
                num_chunks = B_new // B
                out = out.view(B, num_chunks, C_new, H_new, W_new)
                # print(f"Mode {mode} - After reshaping back: {out.shape}")

                # Update chunk_h and chunk_w after pixelunshuffle
                chunk_h = H_new
                chunk_w = W_new

            # Reshape back to original chunk positions
            H_full = num_h * chunk_h
            W_full = num_w * chunk_w
            expected_numel = B * C_new * H_full * W_full
            out = out.permute(0, 2, 1, 3, 4).contiguous().reshape(B, C_new, H_full, W_full)
            assert out.numel() == expected_numel, f"Mismatch in number of elements: expected {expected_numel}, got {out.numel()}"
            # print(f"Mode {mode} - Final output shape: {out.shape}")
            mode_list.append(out)

        # Ensure all tensors in mode_list have the same shape by adjusting operations
        # target_shape = mode_list[0].shape
        # for i in range(1, len(mode_list)):
        #     current_shape = mode_list[i].shape
        #     if current_shape != target_shape:
        #         raise ValueError(f"Inconsistent shapes in mode_list. Mode {i} has shape {current_shape}, expected {target_shape}.")

        mean = (mode_list[0] + mode_list[1] + mode_list[2]) / 3
        return mean

    def forward_channel_token(self, x, r=False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x, r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out


class MambaLayer_3dlocalcube(nn.Module):
    def __init__(self, dim, dim_m, d_state=16, d_conv=4, expand=2, channel_token=False, mode_temp=3, num_tokens=1, local_numbers=8, channel_numbers=7, column_first=False, upscale_factor=2, use_fast_path=True):
        super().__init__()
        self.dim = dim
        self.upscale_factor = upscale_factor
        self.upscale_factor_pow = upscale_factor ** 2        
        self.local_numbers = local_numbers
        self.channel_numbers = channel_numbers
        self.cube = True
        self.norm = nn.LayerNorm(dim // self.upscale_factor_pow)
        self.norm2 = nn.LayerNorm(dim_m // self.local_numbers) #* self.upscale_factor)
        self.mamba = Mamba( #Mamba_rope(
            d_model=dim // self.upscale_factor_pow,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=use_fast_path,
            # rope_scale_factor=4.0,
        )
        self.mamba2 = Mamba( #Mamba_rope( # Mamba( #
            d_model=dim_m // self.local_numbers, #* self.upscale_factor,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            # use_fast_path=use_fast_path,
            # rope_scale_factor=4.0,
        )
        self.channel_token = channel_token  # Whether to use channel as tokens
        self.mode_temp = mode_temp

        self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim // self.upscale_factor_pow))
        # self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m // self.local_numbers * self.upscale_factor))
        
        self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, dim_m // self.local_numbers))

        self.column_first = column_first

        self.pixelshuffle = nn.PixelShuffle(self.upscale_factor)
        self.pixelunshuffle = nn.PixelUnshuffle(self.upscale_factor)

    def forward_patch_token(self, x, r=False):
        mode_list = []
        mode_temp = [0, 1, 2]
        # mode_temp = [0]
        x_1 = x

        B, C_init, H, W = x_1.shape
        chunk_size_h = H // self.local_numbers
        chunk_size_w = W // self.local_numbers
        # Ensure chunk_size_h and chunk_size_w are consistent to avoid misalignment
        if H % self.local_numbers != 0 or W % self.local_numbers != 0:
            raise ValueError("Height and Width must be divisible by local_numbers for consistent chunking.")

        for mode in mode_temp:
            # Use unfold to extract all chunks in a batch operation
            x_unfold = x_1.unfold(2, chunk_size_h, chunk_size_h).unfold(3, chunk_size_w, chunk_size_w)
            B_unfold, C_unfold, num_h, num_w, chunk_h, chunk_w = x_unfold.shape
            x_unfold = x_unfold.contiguous().view(B_unfold, C_unfold, num_h * num_w, chunk_h, chunk_w)
            x_unfold = x_unfold.permute(0, 2, 1, 3, 4).contiguous()  # B, num_chunks, C, chunk_h, chunk_w

            x_unfold = x_unfold.view(-1, C_unfold, chunk_h, chunk_w)
            
            if self.cube and self.upscale_factor>1 and mode==0:
                x_unfold = self.pixelshuffle(x_unfold)  # B * num_chunks, C//4, chunk_h*2, chunk_w*2

            C_ps, H_ps, W_ps = x_unfold.shape[1], x_unfold.shape[2], x_unfold.shape[3]
            num_chunks = x_unfold.shape[0] // B
            x_unfold = x_unfold.view(B, num_chunks, C_ps, H_ps, W_ps)

            if mode == 1:
                x_unfold = x_unfold.permute(0, 1, 3, 2, 4)  # Swap C and H
            elif mode == 2:
                x_unfold = x_unfold.permute(0, 1, 4, 2, 3)  # Swap C and W

            B_chunks, num_chunks, C_current, H_chunk, W_chunk = x_unfold.shape
            d_model = C_current
            n_tokens = H_chunk * W_chunk
            img_dims = (H_chunk, W_chunk)

            # Flatten the chunks for processing
            x_flat = x_unfold.reshape(B_chunks * num_chunks, d_model, -1).transpose(-1, -2)  # (B_chunks * num_chunks), n_tokens, d_model

            # Ensure global_token matches the dimension of x_flat before concatenation
            if mode == 0:
                global_token = self.global_token1.expand(B_chunks * num_chunks, -1, -1)
            else:
                global_token = self.global_token2.expand(B_chunks * num_chunks, -1, -1)

            x_flat = torch.cat((global_token, x_flat), dim=1)

            if mode == 0:
                x_norm = self.norm(x_flat)
                x_mamba = self.mamba(x_norm)
            else:
                x_norm = self.norm2(x_flat)
                x_mamba = self.mamba2(x_norm)

            x_mamba = x_mamba[:, -n_tokens:, :]
            out = x_mamba.transpose(-1, -2).reshape(B_chunks, num_chunks, d_model, *img_dims).contiguous()

            if mode == 1:
                out = out.permute(0, 1, 3, 2, 4)  # Revert the permutation
            elif mode == 2:
                out = out.permute(0, 1, 3, 4, 2)  # Revert the permutation

            B_chunks, num_chunks, C_current, H_chunk, W_chunk = out.shape
            d_model = C_current

            out = out.reshape(-1, d_model, H_chunk, W_chunk)
            
            if self.cube and self.upscale_factor>1 and mode==0:
                out = self.pixelunshuffle(out)

            C_out, H_out, W_out = out.shape[1], out.shape[2], out.shape[3]
            num_chunks = out.shape[0] // B
            out = out.view(B, num_chunks, C_out, H_out, W_out)
            chunk_h = H_out
            chunk_w = W_out

            # Reshape back to original chunk positions
            num_h = H // chunk_size_h
            num_w = W // chunk_size_w
            H_full = num_h * chunk_h
            W_full = num_w * chunk_w
            expected_numel = B * C_out * H_full * W_full
            out = out.permute(0, 2, 1, 3, 4).contiguous().reshape(B, C_out, H_full, W_full)
            assert out.numel() == expected_numel, f"Mismatch in number of elements: expected {expected_numel}, got {out.numel()}"
            mode_list.append(out)

        mean = sum(mode_list) / len(mode_list)
        return mean

    def forward_channel_token(self, x, r=False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x, r=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.type(torch.float32)

        if self.channel_token:
            out = self.forward_channel_token(x, r=r)
        else:
            out = self.forward_patch_token(x, r=r)

        return out

class MambaLayer_cube(nn.Module):
    def __init__(self, dim, dim_m, block_size=8, d_state=16, d_conv=4, expand=2, channel_token=False, mode_temp=3, num_tokens=1, use_fast_path=True):
        super().__init__()
        print(f"Transformer: dim: {dim}")
        print(f"Mamba: dim: {dim_m}")
        self.dim = dim
        self.block_size = block_size
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.block_size)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )
        self.mamba2 = Mamba(
            d_model=self.block_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=use_fast_path,
        )
        self.channel_token = channel_token
        self.mode_temp = mode_temp

        self.global_token1 = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.global_token2 = nn.Parameter(torch.zeros(1, num_tokens, self.block_size))

    def forward_patch_token(self, x, r=False):
        B, C, H, W = x.shape
        mode_list = []

        for mode in [0, 1, 2]:
            if mode == 1:
                x_permuted = x.permute(0, 2, 1, 3).contiguous()  # Swap H and C
            elif mode == 2:
                x_permuted = x.permute(0, 3, 2, 1).contiguous()  # Swap W and C
            else:
                x_permuted = x  # No permutation

            B, C_shape, H_shape, W_shape = x_permuted.shape

            x_reshaped = x_permuted.view(B, C_shape, -1).transpose(1, 2)  # Flatten to B x HW x C
            n_tokens = x_reshaped.shape[1]

            if mode == 0:
                global_token = self.global_token1.expand(B, -1, -1)
                x_input = torch.cat((global_token, x_reshaped), dim=1)
                x_norm = self.norm(x_input)
                x_mamba = self.mamba(x_norm)
            else:
                global_token = self.global_token2.expand(B, -1, -1)
                x_input = torch.cat((global_token, x_reshaped), dim=1)
                x_norm = self.norm2(x_input)
                x_mamba = self.mamba2(x_norm)

            x_output = x_mamba[:, -n_tokens:, :].transpose(1, 2).contiguous()
            out = x_output.view(B, C_shape, H_shape, W_shape)  # Restore original shape

            if mode == 1:
                out = out.permute(0, 2, 1, 3)  # Restore swapped H and C
            elif mode == 2:
                out = out.permute(0, 3, 2, 1)  # Restore swapped W and C

            mode_list.append(out)

        mean_output = sum(mode_list) / len(mode_list)  # Average across modes
        return mean_output

    def forward_channel_token(self, x, r=False):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()

        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)

        return out

    def forward(self, x, r=False):
        # 分块处理输入张量
        B, C, H, W = x.shape
        block_size = self.block_size
        x_blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        B, C, num_blocks_H, num_blocks_W, _, _ = x_blocks.shape

        # 执行 forward_patch_token 操作
        x_blocks = x_blocks.contiguous().view(B * num_blocks_H * num_blocks_W, C, block_size, block_size)
        # print('fyc-x_block:',x_blocks.shape)
        out_blocks = self.forward_patch_token(x_blocks, r=r)

        # 还原张量到 B, C, H, W 形状
        out_blocks = out_blocks.view(B, num_blocks_H, num_blocks_W, C, block_size, block_size)
        out = out_blocks.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)

        return out


class LocalNonLocalBlock(nn.Module):
    """
    The Local and Non-Local Transformer Block (LNLB) is the most important component. Each LNLB consists of three layer-normalizations (LNs), a Local MSA, a Non-Local MSA, and a GDFN (Zamir et al. 2022).
    """
    def __init__(self, 
                 cfg,
                 dim, 
                 dim_m,
                 num_heads,
                 window_size:tuple,
                 window_num:tuple,
                 layernorm_type,
                 num_blocks,
                 ):
        super().__init__()
        self.cfg = cfg
        self.window_size = window_size
        self.window_num = window_num
        self.dim = dim
#    def __init__(self, dim, num_heads=2, num_tokens=1, window_size=8, qkv_bias=False, drop=0., attn_drop=0.)
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                # PreNorm(dim, nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                #     layernorm_type = layernorm_type),

                # PreNorm(dim, MambaLayer_3dlocalcube( #_3dlocalchunk( # MambaLayer_3dlocalchunk( #MambaLayer_hilbert(
                #         dim = dim, 
                #         dim_m = dim_m,
                #         d_state = 16,
                #         local_numbers=16, 
                #         channel_numbers=1,
                #         upscale_factor=1,
                #         d_conv=4,
                #         expand = 2, 
                #         channel_token = False,
                #         mode_temp=3,
                #         use_fast_path=False),
                #     layernorm_type = layernorm_type),

                # SCMAMBA_5stg_simu_fyc_1107_3dlocal_one3_248_1_up1
                PreNorm(dim, MambaLayer_3dlocal( #MambaLayer_hilbert(
                        dim = dim, 
                        dim_m = dim_m,
                        d_state = 16,
                        local_numbers=dim_m//8, 
                        channel_numbers=2,
                        upscale_factor=1,
                        d_conv=4,
                        expand = 2, 
                        channel_token = False,
                        mode_temp=2), # , column_first=True
                    layernorm_type = layernorm_type),

                PreNorm(dim, MambaLayer_global( #MambaLayer_hilbert(
                        dim = dim, 
                        dim_m = dim_m,
                        d_state = 16,
                        d_conv=4,
                        expand = 2, 
                        channel_token = False,
                        mode_temp=2,
                        use_fast_path=True),
                    layernorm_type = layernorm_type),

                PreNorm(dim, FFN_FN(
                    cfg,
                    ffn_name = cfg.MODEL.DENOISER.SCMAMBA.FFN_NAME,
                    dim = dim
                ),
                layernorm_type = layernorm_type)
                                
            ]))


    def forward(self, x, hilbert_curve1=None):
        # for (Attention, MambaLayer, ffn) in self.blocks:

        # x = x + SVDShrink_Fast_GPU(x)

        for (DynamicLocal, MambaLayer, ffn) in self.blocks: # DynamicLocal, MambaLayer,   DynamicLocal1, DynamicLocal2, DynamicLocal3, 
            # x = x + Attention(x) 
            B, C, H, W = x.shape
            # x = x + conv(x)
            x = x + DynamicLocal(x)
            x = x + MambaLayer(x) # , hilbert_curve1
            x = x + ffn(x)

        return x
    

class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    
class LNLT(nn.Module):
    """
    The Local and Non-Local Transformer (LNLT) adopts a three-level U-shaped structure, and each level consists of multiple basic units called Local and Non-Local Transformer Blocks (LNLBs). Up- and down-sampling modules are positioned between LNLBs.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Conv2d(cfg.MODEL.DENOISER.SCMAMBA.IN_DIM, cfg.MODEL.DENOISER.SCMAMBA.DIM, kernel_size=3, stride=1, padding=1, bias=False)

        self.Encoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 0, 
                dim_m = 256,
                num_heads = 2 ** 0, 
                window_size = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.SCMAMBA.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.SCMAMBA.NUM_BLOCKS[0],
            ),
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1, 
                dim_m = 128,
                num_heads = 2 ** 1, 
                window_size = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.SCMAMBA.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.SCMAMBA.NUM_BLOCKS[1],
            ),
        ])

        self.BottleNeck = LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 2, 
                dim_m = 64,
                num_heads = 2 ** 2, 
                window_size = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.SCMAMBA.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.SCMAMBA.NUM_BLOCKS[2],
            )

        self.Decoder = nn.ModuleList([
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1, 
                dim_m = 128,
                num_heads = 2 ** 1, 
                window_size = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.SCMAMBA.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.SCMAMBA.NUM_BLOCKS[3],
            ),
            LocalNonLocalBlock(
                cfg = cfg, 
                dim = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 0, 
                dim_m = 256,
                num_heads = 2 ** 0, 
                window_size = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_SIZE,
                window_num = cfg.MODEL.DENOISER.SCMAMBA.WINDOW_NUM,
                layernorm_type = cfg.MODEL.DENOISER.SCMAMBA.LAYERNORM_TYPE,
                num_blocks = cfg.MODEL.DENOISER.SCMAMBA.NUM_BLOCKS[4],
            )
        ])

        self.Downs = nn.ModuleList([
            DownSample(cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 0),
            DownSample(cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1)
        ])

        self.Ups = nn.ModuleList([
            UpSample(cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 2),
            UpSample(cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 2,
                out_channels = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            ),
            nn.Conv2d(
                in_channels = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 1,
                out_channels = cfg.MODEL.DENOISER.SCMAMBA.DIM * 2 ** 0,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False
            )
        ])

        self.mapping = nn.Conv2d(cfg.MODEL.DENOISER.SCMAMBA.DIM, cfg.MODEL.DENOISER.SCMAMBA.OUT_DIM, kernel_size=3, stride=1, padding=1, bias=False)

    # def hilbert_curve_256_scale(self, ):

    #     # B, nf, C, H, W = x.shape

    #     nf = 7 #28
    #     H = 64
    #     W = 64

    #     hilbert_curve = list(
    #         Hilbert3d(width=H, height=W, depth=nf))  
    #     hilbert_curve = torch.tensor(hilbert_curve).long()
    #     hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    #     return {
    #         'hilbert_curve_256_scale': hilbert_curve
    #     }
    
    # def hilbert_curve_128_scale(self, ):

    #     # B, nf, C, H, W = x.shape

    #     nf = 14
    #     H = 32
    #     W = 32

    #     hilbert_curve = list(
    #         Hilbert3d(width=H, height=W, depth=nf))  
    #     hilbert_curve = torch.tensor(hilbert_curve).long()
    #     hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    #     return {
    #         'hilbert_curve_128_scale': hilbert_curve
    #     }
    
    # def hilbert_curve_64_scale(self, ):

    #     # B, nf, C, H, W = x.shape

    #     nf = 28
    #     H = 16
    #     W = 16

    #     hilbert_curve = list(
    #         Hilbert3d(width=H, height=W, depth=nf))  
    #     hilbert_curve = torch.tensor(hilbert_curve).long()
    #     hilbert_curve = hilbert_curve[:, 0] * W * nf + hilbert_curve[:, 1] * nf + hilbert_curve[:, 2]

    #     return {
    #         'hilbert_curve_64_scale': hilbert_curve
    #     }
    
    # def save_hilbert_curve_256_scale(self, hilbert_curve_256_scale, filename='./Hilbert1009/hilbert_curve_256_scale.pt'):
    #     torch.save(hilbert_curve_256_scale, filename)
        
    # def save_hilbert_curve_128_scale(self, hilbert_curve_128_scale, filename='./Hilbert1009/hilbert_curve_128_scale.pt'):
    #     torch.save(hilbert_curve_128_scale, filename)
       
    # def save_hilbert_curve_64_scale(self, hilbert_curve_64_scale, filename='./Hilbert1009/hilbert_curve_64_scale.pt'):
    #     torch.save(hilbert_curve_64_scale, filename)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        
        # result64 = self.hilbert_curve_256_scale()
        # hilbert_curve64 = result64['hilbert_curve_256_scale'].to(x.device)
        # self.save_hilbert_curve_256_scale(hilbert_curve64)
        
        # result64 = self.hilbert_curve_128_scale()
        # hilbert_curve64 = result64['hilbert_curve_128_scale'].to(x.device)
        # self.save_hilbert_curve_128_scale(hilbert_curve64)

        # result64 = self.hilbert_curve_64_scale()
        # hilbert_curve64 = result64['hilbert_curve_64_scale'].to(x.device)
        # self.save_hilbert_curve_64_scale(hilbert_curve64)

        # hilbert_curve_256_1 = torch.load('./Hilbert1009/hilbert_curve_256_scale.pt', map_location=x.device)
        # hilbert_curve_128_1 = torch.load('./Hilbert1009/hilbert_curve_128_scale.pt', map_location=x.device)
        # hilbert_curve_64_1 = torch.load('./Hilbert1009/hilbert_curve_64_scale.pt', map_location=x.device)

        
        # result64 = self.hilbert_curve_256_2_scale()
        # hilbert_curve64 = result64['hilbert_curve_256_2_scale'].to(x.device)
        # self.save_hilbert_curve_256_2_scale(hilbert_curve64)
        
        # result64 = self.hilbert_curve_128_2_scale()
        # hilbert_curve64 = result64['hilbert_curve_128_2_scale'].to(x.device)
        # self.save_hilbert_curve_128_2_scale(hilbert_curve64)

        # result64 = self.hilbert_curve_64_2_scale()
        # hilbert_curve64 = result64['hilbert_curve_64_2_scale'].to(x.device)
        # self.save_hilbert_curve_64_2_scale(hilbert_curve64)

        # hilbert_curve_256_2 = torch.load('./Hilbert4/hilbert_curve_256_2_scale.pt', map_location=x.device)
        # hilbert_curve_128_2 = torch.load('./Hilbert4/hilbert_curve_128_2_scale.pt', map_location=x.device)
        # hilbert_curve_64_2 = torch.load('./Hilbert4/hilbert_curve_64_2_scale.pt', map_location=x.device)

        x1 = self.embedding(x)
        res1 = self.Encoder[0](x1) # , hilbert_curve_256_1)

        x2 = self.Downs[0](x1)
        res2 = self.Encoder[1](x2) #, hilbert_curve_128_1)

        x4 = self.Downs[1](res2)
        res4 = self.BottleNeck(x4) #, hilbert_curve_64_1)

        dec_res2 = self.Ups[0](res4) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1) # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2) # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = self.Decoder[0](dec_res2) #, hilbert_curve_128_1)

        dec_res1 = self.Ups[1](dec_res2) # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1) # dim * 2 ** 1 
        dec_res1 = self.fusions[1](dec_res1) # dim * 2 ** 1 -> dim * 2 ** 0        
        dec_res1 = self.Decoder[1](dec_res1) #, hilbert_curve_256_1)

        if self.cfg.MODEL.DENOISER.SCMAMBA.WITH_NOISE_LEVEL:
            out = self.mapping(dec_res1) + x[:, 1:, :, :]
        else:
            out = self.mapping(dec_res1) + x
            
        return out[:, :, :h_inp, :w_inp]
    

def PWDWPWConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=True),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=False)
    )

def A(x, Phi):
    B, nC, H, W = x.shape
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [B, C, H, W] = inputs.shape
    temp = torch.zeros((B, C, H, W+(C-1)*step)).to(inputs.device)
    temp[:, :, :, :W] = inputs
    for i in range(C):
        temp[:,i,:,:] = torch.roll(temp[:,i,:,:], shifts=step*i, dims=2)
    return temp

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


class DegradationEstimation(nn.Module):
    """
    The Degradation Estimation Network (DEN) is proposed to estimate degradation-related parameters from the input of the current recurrent step and with reference to the sensing matrix.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.DL = nn.Sequential(
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2),
            PWDWPWConv(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS),
        )
        self.down_sample = nn.Conv2d(self.cfg.DATASETS.WAVE_LENS, self.cfg.DATASETS.WAVE_LENS*2, 3, 2, 1, bias=True) # (B, 64, H, W) -> (B, 64, H//2, W//2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, self.cfg.DATASETS.WAVE_LENS*2, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.cfg.DATASETS.WAVE_LENS*2, 2, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)


    def forward(self, y, phi):

        inp = torch.cat([phi, y], dim=1)
        phi_r = self.DL(inp)

        phi = phi + phi_r

        x = self.down_sample(self.relu(phi_r))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6
        mu = x[:, 0, :, :]
        noise_level = x[:, 1, :, :]

        return phi, mu, noise_level[:, None, :, :]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

class l_nl_attn(nn.Module):
    def __init__(self, dim, num_heads=2, num_tokens=1, window_size=8, norm_layer=nn.LayerNorm, qkv_bias=False, drop=0., mlp_ratio=4.,attn_drop=0.,drop_path=0.,):
        super(l_nl_attn, self).__init__()

        self.attn = Attention(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.global_token = nn.Parameter(torch.zeros(1, num_tokens, dim))
        self.dim= dim
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

   
    def forward(self, x):
        
        B, C, H, W = x.shape
        
        x = x.flatten(2).transpose(1, 2)
        global_token = self.global_token.expand(x.shape[0], -1, -1)
        
        

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, H*W, C)

        x_att = torch.cat((global_token, x), dim=1)
        x = self.attn(x_att,H, W)


        x = x[:, -H*W:]
        out = x.view(-1, H, W, self.dim).permute(0, 3, 1, 2).contiguous()      

        x = self.out_proj(x * act_res)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_tokens=1, num_heads=2, window_size=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward_global_aggregation(self, q, k, v):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, q, k, v, H, W):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape# 4,65537,28
        NC = self.num_tokens#1
        # pad
        x_img, x_global = x[:, NC:], x[:, :NC]
        x_img = x_img.view(B, H, W, C)
        pad_l = pad_t = 0
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_img = F.pad(x_img, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x_img.shape[1], x_img.shape[2]
        x_img = x_img.view(B, -1, C)
        x = torch.cat([x_global, x_img], dim=1)

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NC:], k[:, :, NC:], v[:, :, NC:]
        q_cls, _, _ = q[:, :, :NC], k[:, :, :NC], v[:, :, :NC]

        # local window attention
        x_img = self.forward_local(q_img, k_img, v_img, Hp, Wp)
        # restore to the original size
        x_img = x_img.view(B, Hp, Wp, -1)[:, :H, :W].reshape(B, H*W, -1)
        q_img = q_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        k_img = k_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        v_img = v_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)

        # global aggregation
        x_cls = self.forward_global_aggregation(q_cls, k_img, v_img)
        k_cls, v_cls = self.kv_global(x_cls).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # gloal broadcast
        x_img = x_img + self.forward_global_broadcast(q_img, k_cls, v_cls)

        x = torch.cat([x_cls, x_img], dim=1)
        x = self.proj(x)
        return x


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)

def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output

class SCMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.fusion = nn.Conv2d(cfg.DATASETS.WAVE_LENS*2, cfg.DATASETS.WAVE_LENS, 1, padding=0, bias=True)

        self.DP = nn.ModuleList([
           DegradationEstimation(cfg) for _ in range(cfg.MODEL.DENOISER.SCMAMBA.STAGE)
        ]) if not cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else DegradationEstimation(cfg)
        self.PP = nn.ModuleList([
            LNLT(cfg) for _ in range(cfg.MODEL.DENOISER.SCMAMBA.STAGE)
        ]) if not cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else LNLT(cfg)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initial(self, y, Phi):
        nC = self.cfg.DATASETS.WAVE_LENS
        step = self.cfg.DATASETS.STEP
        bs, nC, row, col = Phi.shape
        y_shift = torch.zeros(bs, nC, row, col).to(y.device).float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fusion(torch.cat([y_shift, Phi], dim=1))
        return z

    def prepare_input(self, data):
        hsi = data['hsi']
        mask = data['mask']
        YH = gen_meas_torch_batch(hsi, mask, step=self.cfg.DATASETS.STEP, wave_len=self.cfg.DATASETS.WAVE_LENS, mask_type=self.cfg.DATASETS.MASK_TYPE, with_noise=self.cfg.DATASETS.TRAIN.WITH_NOISE)
        data['Y'] = YH['Y']
        data['H'] = YH['H']
        return data
    

    def forward_train(self, data):
        y = data['Y']
        phi = data['mask']
        x0 = data['H']
        z = self.initial(y, phi)
        B, C, H, W = phi.shape
        B, C, H_, W_ = x0.shape      
        z_hat = z
        z_list=[]
        z_list.append(z)
        beta=0.5* torch.ones((W, 1)).to(y.device)

        for i in range(self.cfg.MODEL.DENOISER.SCMAMBA.STAGE):
            Phi, mu, noise_level = self.DP[i](z, phi) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.DP(z, phi)

            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_DL:
                Phi = phi
            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_MU:
                mu = torch.FloatTensor([1e-6]).to(y.device)

            Phi_s = torch.sum(Phi**2,1)
            Phi_s[Phi_s==0] = 1
            Phi_z = A(z_hat, Phi)
            x = z + At(torch.div(y-Phi_z,mu+Phi_s), Phi)
            x = shift_back_3d(x)[:, :, :, :W_]
            noise_level_repeat = noise_level.repeat(1,1,x.shape[2], x.shape[3])
            # z = self.PP(x)
            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_NOISE_LEVEL:
                z = self.PP[i](x) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.PP(x)
            else:
                z = self.PP[i](torch.cat([noise_level_repeat, x],dim=1)) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.PP(torch.cat([noise_level_repeat, x],dim=1))
            z = shift_3d(z)
            z_list.append(z)
            z_hat = z + beta[i]*(z_list[-1]-z_list[-2])

        z = shift_back_3d(z)[:, :, :, :W_]

        out = z[:, :, :, :W_]

        return out
    
    def forward_test(self, data):
        y = data['Y']
        phi = data['mask']
        x0 = data['H']
        z = self.initial(y, phi)
        B, C, H, W = phi.shape
        B, C, H_, W_ = x0.shape      

        z_hat = z
        z_list=[]
        z_list.append(z)
        beta=0.5* torch.ones((W, 1)).to(y.device)


        for i in range(self.cfg.MODEL.DENOISER.SCMAMBA.STAGE):
            Phi, mu, noise_level = self.DP[i](z, phi) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.DP(z, phi)

            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_DL:
                Phi = phi
            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_MU:
                mu = torch.FloatTensor([1e-6]).to(y.device)

            Phi_s = torch.sum(Phi**2,1)
            Phi_s[Phi_s==0] = 1
            Phi_z = A(z_hat, Phi)
            x = z + At(torch.div(y-Phi_z,mu+Phi_s), Phi)
            x = shift_back_3d(x)[:, :, :, :W_]
            noise_level_repeat = noise_level.repeat(1,1,x.shape[2], x.shape[3])

            if not self.cfg.MODEL.DENOISER.SCMAMBA.WITH_NOISE_LEVEL:
                z = self.PP[i](x) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.PP(x)
            else:
                z = self.PP[i](torch.cat([noise_level_repeat, x],dim=1)) if not self.cfg.MODEL.DENOISER.SCMAMBA.SHARE_PARAMS else self.PP(torch.cat([noise_level_repeat, x],dim=1))
            z = shift_3d(z)
            z_list.append(z)
            z_hat = z + beta[i]*(z_list[-1]-z_list[-2])


        z = shift_back_3d(z)[:, :, :, :W_]

        out = z[:, :, :, :W_]

        return out
    
    def forward(self, data):
        if self.training:
            data = self.prepare_input(data)
            x = self.forward_train(data)
        else:
             x = self.forward_test(data)
        return x
    

