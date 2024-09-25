#################################################################
### UNet code from https://github.com/openai/guided-diffusion ###
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
from functools import partial
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import wandb
from config import models_dir
import os
from torchdiffeq import odeint
import numpy as np
from torchmetrics.functional.segmentation import mean_iou, generalized_dice_score
from accelerate import Accelerator
from abc import abstractmethod

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_head_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_head_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_head_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
    
def gaussian_to_class(mean, map):
    class_dist = []
    for m in mean:
        if len(map.shape) > 2:
            dist = abs(map - m[:, None, None].to(map.device))
        else:
            dist = abs(map - m[:, None].to(map.device))
        class_dist.append(torch.mean(dist, axis=1))
    class_dist = torch.stack(class_dist)
    class_map = torch.argmin(class_dist, axis=0).unsqueeze(1)

    return class_map.float()


def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'SemFM')):
        os.makedirs(os.path.join(models_dir, 'SemFM'))

def initial_means(n_classes, dist=4.0):
    N = n_classes  # for example, 1000 points

    # Find the cube root of N to determine how many points per axis
    side_points = round(N ** (1/3))

    # Adjust N to be a perfect cube of side_points
    N_actual = side_points ** 3

    # Generate a grid with evenly spaced points along each axis
    linspace = torch.linspace(-dist*(side_points-1)/2., dist*(side_points-1)/2., side_points)
    x, y, z = torch.meshgrid(linspace, linspace, linspace)

    # Stack the grid coordinates and reshape them into N_actual x 3 shape
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    # Trim the grid to the original N points, in case N is not a perfect cube
    points = points[:N]

    return points.float()



class SemFM(nn.Module):

    def __init__(self, args, channels=3):
        '''
        SemFM model
        :param args: arguments
        '''
        super(SemFM, self).__init__()
        self.args = args
        self.channels = channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mu = torch.nn.Parameter(initial_means(args.n_classes, args.dist).to(self.device), requires_grad=not args.anchor)
        #self.var = torch.nn.Parameter(torch.rand(args.n_classes, channels).clamp(0.5,1.0).to(self.device), requires_grad=True)
        self.var = [torch.nn.Parameter(torch.rand(channels).clamp(0.5,1.0).to(self.device), requires_grad=True) for _ in range(args.n_classes)]
        self.prior = [torch.distributions.Normal(self.mu[i], self.var[i]) for i in range(args.n_classes)]
        self.unet = UNetModel(
            image_size=args.size,
            in_channels=channels,
            model_channels=args.model_channels,
            out_channels=channels,
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=args.attention_resolutions,
            dropout=args.dropout,
            channel_mult=args.channel_mult,
            conv_resample=args.conv_resample,
            dims=args.dims,
            num_classes=None,
            use_checkpoint=False,
            num_heads=args.num_heads,
            num_head_channels=args.num_head_channels,
            num_heads_upsample=-1,
            use_scale_shift_norm=args.use_scale_shift_norm,
            resblock_updown=args.resblock_updown,
            use_new_attention_order=args.use_new_attention_order
        )
        self.unet.to(self.device)
        self.n_classes = args.n_classes
        self.dataset = args.dataset
        self.warmup = args.warmup
        self.decay = args.decay
        self.ode = args.ode
        self.solver = args.solver
        self.w_seg = args.w_seg
        self.tolerance = args.tolerance
        self.dequantize = args.dequantize
        # self.colors should be a list of n_classes of rgb values
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(args.n_classes, 3))
        np.random.seed(None)

    def forward(self, x, time):
        '''
        Forward pass of the SemFM model
        :param x: input image
        :param time: time embedding
        '''
        return self.unet(x, time)

    def bce_loss(self, predicted_mask, mask):
        '''
        Binary cross entropy loss
        :param predicted_mask: predicted mask
        :param mask: ground truth mask
        '''
        bce = nn.BCELoss()
        return bce(predicted_mask, mask)
    
    def kl_loss(self):
        '''
        KL divergence loss
        :param predicted_mask: predicted mask
        :param mask: ground truth mask
        '''
        cnt = 0
        # kl divergence
        for i in range(self.n_classes):
            min_kl = np.inf
            for j in range(self.n_classes):
                if i!=j:
                    kl = torch.distributions.kl_divergence(self.prior[i], self.prior[j]).mean()
                    if kl < min_kl:
                        min_kl = kl
            if 1/min_kl < self.tolerance:
                # disable grad for mean and var of the class with high kl divergence
                self.mu[i].requires_grad = False
                self.var[i].requires_grad = False

        return min_kl
    
    def dice_loss(self, predicted_mask, mask):
        '''
        Dice loss
        :param predicted_mask: predicted mask
        :param mask: ground truth mask
        '''
        return 1 - generalized_dice_score(predicted_mask, mask, self.n_classes).mean()

    def conditional_flow_matching_loss(self, x, mask):
        
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)

        for i in range(self.n_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                noise = self.mask_to_gaussian(i, inst_mask, x.shape)
            else:
                noise += self.mask_to_gaussian(i, inst_mask, x.shape)

        if self.dequantize:
            x = x + torch.rand_like(x) / 128.0

        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * noise + t[:, None, None, None] * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self.unet(x_t, t)
        predicted_noise = x_t - predicted_flow*t[:, None, None, None]

        labels = F.one_hot(mask.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        labels = labels.float()
        preds = gaussian_to_class(self.mu, predicted_noise).squeeze(1)
        preds = F.one_hot(preds.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        preds = preds.float()
        
        bce_loss = self.bce_loss(preds, labels)
        kl_loss = self.kl_loss()
        dice_loss = self.dice_loss(preds, labels)
        
        return (predicted_flow - optimal_flow).square().mean(), bce_loss, dice_loss, kl_loss
    
    def train_model(self, dataloader, testloader=None):
        '''
        Train the SemFM model
        :param init_dataloader: initial dataloader for training model and distributions
        :param final_dataloader: final dataloader for training model only
        :param testloader: test loader
        '''
        accelerate = Accelerator(log_with="wandb")
        accelerate.init_trackers(project_name='SemFM',
            config = {
                    'batch_size': self.args.batch_size,
                    'n_epochs': self.args.n_epochs,
                    'lr': self.args.lr,
                    'sample_and_save_freq': self.args.sample_and_save_freq,
                    'n_samples': self.args.n_samples,
                    'n_steps': self.args.n_steps,
                    'n_classes': self.args.n_classes,
                    'dataset': self.args.dataset,
                    'size': self.args.size,
                    'dist': self.args.dist,
                    'warmup': self.args.warmup,
                    'decay': self.args.decay,
                    'w_seg': self.args.w_seg,
                    'anchor': self.args.anchor,
                    'model_channels': self.args.model_channels,
                    'num_res_blocks': self.args.num_res_blocks,
                    'attention_resolutions': self.args.attention_resolutions,
                    'dropout': self.args.dropout,
                    'channel_mult': self.args.channel_mult,
                    'conv_resample': self.args.conv_resample,
                    'dims': self.args.dims,
                    'num_heads': self.args.num_heads,
                    'num_head_channels': self.args.num_head_channels,
                    'use_scale_shift_norm': self.args.use_scale_shift_norm,
                    'resblock_updown': self.args.resblock_updown,
                    'use_new_attention_order': self.args.use_new_attention_order,
                    'dequantize': self.args.dequantize,       
                },
                init_kwargs={"wandb":{"name": f"SemFM_{self.args.dataset}"}})
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.decay)
        epoch_bar = trange(self.args.n_epochs, desc='Epochs', leave=True)
        create_checkpoint_dir()

        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_loss = float('inf')

        if not self.args.anchor:
            dataloader, self.unet, self.mu, self.var, optimizer, scheduler = accelerate.prepare(dataloader, self.unet, self.mu, self.var, optimizer, scheduler)
        else:
            dataloader, self.unet, self.var, optimizer, scheduler = accelerate.prepare(dataloader, self.unet, self.var, optimizer, scheduler)

        for epoch in epoch_bar:

            epoch_loss_rec = 0.
            epoch_loss_ce = 0.
            epoch_loss_kl = 0.
            epoch_loss = 0.
            self.train()

            for x, mask in tqdm(dataloader, desc='Batches', leave=False):
                #x = x.to(self.device)
                #mask = mask.to(self.device)
                optimizer.zero_grad()

                recon_loss, bce_loss, dice_loss, kl_loss = self.conditional_flow_matching_loss(x, mask)
                loss = recon_loss + self.w_seg*0.5*(bce_loss + dice_loss)

                accelerate.backward(loss, retain_graph=True)

                optimizer.step()
                epoch_loss += loss.item()*x.shape[0]
                epoch_loss_rec += recon_loss.item()*x.shape[0]
                epoch_loss_ce += bce_loss.item()*x.shape[0]
                epoch_loss_kl += kl_loss.item()*x.shape[0]

            epoch_loss /= len(dataloader.dataset)
            epoch_bar.set_postfix(loss=epoch_loss)
            accelerate.log({"loss": epoch_loss})
            accelerate.log({"recon_loss": epoch_loss_rec/len(dataloader.dataset)})
            accelerate.log({"ce_loss": epoch_loss_ce/len(dataloader.dataset)})
            scheduler.step()

            if epoch_loss > 3*best_loss:
                self.load_state_dict(torch.load(os.path.join(models_dir, 'SemFM', f'SemFM_{self.dataset}_{self.args.size}.pt')))

            if (epoch+1) % self.args.sample_and_save_freq == 0 or epoch==0:
                x, mask = next(iter(testloader))
                mask = mask.to(self.device)
                self.sample_from_mask(mask, self.args.n_steps, shape=x.shape, accelerate=accelerate)
                x = x.to(self.device)
                self.segment_image(x, self.args.n_steps, mask, accelerate=accelerate)
                self.draw_gaussians(accelerate=accelerate)
            
            '''
            if epoch_loss_kl/len(dataloader.dataset) < self.tolerance:
                # disable gradient in the means and variances if the distributions are already far apart
                self.mu.requires_grad = False
                self.var.requires_grad = False
                #dataloader = final_dataloader
                #dataloader.batch_size *= 2
            '''
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                #torch.save(self.state_dict(), os.path.join(models_dir, 'SemFM', f'SemFM_{self.dataset}_{self.args.size}.pt'))
                accelerate.save(self.state_dict(), os.path.join(models_dir, 'SemFM', f'SemFM_{self.dataset}_{self.args.size}.pt'))
        
        accelerate.end_training()
    
    def mask_to_gaussian(self, index, mask, img_shape = None):
        if img_shape is None:
            img_shape = mask.shape
        z = self.prior[index].rsample((img_shape[0],img_shape[2],img_shape[3])).permute(0,3,1,2)
        z = z.to(mask.device)*mask.unsqueeze(1)
        return z
    
    @torch.no_grad()
    def sample_from_mask(self, mask, n_steps, accelerate=None, train=True, shape = None):
        if shape is None:
            shape = mask.shape
        for i in range(self.n_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                noise = self.mask_to_gaussian(i, inst_mask, shape)
            else:
                noise += self.mask_to_gaussian(i, inst_mask, shape)
        x_t = noise
        t = 0.

        if self.ode:

            print('Sampling using ODE solver...')

            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
            
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_t, t=torch.linspace(0, 1, 2).to(self.device), options={'step_size': 1./float(n_steps)}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_t, t=torch.linspace(0, 1, 2).to(self.device), method=self.solver, options={'max_num_steps': n_steps}, rtol=1e-5, atol=1e-5)
            x_t = samples[1]

        else:
            for i in tqdm(range(n_steps), desc='Sampling'):
                x_t = self.unet(x_t,torch.full(x_t.shape[:1], t, device=self.device))*1./n_steps + x_t
                t += 1./n_steps
        
        x_t = (x_t + 1.) / 2.
        x_t = torch.clamp(x_t, 0., 1.)
        mask = mask.long().cpu()
        mask = mask.unsqueeze(1)

        if self.dataset != 'brats':
            fig = plt.figure(figsize=(15, 10))

            mask_color = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3])
            for i in range(self.n_classes):
                mask_color += (mask==i).float() * torch.tensor(self.colors[i])[:, None, None]
            mask_color = mask_color/255.
            mask_grid = make_grid(mask_color, nrow=int(mask.shape[0]**0.5), normalize=True)

            grid = make_grid(x_t, nrow=int(x_t.shape[0]**0.5), normalize=True)
            # plot both grids side by side
            plt.subplot(1, 2, 1)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Sampled Images')
            # remove ticks
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.imshow(mask_grid.permute(1, 2, 0).cpu().numpy())
            plt.title('GT')
            # remove ticks
            plt.xticks([])
            plt.yticks([])
        else:
            # i want 3 grids, each containing all images, but a channel each
            fig, axs = plt.subplots(1, self.channels+1, figsize=(4, 20))
            mask_grid = make_grid(mask, nrow=1, normalize=True)
            axs[0].imshow(mask_grid.permute(1, 2, 0).cpu().numpy())
            axs[0].set_title('GT')
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            labels = ['T1', 'T1c','T2']
            for i in range(self.channels):
                grid = make_grid(x_t[:, i, :, :].unsqueeze(1), nrow=1, normalize=True)
                axs[i+1].imshow(grid.permute(1,2,0).cpu().numpy())
                axs[i+1].set_xticks([])
                axs[i+1].set_yticks([])
                axs[i+1].set_title(labels[i])
            #tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
        if train:
            accelerate.log({"samples": fig})
        else:
            plt.show()
        plt.close(fig)
    
    @torch.no_grad()
    def segment_image(self, x, n_steps, mask, accelerate=None, train=True, test=False):
        '''
        Segment the image
        :param x: input image
        '''
        if self.dequantize:
            x = x + 0.5 / 128.0 # keep it deterministic
        original_x = x.clone()
        x_t = x
        t=1.

        if self.ode:

            print('Segmenting using ODE solver...')

            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
            
            
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_t, t=torch.linspace(1, 0, 2).to(self.device), options={'step_size': 1./float(n_steps)}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_t, t=torch.linspace(1, 0, 2).to(self.device), method=self.solver, options={'max_num_steps': n_steps}, rtol=1e-5, atol=1e-5)
            x_t = samples[1]
        
        #x_t = odeint(f, x_t, 1, 0, phi=self.unet.parameters())

        else:
            for i in tqdm(range(n_steps), desc='Segmenting', leave=not test):
                x_t = -self.unet(x_t,torch.full(x_t.shape[:1], t, device=self.device))*1./n_steps + x_t
                t -= 1./n_steps
        
        x_t = gaussian_to_class(self.mu, x_t.cpu())
        if test:
            return x_t
        # normalize x_t to [0, 1]
        x_t = x_t.long()

        original_x = (original_x + 1.) / 2.
        original_x = torch.clamp(original_x, 0., 1.)

        mask = mask.long().cpu()
        mask = mask.unsqueeze(1)

        if self.dataset != 'brats':
            # color the segmented image
            x_t_color = torch.zeros(x_t.shape[0], 3, x_t.shape[2], x_t.shape[3])

            for i in range(self.n_classes):
                x_t_color += (x_t == i).float() * torch.tensor(self.colors[i])[:, None, None]
            x_t_color = x_t_color/255.

            mask_color = torch.zeros(mask.shape[0], 3, mask.shape[2], mask.shape[3])
            for i in range(self.n_classes):
                mask_color += (mask==i).float() * torch.tensor(self.colors[i])[:, None, None]
            mask_color = mask_color/255.

            fig = plt.figure(figsize=(20, 10))
            grid = make_grid(x_t_color, nrow=int(x_t.shape[0]**0.5), normalize=True)

            # make another grid for the original image
            original_grid = make_grid(original_x, nrow=int(original_x.shape[0]**0.5))

            mask_grid = make_grid(mask_color, nrow=int(mask.shape[0]**0.5), normalize=True)

            # plot both grids
            plt.subplot(1, 3, 1)
            plt.imshow(original_grid.permute(1, 2, 0).cpu().numpy())
            # set title
            plt.title('Original Images')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Segmentations')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(mask_grid.permute(1, 2, 0).cpu().numpy())
            plt.title('GT')
            # remove ticks
            plt.xticks([])
            plt.yticks([])

        else:
            fig, axs = plt.subplots(1, self.channels+2, figsize=(4, 10))
            labels = ['T1', 'T1c','T2']
            for i in range(self.channels):
                grid = make_grid(original_x[:, i, :, :].unsqueeze(1), nrow=1, normalize=True)
                axs[i].imshow(grid.permute(1,2,0).cpu().numpy())
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                axs[i].set_title(labels[i])
            grid = make_grid(x_t, nrow=1, normalize=True)
            axs[self.channels].imshow(grid.permute(1,2,0).cpu().numpy())
            axs[self.channels].set_xticks([])
            axs[self.channels].set_yticks([])
            axs[self.channels].set_title('Segs')
            grid = make_grid(mask, nrow=1, normalize=True)
            axs[self.channels+1].imshow(grid.permute(1,2,0).cpu().numpy())
            axs[self.channels+1].set_xticks([])
            axs[self.channels+1].set_yticks([])
            axs[self.channels+1].set_title('GT')
            plt.tight_layout()

        if train:
            accelerate.log({"segmentation": fig})
        else:
            plt.show()
        plt.close(fig)
        
    def load_model(self, model_path):
        '''
        Load the SemFM model
        :param model_path: path to the model
        '''
        # edit the keys of the state dict if it contains 'module.'
        state_dict = torch.load(model_path)
        # edit state dict keys, remove every "module."
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        self.eval()

    def sample(self, test_loader):
        '''
        Sample from the SemFM model
        :param test_loader: test loader
        '''
        self.unet.eval()
        x, mask = next(iter(test_loader))
        mask = mask.to(self.device)
        self.sample_from_mask(mask, self.args.n_steps, train=False, shape=x.shape)
        x = x.to(self.device)
        self.segment_image(x, self.args.n_steps, mask, train=False)

    @torch.no_grad()
    def eval_model(self, test_loader):
        '''
        Test the SemFM model
        :param test_loader: test loader
        '''
        self.unet.eval()
        # save all the masks in a tensor
        masks = torch.tensor([])
        preds = torch.tensor([])
        for x, mask in tqdm(test_loader, desc='Testing', leave=True):
            mask = mask.to(self.device)
            x = x.to(self.device)
            pred = self.segment_image(x, self.args.n_steps, mask, train=False, test=True)
            # concatenate the masks
            masks = torch.cat((masks, mask.cpu().long()), dim=0)
            preds = torch.cat((preds, pred.cpu().squeeze(1).long()), dim=0)
        # preds and masks should be a one-hot boolean tensor of shape N, n_claases, H,W
        preds = F.one_hot(preds.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        masks = F.one_hot(masks.long(), num_classes=self.n_classes).permute(0, 3, 1, 2)
        miou = mean_iou(preds, masks, num_classes=self.n_classes, include_background=False).mean()
        gds = generalized_dice_score(preds, masks, num_classes=self.n_classes, include_background=False).mean()
        print(f'Mean IoU: {miou:.4f}')
        print(f'Dice Score: {gds:.4f}')

    @torch.no_grad()
    def segment_edit(self, img, n_steps):
        '''
        Segment the image
        :param img: input image
        '''
        self.unet.eval()
        original_img = img.clone()
        x_t = img
        t=1.

        if self.ode:

            print('Segmenting using ODE solver...')

            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
            
            
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_t, t=torch.linspace(1, 0, 2).to(self.device), options={'step_size': 1./float(n_steps)}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_t, t=torch.linspace(1, 0, 2).to(self.device), method=self.solver, options={'max_num_steps': n_steps}, rtol=1e-5, atol=1e-5)
            x_t = samples[1]

        else:
            for i in tqdm(range(n_steps), desc='Segmenting', leave=False):
                x_t = -self.unet(x_t,torch.full(x_t.shape[:1], t, device=self.device))*1./n_steps + x_t
                t -= 1./n_steps
        
        x_map = gaussian_to_class(self.mu, x_t.cpu())
        x_map_color = torch.zeros(x_map.shape[0], 3, x_map.shape[2], x_map.shape[3])

        for i in range(self.n_classes):
            x_map_color += (x_map == i).float() * torch.tensor(self.colors[i])[:, None, None]
        
        return x_t, x_map_color
    
    @torch.no_grad()
    def sample_edit(self, noise, n_steps):
        '''
        Sample from the SemFM model
        :param img: input image
        '''
        self.unet.eval()
        x_t = noise
        t=0.

        if self.ode:

            print('Sampling using ODE solver...')

            def f(t: float, x):
                return self.forward(x, torch.full(x.shape[:1], t, device=self.device))
            
            if self.solver == 'euler' or self.solver == 'rk4' or self.solver == 'midpoint' or self.solver == 'explicit_adams' or self.solver == 'implicit_adams':
                samples = odeint(f, x_t, t=torch.linspace(0, 1, 2).to(self.device), options={'step_size': 1./float(n_steps)}, method=self.solver, rtol=1e-5, atol=1e-5)
            else:
                samples = odeint(f, x_t, t=torch.linspace(0, 1, 2).to(self.device), method=self.solver, options={'max_num_steps': n_steps}, rtol=1e-5, atol=1e-5)
            x_t = samples[1]

        else:
            for i in tqdm(range(n_steps), desc='Sampling', leave=False):
                x_t = self.unet(x_t,torch.full(x_t.shape[:1], t, device=self.device))*1./n_steps + x_t
                t += 1./n_steps
        
        x_t = x_t*0.5 + 0.5
        return x_t
    
    @torch.no_grad()
    def draw_gaussians(self, accelerate=None):
        '''
        Plot the gaussians that represent the classes
        '''
        #3d plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.n_classes):
            samples = self.prior[i].sample((1000,)).cpu().numpy()
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], label=f'Class {i}', color=self.colors[i]/255.)
        ax.set_xlabel('Channel R')
        ax.set_ylabel('Channel G')
        ax.set_zlabel('Channel B')
        # push legend out of the plot, to the top
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=min(6, self.n_classes))
        accelerate.log({"gaussians": wandb.Image(fig)})
        plt.close(fig)
            
        
        

        

                
