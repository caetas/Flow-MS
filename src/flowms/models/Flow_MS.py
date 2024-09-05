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

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Attention(nn.Module):
    def __init__(self, num_channels, num_heads=4, head_dim=32):
        '''
        Attention module
        :param num_channels: number of channels in the input image
        :param num_heads: number of heads in the multi-head attention
        :param head_dim: dimension of each head
        '''
        super().__init__()
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        self.to_qkv = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1)
        
    def forward(self, x):
        '''
        Forward pass of the attention module
        :param x: input image
        '''
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        '''
        Block module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param groups: number of groups for group normalization
        '''
        super().__init__()
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_gruops=groups, num_channels=out_channels)
        self.activation = nn.SiLU()

    def forward(self, x, scale_shift=None):
        '''
        Forward pass of the block module
        :param x: input image
        :param scale_shift: scale and shift values
        '''
        x = self.projection(x)
        x = self.group_norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return x

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, channel_scale_factor=2, normalize=True):
        '''
        ConvNextBlock module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param time_embedding_dim: dimension of the time embedding
        :param channel_scale_factor: scaling factor for the number of channels
        :param normalize: whether to normalize the output
        '''
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.GELU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=in_channels)
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.ds_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, groups=in_channels))

        self.net = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * channel_scale_factor, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=1, num_channels=out_channels * channel_scale_factor), 
            nn.Conv2d(in_channels=out_channels * channel_scale_factor, out_channels=out_channels, kernel_size=3, padding=1),
        )

        self.residual_connection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        '''
        Forward pass of the ConvNextBlock module
        :param x: input image
        :param time_emb: time embedding
        '''
        h = self.ds_conv(x)
        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            condition = self.time_projection(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        

        h = self.net(h)
        return h + self.residual_connection(x)
    
class Downsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class LinearAttention(nn.Module):
    def __init__(self, num_channels, num_heads=4, head_dim=32):
        '''
        LinearAttention module
        :param num_channels: number of channels in the input image
        :param num_heads: number of heads in the multi-head attention
        :param head_dim: dimension of each head
        '''
        super().__init__()
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        hidden_dim = head_dim * num_heads
        self.to_qkv = nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim * 3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1), 
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
        )

    def forward(self, x):
        '''
        Forward pass of the linear attention module
        :param x: input image
        '''
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.num_heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)

        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.num_heads, x=h, y=w)
        return self.to_out(out)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        '''
        SinusoidalPositionEmbeddings module
        :param dim: dimension of the sinusoidal position embeddings
        '''
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.partial_embeddings = math.log(10000) / (self.half_dim - 1)
        
    
    def forward(self, time):
        device = time.device 
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -self.partial_embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PreNorm(nn.Module):
    def __init__(self, num_channels, fn):
        super().__init__()
        self.fn = fn
        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=num_channels)

    def forward(self, x):
        x = self.group_norm(x)
        return self.fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class ResNetBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, *, time_embedding_dim=None, groups=8):
        '''
        ResNetBlock module
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param time_embedding_dim: dimension of the time embedding
        :param groups: number of groups for group normalization
        '''
        super().__init__()
        self.time_projection = (
            nn.Sequential(
                nn.SiLU(), 
                nn.Linear(in_features=time_embedding_dim, out_features=out_channels) 
            )
            if exists(x=time_embedding_dim)
            else None
        )

        self.block1 = Block(in_channels=in_channels, out_channels=out_channels, groups=groups)
        self.block2 = Block(in_channels=out_channels, out_channels=out_channels, groups=groups)
        self.residual_connection = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(x=self.time_projection) and exists(x=time_emb):
            assert exists(x=time_emb), "time embedding must be passed in"
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.residual_connection(x)
    
class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_features, init_channels=None, out_channels=None, channel_scale_factors=(1, 2, 4, 8), in_channels=3, with_time_emb=True, resnet_block_groups=8, use_convnext=True, convnext_scale_factor=2):
        '''
        UNet module
        :param n_features: number of features
        :param init_channels: number of initial channels
        :param out_channels: number of output channels
        :param channel_scale_factors: scaling factors for the number of channels
        :param in_channels: number of input channels
        :param with_time_emb: whether to use time embeddings
        :param resnet_block_groups: number of groups for group normalization in the ResNet block
        :param use_convnext: whether to use ConvNext block
        :param convnext_scale_factor: scaling factor for the number of channels in the ConvNext block
        '''
        super().__init__()

        # determine dimensions
        self.in_channels = in_channels

        init_channels = default(init_channels, n_features // 3 * 2)
        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=init_channels, kernel_size=7, padding=3)

        dims = [init_channels, *map(lambda m: n_features * m, channel_scale_factors)]
        resolution_translations = list(zip(dims[:-1], dims[1:]))
        
        if use_convnext:
            block_klass = partial(ConvNextBlock, channel_scale_factor=convnext_scale_factor)
        else:
            block_klass = partial(ResNetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = n_features * 4
            self.time_projection = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=n_features),
                nn.Linear(in_features=n_features, out_features=time_dim),
                nn.GELU(),
                nn.Linear(in_features=time_dim, out_features=time_dim),
            )
        else:
            time_dim = None
            self.time_projection = None

        # layers
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        num_resolutions = len(resolution_translations)

        for idx, (in_chan, out_chan) in enumerate(resolution_translations):
            is_last = idx >= (num_resolutions - 1)
            self.encoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=in_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=out_chan, out_channels=out_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=out_chan, fn=LinearAttention(num_channels=out_chan))),
                        Downsample(num_channels=out_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        bottleneck_capacity = dims[-1]
        self.mid_block1 = block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim)
        self.mid_attn = Residual(PreNorm(bottleneck_capacity, Attention(bottleneck_capacity)))
        self.mid_block2 = block_klass(bottleneck_capacity, bottleneck_capacity, time_embedding_dim=time_dim)

        

        for idx, (in_chan, out_chan) in enumerate(reversed(resolution_translations[1:])):
            is_last = idx >= (num_resolutions - 1)

            self.decoder.append(
                nn.ModuleList(
                    [
                        block_klass(in_channels=out_chan * 2, out_channels=in_chan, time_embedding_dim=time_dim),
                        block_klass(in_channels=in_chan, out_channels=in_chan, time_embedding_dim=time_dim),
                        Residual(fn=PreNorm(num_channels=in_chan, fn=LinearAttention(num_channels=in_chan))),
                        Upsample(num_channels=in_chan) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_chan = default(out_channels, in_channels)
        self.final_conv = nn.Sequential(
            block_klass(in_channels=n_features, out_channels=n_features), 
            nn.Conv2d(in_channels=n_features, out_channels=out_chan, kernel_size=1)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        t = self.time_projection(time) if exists(self.time_projection) else None

        noisy_latent_representation_stack = []

        # downsample
        for block1, block2, attn, downsample in self.encoder:
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            noisy_latent_representation_stack.append(x)
            x = downsample(x)
        
        # bottleneck
        x = self.mid_block1(x, time_emb=t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=t)

        # upsample
        for block1, block2, attn, upsample in self.decoder:
            x = torch.cat((x, noisy_latent_representation_stack.pop()), dim=1)
            x = block1(x, time_emb=t)
            x = block2(x, time_emb=t)
            x = attn(x)
            x = upsample(x)
        
        return self.final_conv(x)

#def reparametrize(noise, mu, sigma):
#    return noise * sigma + mu

def mask_to_gaussian(mask, mean, variance, dist=None, img_shape = None):
    if img_shape is None:
        img_shape = mask.shape
    eps = torch.randn(img_shape)
    z = reparametrize(mean, variance, eps, dist)
    z = z.to(mask.device)*mask.unsqueeze(1)
    return z

def reparametrize(mean, var, eps, dist = None):
    # make mean the same shape as eps by repeating the mean 1000 times
    mean = mean[:, None, None]
    # make var the same shape as eps
    var = var[:, None, None]
    if dist is not None:
        # clip the values of each channel to be within mean - dist and mean + dist
        return torch.clamp(mean + eps * torch.sqrt(var), min=mean-dist/2 + 1e-6, max=mean+dist/2+1e-6)
    else:
        return mean + eps * torch.sqrt(var)

def single_plane(n_classes, dist, z=0):
    if n_classes == 1:
        mean = [torch.tensor([0, 0, z])]
    elif n_classes == 2:
        mean = [torch.tensor([-dist/2, 0, z]), torch.tensor([dist/2, 0, z])]
    elif n_classes == 3:
        mean = [torch.tensor([-dist, 0, z]), torch.tensor([0, 0, z]), torch.tensor([dist, 0, z])]
    elif n_classes == 4:
        mean = [torch.tensor([-dist/2, -dist/2, z]), torch.tensor([dist/2, -dist/2, z]), torch.tensor([-dist/2, dist/2, z]), torch.tensor([dist/2, dist/2, z])]
    elif n_classes == 5:
        mean = [torch.tensor([-dist, 0, z]), torch.tensor([0, 0, z]), torch.tensor([dist, 0, z]), torch.tensor([0, dist, z]), torch.tensor([0, -dist, z])]
    elif n_classes == 6:
        mean = [torch.tensor([-dist/2, -dist, z]), torch.tensor([dist/2, -dist, z]), torch.tensor([-dist/2, 0, z]), torch.tensor([dist/2, 0, z]), torch.tensor([-dist/2, dist, z]), torch.tensor([dist/2, dist, z])]
    elif n_classes == 7:
        mean = [torch.tensor([-dist, -dist, z]), torch.tensor([dist, -dist, z]), torch.tensor([-dist, 0, z]), torch.tensor([0, 0, z]), torch.tensor([dist, 0, z]), torch.tensor([-dist, dist, z]), torch.tensor([dist, dist, z])]
    elif n_classes == 8:
        mean = [torch.tensor([-dist, -dist, z]), torch.tensor([0, -dist, z]), torch.tensor([dist, -dist, z]), torch.tensor([-dist, 0, z]), torch.tensor([dist, 0, z]), torch.tensor([-dist, dist, z]), torch.tensor([0, dist, z]), torch.tensor([dist, dist, z])]
    else:
        mean = [torch.tensor([-dist, -dist, z]), torch.tensor([0, -dist, z]), torch.tensor([dist, -dist, z]), torch.tensor([-dist, 0, z]), torch.tensor([0, 0, z]), torch.tensor([dist, 0, z]), torch.tensor([-dist, dist, z]), torch.tensor([0, dist, z]), torch.tensor([dist, dist, z])]
    return mean

def class_to_gaussian(n_classes, dist=2.5, variance = 0.5):
    assert n_classes <= 27, "Only 27 classes are supported"
    assert n_classes > 1, "At least 2 classes are needed"
    mean = []
    if n_classes <= 3:
        var = torch.tensor([variance, 1, 1]) # only need to use 1 channel
        mean = single_plane(n_classes, dist)
    elif n_classes <= 9:
        var = torch.tensor([variance, variance, 1]) # need 2 channels only
        mean = single_plane(n_classes, dist)
    else:
        if n_classes <= 18:
            var = torch.tensor([variance, variance, variance])
            mean = single_plane(9, dist, -dist/2)
            mean.extend(single_plane(n_classes-9, dist, dist/2))
        else:
            var = torch.tensor([variance, variance, variance])
            mean = single_plane(9, dist, -dist)
            mean.extend(single_plane(9, dist, 0))
            mean.extend(single_plane(n_classes-18, dist, dist))
    return mean, var

def gaussian_to_class(mean, map):
    class_dist = []
    for m in mean:
        if len(map.shape) > 2:
            dist = abs(map - m[:, None, None])
        else:
            dist = abs(map - m[:, None])
        class_dist.append(torch.mean(dist, axis=1))
    class_dist = torch.stack(class_dist)
    class_map = torch.argmin(class_dist, axis=0).unsqueeze(1)

    return class_map.float()


def create_checkpoint_dir():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(os.path.join(models_dir, 'FlowMS')):
        os.makedirs(os.path.join(models_dir, 'FlowMS'))

class FlowMS(nn.Module):

    def __init__(self, args, channels=3):
        '''
        FlowMS model
        :param args: arguments
        '''
        super(FlowMS, self).__init__()
        self.args = args
        self.channels = channels
        self.unet = UNet(n_features=args.n_features, init_channels=args.init_channels, out_channels=channels, channel_scale_factors=args.channel_scale_factors, in_channels=channels, with_time_emb=True, resnet_block_groups=args.resnet_block_groups, use_convnext=args.use_convnext, convnext_scale_factor=args.convnext_scale_factor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet.to(self.device)
        self.mean, self.var = class_to_gaussian(args.n_classes, dist = args.dist, variance=args.var)
        self.n_classes = args.n_classes
        self.dataset = args.dataset
        self.warmup = args.warmup
        self.decay = args.decay
        self.ode = args.ode
        self.solver = args.solver
        self.clip = args.clip
        if self.clip:
            self.dist = args.clip_dist
        else:
            self.dist = None

    def forward(self, x, time):
        '''
        Forward pass of the FlowMS model
        :param x: input image
        :param time: time embedding
        '''
        return self.unet(x, time)
    
    def conditional_flow_matching_loss(self, x, mask):
        sigma_min = 1e-4
        t = torch.rand(x.shape[0], device=x.device)
        #noise = torch.randn_like(x)
        #noise = (mask>0.).float() * reparametrize(noise, 2, 0.75) + (mask<=0.).float() * reparametrize(noise, -2, 0.75)
        for i in range(self.n_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                noise = mask_to_gaussian(inst_mask, self.mean[i], self.var, self.dist, x.shape)
            else:
                noise += mask_to_gaussian(inst_mask, self.mean[i], self.var, self.dist, x.shape)

        x_t = (1 - (1 - sigma_min) * t[:, None, None, None]) * noise + t[:, None, None, None] * x
        optimal_flow = x - (1 - sigma_min) * noise
        predicted_flow = self.unet(x_t, t)

        return (predicted_flow - optimal_flow).square().mean()
    
    @torch.no_grad()
    def sample_from_mask(self, mask, n_steps, train=True, shape = None):
        if shape is None:
            shape = mask.shape
        for i in range(self.n_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                noise = mask_to_gaussian(inst_mask, self.mean[i], self.var, self.dist, shape)
            else:
                noise += mask_to_gaussian(inst_mask, self.mean[i], self.var, self.dist, shape)
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
        mask = mask.float()
        mask = mask/float(self.n_classes-1)
        mask = torch.clamp(mask, 0., 1.)
        mask = mask.unsqueeze(1)

        if self.dataset != 'brats':
            fig = plt.figure(figsize=(15, 10))
            mask_grid = make_grid(mask, nrow=int(mask.shape[0]**0.5), normalize=True)
            grid = make_grid(x_t, nrow=int(x_t.shape[0]**0.5), normalize=True)
            # plot both grids side by side
            plt.subplot(1, 2, 1)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Sampled Images')
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
            wandb.log({"samples": fig})
        else:
            plt.show()
        plt.close(fig)
    
    @torch.no_grad()
    def segment_image(self, x, n_steps, mask, train=True):
        '''
        Segment the image
        :param x: input image
        '''
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
            for i in tqdm(range(n_steps), desc='Segmenting'):
                x_t = -self.unet(x_t,torch.full(x_t.shape[:1], t, device=self.device))*1./n_steps + x_t
                t -= 1./n_steps
        
        x_t = gaussian_to_class(self.mean, x_t.cpu())
        # normalize x_t to [0, 1]
        x_t = x_t/float(self.n_classes-1)
        #x_t = (x_t + 1.) / 2.
        #x_t = torch.clamp(x_t, 0., 1.)
        # average the channels
        #x_t = x_t.mean(dim=1, keepdim=True)
        #x_t = x_t[:, 0, :, :].unsqueeze(1)
        #x_t[x_t > 0.5] = 1.
        #x_t[x_t <= 0.5] = 0.

        original_x = (original_x + 1.) / 2.
        original_x = torch.clamp(original_x, 0., 1.)

        mask = mask.float()
        mask = mask/float(self.n_classes-1)
        mask = torch.clamp(mask, 0., 1.)
        mask = mask.unsqueeze(1)

        if self.dataset != 'brats':

            fig = plt.figure(figsize=(20, 10))
            grid = make_grid(x_t, nrow=int(x_t.shape[0]**0.5), normalize=True)

            # make another grid for the original image
            original_grid = make_grid(original_x, nrow=int(original_x.shape[0]**0.5))

            mask_grid = make_grid(mask, nrow=int(mask.shape[0]**0.5), normalize=True)

            # plot both grids
            plt.subplot(1, 3, 1)
            plt.imshow(original_grid.permute(1, 2, 0).cpu().numpy())
            # set title
            plt.title('Original Images')
            plt.subplot(1, 3, 2)
            plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            plt.title('Segmentations')
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
            wandb.log({"segmentation": fig})
        else:
            plt.show()
        plt.close(fig)
        

    def train_model(self, dataloader, testloader=None):
        '''
        Train the FlowMS model
        :param dataloader: dataloader for the dataset
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.decay)
        epoch_bar = trange(self.args.n_epochs, desc='Epochs', leave=True)
        create_checkpoint_dir()

        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_loss = float('inf')
        for epoch in epoch_bar:
            epoch_loss = 0.
            self.unet.train()
            for x, mask in tqdm(dataloader, desc='Batches', leave=False):
                x = x.to(self.device)
                mask = mask.to(self.device)
                optimizer.zero_grad()
                loss = self.conditional_flow_matching_loss(x, mask)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()*x.shape[0]
            epoch_loss /= len(dataloader.dataset)
            epoch_bar.set_postfix(loss=epoch_loss)
            wandb.log({"loss": epoch_loss})
            scheduler.step()

            if (epoch+1) % self.args.sample_and_save_freq == 0 or epoch==0:
                x, mask = next(iter(testloader))
                mask = mask.to(self.device)
                self.sample_from_mask(mask, self.args.n_steps, shape=x.shape)
                x = x.to(self.device)
                self.segment_image(x, self.args.n_steps, mask)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.unet.state_dict(), os.path.join(models_dir, 'FlowMS', f'FlowMS_{self.dataset}.pt'))

    def load_model(self, model_path):
        '''
        Load the FlowMS model
        :param model_path: path to the model
        '''
        self.unet.load_state_dict(torch.load(model_path))
        self.unet.eval()

    def sample(self, test_loader):
        '''
        Sample from the FlowMS model
        :param test_loader: test loader
        '''
        self.unet.eval()
        x, mask = next(iter(test_loader))
        mask = mask.to(self.device)
        self.sample_from_mask(mask, self.args.n_steps, train=False, shape=x.shape)
        x = x.to(self.device)
        self.segment_image(x, self.args.n_steps, mask, train=False)
        

                
