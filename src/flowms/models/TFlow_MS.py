###########################################################
### Code based on: https://github.com/cloneofsimo/minRF ###
###########################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import wandb
import os
from config import models_dir


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.n_rep = 1
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_layers=5,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=256, n_layers=16, n_heads=32, **kwargs)


def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs)

def mask_to_gaussian(mask, mean, variance, img_shape = None):
    if img_shape is None:
        img_shape = mask.shape
    eps = torch.randn(img_shape)
    z = reparametrize(mean, variance, eps)
    z = z.to(mask.device)*mask.unsqueeze(1)
    return z

def reparametrize(mean, var, eps):
    # random array of size 3,1000
    # make mean the same shape as eps by repeating the mean 1000 times
    mean = mean[:, None, None]
    # make var the same shape as eps
    var = var[:, None, None]
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
    if not os.path.exists(os.path.join(models_dir, "RectifiedFlows")):
        os.makedirs(os.path.join(models_dir, "RectifiedFlows"))

class TFlowMS:
    def __init__(self, args, img_size, channels, ln=True):
        self.model = DiT_Llama(channels, img_size, args.patch_size, args.dim, args.n_layers, args.n_heads, args.multiple_of, args.ffn_dim_multiplier, args.norm_eps, args.class_dropout_prob, 1)
        self.ln = ln
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.num_classes = args.num_classes
        self.channels = channels
        self.img_size = img_size
        self.sample_and_save_freq = args.sample_and_save_freq
        self.dataset = args.dataset
        self.sample_steps = args.sample_steps
        self.cfg = args.cfg
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {model_size}, {model_size / 1e6}M")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.no_wandb = args.no_wandb
        self.mean, self.var = class_to_gaussian(self.num_classes)

    def forward(self, x, cond, mask):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        for i in range(self.num_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                z1 = mask_to_gaussian(inst_mask, self.mean[i], self.var, x.shape)
            else:
                z1 += mask_to_gaussian(inst_mask, self.mean[i], self.var, x.shape)
        z1 = z1.to(x.device)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def get_sample(self, z, cond, mask, null_cond=None, sample_steps=50, cfg=2.0, train=False):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in tqdm(range(sample_steps, 0, -1), desc='Sampling', leave=False):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        imgs = images[-1]
        imgs = imgs*0.5 + 0.5
        x_t = imgs.clamp(0, 1)

        mask = mask.float()
        mask = mask/float(self.num_classes-1)
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
    def get_mask(self, img, cond, mask, null_cond=None, sample_steps=50, cfg=2.0, train=True):
        self.model.eval()
        b = img.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(img.device).view([b, *([1] * len(img.shape[1:]))])
        images = [img]
        original_x = img.clone()
        original_x = original_x*0.5 + 0.5
        original_x = original_x.clamp(0, 1)
        for i in tqdm(range(sample_steps), desc='Segmenting', leave=False):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(img.device)

            vc = self.model(img, t, cond)
            if null_cond is not None:
                vu = self.model(img, t, null_cond)
                vc = vu + cfg * (vc - vu)

            img = img + dt * vc
            images.append(img)
        
        imgs = images[-1]
        x_t = gaussian_to_class(self.mean, imgs.cpu())
        x_t = x_t/float(self.num_classes-1)

        mask = mask.float()
        mask = mask/float(self.num_classes-1)
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

    def train_model(self, train_loader, test_loader):
        
        create_checkpoint_dir()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        epoch_bar = trange(self.n_epochs, desc="Epochs")

        best_loss = float("inf")
        for epoch in epoch_bar:
            self.model.train()
            train_loss = 0
            for x, mask in tqdm(train_loader, desc='Batches', leave=False):
                x = x.to(self.device)
                cond = torch.zeros(x.size(0), dtype=torch.int).to(self.device)
                optimizer.zero_grad()
                loss, _ = self.forward(x, cond, mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*x.shape[0]
            epoch_bar.set_postfix(loss=train_loss / len(train_loader.dataset))
            if not self.no_wandb:
                wandb.log({"train_loss": train_loss / len(train_loader.dataset)})

            if train_loss/len(train_loader.dataset) < best_loss:
                best_loss = train_loss/len(train_loader.dataset)
                torch.save(self.model.state_dict(), os.path.join(models_dir, "RectifiedFlows", f"RF_{self.dataset}.pt"))
        
            if epoch ==0 or ((epoch+1) % self.sample_and_save_freq == 0):
                # get one batch of data
                x, mask = next(iter(test_loader))
                x = x.to(self.device)
                cond = torch.zeros(x.size(0), dtype=torch.int).to(self.device)
                mask = mask.to(self.device)
                self.sample(x, 16, mask, train=True)

    def sample(self, x, num_samples, mask, train=False):
        self.model.eval()
        cond = torch.zeros(num_samples, dtype=torch.int).to(self.device)
        for i in range(self.num_classes):
            inst_mask = (mask == i).float()
            if i == 0:
                z = mask_to_gaussian(inst_mask, self.mean[i], self.var, (num_samples, self.channels, self.img_size, self.img_size))
            else:
                z += mask_to_gaussian(inst_mask, self.mean[i], self.var, (num_samples,self.channels, self.img_size, self.img_size)) 
        self.get_sample(z, cond, mask, train=train, sample_steps=self.sample_steps, cfg=self.cfg)
        self.get_mask(x, cond, mask, train=train, sample_steps=self.sample_steps, cfg=self.cfg)

    def load_checkpoint(self, checkpoint):
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))