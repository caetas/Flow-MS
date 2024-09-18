import argparse
import torch

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=256, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--n_features', type=int, default = 64, help='number of features')
    argparser.add_argument('--init_channels', type=int, default = 32, help='initial channels')
    argparser.add_argument('--channel_scale_factors', type=int, nargs='+', default = [1, 2, 2], help='channel scale factors')
    argparser.add_argument('--resnet_block_groups', type=int, default = 8, help='resnet block groups')
    argparser.add_argument('--use_convnext', type=bool, default = True, help='use convnext (default: True)')
    argparser.add_argument('--convnext_scale_factor', type=int, default = 2, help='convnext scale factor (default: 2)')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--n_samples', type=int, default=16, help='number of samples')
    argparser.add_argument('--n_steps', type=int, default=50, help='number of steps')
    argparser.add_argument('--n_classes', type=int, default=2, help='number of classes')
    argparser.add_argument('--dataset', type=str, default='bccd', help='dataset', choices=['bccd', 'brats', 'celeb', 'cityscapes'])
    argparser.add_argument('--size', type=int, default=64, help='size of image')
    argparser.add_argument('--dist', type=float, default=3.0, help='distance between distributions')
    argparser.add_argument('--var', type=float, default=0.25, help='variance of distribution')
    argparser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    argparser.add_argument('--decay', type=float, default=1e-5, help='decay rate')
    argparser.add_argument('--solver', type=str, default='dopri5', help='solver for ODE', choices=['dopri5', 'rk4', 'dopri8', 'euler', 'bosh3', 'adaptive_heun', 'midpoint', 'explicit_adams', 'implicit_adams'])
    argparser.add_argument('--ode', action='store_true', default=False, help='use ODE solver')
    argparser.add_argument('--clip', action='store_true', default=False, help='clip the gaussians')
    argparser.add_argument('--clip_dist', type=float, default=3.0, help='length of the cube')
    argparser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    argparser.add_argument('--w_seg', type=float, default=0.2, help='weight for the segmentation loss')
    argparser.add_argument('--tolerance', type=float, default=1e-2, help='minimum tolerance for training the gaussians')
    argparser.add_argument('--anchor', action='store_true', default=False, help='anchor the mean of the gaussians during training')
    args = argparser.parse_args()
    args.channel_scale_factors = tuple(args.channel_scale_factors)
    return args

def parse_args_RF():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=128, help='batch size')
    argparser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    argparser.add_argument('--patch_size', type=int, default=2, help='patch size')
    argparser.add_argument('--dim', type=int, default=64, help='dimension')
    argparser.add_argument('--n_layers', type=int, default=6, help='number of layers')
    argparser.add_argument('--n_heads', type=int, default=4, help='number of heads')
    argparser.add_argument('--multiple_of', type=int, default=256, help='multiple of')
    argparser.add_argument('--ffn_dim_multiplier', type=int, default=None, help='ffn dim multiplier')
    argparser.add_argument('--norm_eps', type=float, default=1e-5, help='norm eps')
    argparser.add_argument('--class_dropout_prob', type=float, default=0.1, help='class dropout probability')
    argparser.add_argument('--sample_and_save_freq', type=int, default=5, help='sample and save frequency')
    argparser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    argparser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
    argparser.add_argument('--cfg', type=float, default=1.0, help='label guidance')
    argparser.add_argument('--sample_steps', type=int, default=50, help='number of steps for sampling')
    argparser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    argparser.add_argument('--dataset', type=str, default='bccd', help='dataset', choices=['bccd', 'brats', 'celeb'])
    argparser.add_argument('--size', type=int, default=64, help='size of image')
    argparser.add_argument('--dist', type=float, default=3.0, help='distance between distributions')
    argparser.add_argument('--clip_dist', type=float, default=3.0, help='length of the cube')
    argparser.add_argument('--var', type=float, default=0.25, help='variance of distribution')
    argparser.add_argument('--n_samples', type=int, default=16, help='number of samples')
    return argparser.parse_args()