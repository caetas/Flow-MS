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
    argparser.add_argument('--dataset', type=str, default='bccd', help='dataset', choices=['bccd', 'brats', 'celeb'])
    argparser.add_argument('--size', type=int, default=64, help='size of image')
    argparser.add_argument('--dist', type=float, default=3.0, help='distance between distributions')
    argparser.add_argument('--var', type=float, default=0.25, help='variance of distribution')
    args = argparser.parse_args()
    args.channel_scale_factors = tuple(args.channel_scale_factors)
    return args
