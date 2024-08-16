from data.Dataloaders import *
import matplotlib.pyplot as plt
from utils.util import parse_args
from models.Flow_MS import FlowMS
import wandb

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'bccd':
        trainloader = train_loader_bccd(batch_size=args.batch_size, size=args.size)
        testloader = test_loader_bccd(batch_size=args.n_samples, size=args.size)

    elif args.dataset == 'brats':
        trainloader = train_loader_brats(batch_size=args.batch_size, size=args.size)
        testloader = test_loader_brats(batch_size=args.n_samples, size=args.size)

    elif args.dataset == 'celeb':
        trainloader = train_loader_celebamaskhq(batch_size=args.batch_size, size=args.size)
        testloader = test_loader_celebamaskhq(batch_size=args.n_samples, size=args.size)

    model = FlowMS(args)

    wandb.init(project='Flow-MS',
            
            config = {
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'n_features': args.n_features,
                    'init_channels': args.init_channels,
                    'channel_scale_factors': args.channel_scale_factors,
                    'resnet_block_groups': args.resnet_block_groups,
                    'use_convnext': args.use_convnext,
                    'convnext_scale_factor': args.convnext_scale_factor,
                    'sample_and_save_freq': args.sample_and_save_freq,
                    'n_samples': args.n_samples,
                    'n_steps': args.n_steps,
                    'n_classes': args.n_classes,
                    'dataset': args.dataset,
                    'size': args.size,
                    'dist': args.dist,
                    'var': args.var,
                    'warmup': args.warmup,
                    'decay': args.decay
                },

                name = f"Flow-MS_{args.dataset}",)
    model.train_model(trainloader, testloader)