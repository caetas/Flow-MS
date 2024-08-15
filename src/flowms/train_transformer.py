from data.Dataloaders import *
import matplotlib.pyplot as plt
from utils.util import parse_args_RF
from models.TFlow_MS import TFlowMS
import wandb

args = parse_args_RF()

if args.dataset == 'bccd':
    trainloader = train_loader_bccd(batch_size=args.batch_size, size=args.size)
    testloader = test_loader_bccd(batch_size=args.n_samples, size=args.size)

elif args.dataset == 'brats':
    trainloader = train_loader_brats(batch_size=args.batch_size, size=args.size)
    testloader = test_loader_brats(batch_size=args.n_samples, size=args.size)

elif args.dataset == 'celeb':
    trainloader = train_loader_celebamaskhq(batch_size=args.batch_size, size=args.size)
    testloader = test_loader_celebamaskhq(batch_size=args.n_samples, size=args.size)

model = TFlowMS(args, args.size, 3)

wandb.init(project='TFlow-MS',
           
              config = {
                    'batch_size': args.batch_size,
                    'n_epochs': args.n_epochs,
                    'lr': args.lr,
                    'patch_size': args.patch_size,
                    'dim': args.dim,
                    'n_layers': args.n_layers,
                    'n_heads': args.n_heads,
                    'multiple_of': args.multiple_of,
                    'ffn_dim_multiplier': args.ffn_dim_multiplier,
                    'norm_eps': args.norm_eps,
                    'class_dropout_prob': args.class_dropout_prob,
                    'num_classes': args.num_classes,
                    'cfg': args.cfg,
                    'sample_steps': args.sample_steps,
                    'dataset': args.dataset,
                    'size': args.size,
                    'dist': args.dist,
                    'var': args.var,
                    'n_samples': args.n_samples
                },
                name = f"TFlow-MS_{args.dataset}",)

model.train_model(trainloader, testloader)