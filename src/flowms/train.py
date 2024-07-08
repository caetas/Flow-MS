from data.Dataloaders import train_loader, test_loader
import matplotlib.pyplot as plt
from utils.util import parse_args
from models.Flow_MS import FlowMS
import wandb

args = parse_args()

trainloader = train_loader(batch_size=args.batch_size)
testloader = test_loader(batch_size=args.n_samples)

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
                'n_steps': args.n_steps
              },

              name = 'Flow-MS')
model.train_model(trainloader, testloader)
