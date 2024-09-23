from data.Dataloaders import *
import matplotlib.pyplot as plt
from utils.util import parse_args_SemFM
from models.SemFM import SemFM
import wandb

if __name__ == '__main__':
    args = parse_args_SemFM()

    if args.dataset == 'bccd':
        trainloader = train_loader_bccd(batch_size=args.batch_size, size=args.size, num_workers=args.num_workers)
        testloader = test_loader_bccd(batch_size=args.n_samples, size=args.size)

    elif args.dataset == 'brats':
        trainloader = train_loader_brats(batch_size=args.batch_size, size=args.size, num_workers=args.num_workers)
        testloader = test_loader_brats(batch_size=args.n_samples, size=args.size)

    elif args.dataset == 'celeb':
        trainloader = train_loader_celebamaskhq(batch_size=args.batch_size, size=args.size, num_workers=args.num_workers)
        #init_trainloader, final_trainloader = train_loader_celebamaskhq(batch_size=args.batch_size, size=args.size, num_workers=args.num_workers, double=True)
        testloader = test_loader_celebamaskhq(batch_size=args.n_samples, size=args.size)

    elif args.dataset == 'cityscapes':
        trainloader = train_loader_cityscapes(batch_size=args.batch_size, size=args.size, num_workers=args.num_workers)
        testloader = test_loader_cityscapes(batch_size=args.n_samples, size=args.size)

    model = SemFM(args)
    model.train_model(trainloader, testloader)