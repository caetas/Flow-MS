from data.Dataloaders import *
import matplotlib.pyplot as plt
from utils.util import parse_args
from models.Flow_MS import FlowMS

args = parse_args()

if args.dataset == 'bccd':
    testloader = test_loader_bccd(batch_size=args.n_samples, size=args.size)
elif args.dataset == 'brats':
    testloader = test_loader_brats(batch_size=args.n_samples, size=args.size)
elif args.dataset == 'celeb':
    testloader = test_loader_celebamaskhq(batch_size=args.n_samples, size=args.size)

model = FlowMS(args)
model.load_model(args.checkpoint)
model.sample(testloader)