from data.Dataloaders import train_loader, test_loader
import matplotlib.pyplot as plt
from utils.util import parse_args
from models.Flow_MS import FlowMS

args = parse_args()

testloader = test_loader(batch_size=args.n_samples)

model = FlowMS(args)
model.load_model(args.checkpoint)
model.sample(testloader)