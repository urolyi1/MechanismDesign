import torch
from torch import nn, optim
import torch.nn.functional as F
from argparse import ArgumentParser
from double_net.double_net import DoubleNet, train_loop
from double_net import datasets as ds
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=2048)
parser.add_argument('--test-num-examples', type=int, default=3000)
parser.add_argument('--test-iter', type=int, default=5)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('--model-lr', type=float, default=1e-3)
parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--rho', type=float, default=0.5)
parser.add_argument('--rho-incr-iter', type=int, default=5)
parser.add_argument('--rho-incr-amount', type=float, default=1)
parser.add_argument('--lagr-update-iter', type=int, default=6)
parser.add_argument('--rgt-start', type=int, default=0)

# dataset selection: specifies a configuration of agent/item/valuation
parser.add_argument('--dataset', nargs='+', default=[])
parser.add_argument('--resume', default="")

# architectural arguments
parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--name', default='testing_name')

dataset_name = "double_net/1x2-pv.csv"
args = parser.parse_args()

item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, dataset_name)
clamp_op = ds.get_clamp_op(item_ranges)

model = DoubleNet(
    args.n_agents, args.n_items, clamp_op
).to(device)

train_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.num_examples, item_ranges)
train_loader = ds.Dataloader(train_data, batch_size=512, shuffle=True)

train_loop(model, train_loader, args, device=device)