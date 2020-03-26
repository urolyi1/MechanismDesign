import numpy as np
import torch
import argparse
import datetime
import os
import json

# Custom imports
import HospitalGenerators as gens
import match_net_torch as mn
import Experiment
from matchers import MatchNet, GreedyMatcher
from util import convert_internal_S
import matplotlib.pyplot as plt

SAVE = False


def visualize_match_outcome(bids, allocation):
    hospital_results = allocation.detach().view(2,7)
    inds = np.arange(7)

    fig, axes = plt.subplots(2)

    bar_width = 0.25
    axes[0].bar(inds, bids[0,0,:].numpy(), bar_width, color='b')
    axes[0].bar(inds + bar_width, bids[0,1,:].numpy(), bar_width, color='r')
    axes[0].set_title('Bids')

    axes[1].bar(inds, hospital_results[0,:].numpy(), bar_width, color='b')
    axes[1].bar(inds, hospital_results[1,:].numpy(), bar_width, color='r')
    axes[1].set_title('Allocation')

    plt.show()





# Command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--main-lr', type=float, default=5e-2, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=40, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=2, help='batch size')
parser.add_argument('--nbatch', type=int, default=3, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=1.0, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=1.0, help='control strength in cvxpy objective')
args = parser.parse_args()

N_HOS = 2
N_TYP = 7
hos1_probs = [0.25, 0, 0, 0.25, 0.25, 0.25, 0]
hos2_probs = [0, 0.33, 0.33, 0, 0, 0, 0.34]

hos_gen_lst = [gens.GenericTypeHospital(hos1_probs, 10),
               gens.GenericTypeHospital(hos2_probs, 10)]

generator = gens.ReportGenerator(hos_gen_lst, (N_HOS, N_TYP))
# batches = torch.tensor([
#     [[[10.0000, 0.0000, 0.0000, 10.0000, 10.0000, 10.0000, 0.0000],
#         [0.0000, 10.0, 10.0000, 0.0000, 0.0000, 0.0000, 10.0000]]]])
batches = torch.tensor([
    [[[10.0000, 0.0000, 0.0000, 10.0000, 10.0000, 10.0000, 0.0000],
        [0.0000, 10.0, 10.0000, 0.0000, 0.0000, 0.0000, 10.0000]]],
    [[[10.0000, 0.0000, 0.0000, 10.0000, 10.0000, 10.0000, 0.0000],
      [0.0000, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 10.0000]]],
[[[10.0000, 0.0000, 0.0000, 10.0000, 10.0000, 10.0000, 0.0000],
        [0.0000, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 10.0000]]]
])
strategic_batch_1 = torch.tensor([
    [[[10.0000, 0.0000, 0.0000, 10.0000, 10.0000, 10.0000, 0.0000],
        [0.0000, 0.0, 0.0000, 0.0000, 0.0000, 0.0000, 10.0000]]]
])

strategic_batch_2 = torch.tensor([
    [[[10.0000, 0.0000, 0.0000, 0.0000, 0.0000, 10.0000, 0.0000],
      [0.0000, 10.0, 10.0000, 0.0000, 0.0000, 0.0000, 10.0000]]]
])

ashlagi_compat_dict = {}
for i in range(1, N_TYP - 1):
    ashlagi_compat_dict[i] = []
    ashlagi_compat_dict[i].append(i - 1)
    ashlagi_compat_dict[i].append(i + 1)
ashlagi_compat_dict[0] = [1]
ashlagi_compat_dict[N_TYP - 1] = [N_TYP - 2]

internal_s = torch.tensor(np.load('type_matrix/ashlagi_7_type.npy'),
                          requires_grad=False, dtype=torch.float32)

central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
num_structures = central_s.shape[1]
batch_size = batches.shape[1]
int_structures = internal_s.shape[1]

# Make directory and save args
prefix = f'mix_match_{mn.curr_timestamp()}/'

model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s,
                 control_strength=args.control_strength)

# Create experiment
ashlagi_experiment = Experiment.Experiment(args, internal_s, N_HOS, N_TYP, model, dir=prefix)
ashlagi_experiment.run_experiment(batches, batches, save=SAVE, verbose=True)
print('allocations on batch ', batches)
allocs = model.forward(batches[0:1], batch_size) @ central_s.transpose(0, 1)
visualize_match_outcome(batches[0], allocs)
print(allocs.view(2, 7))
print('allocations on misreport ', strategic_batch_1)
allocs = model.forward(strategic_batch_1, batch_size) @ central_s.transpose(0, 1)
print(allocs.view(2, 7))
visualize_match_outcome(strategic_batch_1[0], allocs)
print('allocations on misreport ', strategic_batch_2)
allocs = model.forward(strategic_batch_2, batch_size) @ central_s.transpose(0, 1)
print(allocs.view(2, 7))
visualize_match_outcome(strategic_batch_2[0], allocs)
