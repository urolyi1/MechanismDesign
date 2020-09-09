import numpy as np
import torch
import argparse
import os
import json

# Custom imports
import HospitalGenerators as gens
import match_net_torch as mn
import Experiment
import util
from matchers import MatchNet
from util import convert_internal_S

SAVE = False

# Command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--main-lr', type=float, default=5e-2, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=2, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--nbatch', type=int, default=2, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=10, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=.25, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=5.0, help='control strength in cvxpy objective')
args = parser.parse_args()

# lower and upper bound for hospital bid generator
lower_lst = [[10, 20], [30, 60]]
upper_lst = [[20, 40], [50, 100]]

# Create generator and batches
generator = gens.create_simple_generator(lower_lst, upper_lst, 2, 2)
batches = util.create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
test_batches = util.create_train_sample(generator, args.nbatch, batch_size=args.batchsize)


# parameters
N_HOS = 2
N_TYP = 2
num_structures = 4
int_structures = 1
batch_size = batches.shape[1]

# single structure matrix
internal_s = torch.tensor([[1.0],
                           [1.0]], requires_grad=False)
central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS), requires_grad=False, dtype=torch.float32)

# Make directory and save args
prefix = f'two_two_test_{mn.curr_timestamp()}/'
if SAVE:
    os.mkdir(prefix)
    print(vars(args))
    with open(prefix+'argfile.json', 'w') as f:
        json.dump(vars(args), f)
    np.save(prefix+'internal_s.npy', internal_s.numpy())  # save structure matrix

# Create the model
model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s, control_strength=args.control_strength)

# Create experiment
two_two_experiment = Experiment.Experiment(args, internal_s, N_HOS, N_TYP, model, dir=prefix)
two_two_experiment.run_experiment(batches, test_batches, save=SAVE)