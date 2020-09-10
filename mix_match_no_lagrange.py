import numpy as np
import torch
import argparse

# Custom imports
import match_net.HospitalGenerators as gens
import match_net.match_net_torch as mn
from match_net import maximum_match as mm
import match_net.util as util
from match_net.match_net import MatchNet
from match_net.util import convert_internal_S, all_possible_misreports

SAVE = False

# Random Seeds
np.random.seed(0)
torch.manual_seed(0)

# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--main-lr', type=float, default=1e-2, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=10, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--nbatch', type=int, default=2, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=10.0, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=1.0, help='control strength in cvxpy objective')
args = parser.parse_args()

VALID_STRUCTURES_INDS = [1, 7, 10, 12, 16, 21]
N_HOS = 2
N_TYP = 7
HOS1_PROBS = [0.25, 0, 0, 0.25, 0.25, 0.25, 0]
HOS2_PROBS = [0, 0.33, 0.33, 0, 0, 0, 0.34]