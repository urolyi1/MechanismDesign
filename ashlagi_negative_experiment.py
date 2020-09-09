import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

# Custom imports
import HospitalGenerators as gens
import Experiment
import util
from matchers import MatchNet
from util import convert_internal_S, all_possible_misreports

SAVE = False
np.random.seed(0)
torch.manual_seed(0)

def visualize_match_outcome(bids, allocation):
    hospital_results = allocation.detach().view(2,4)
    inds = np.arange(4)

    fig, axes = plt.subplots(2)

    bar_width = 0.25
    axes[0].bar(inds, bids[0,0,:].numpy(), bar_width, color='b')
    axes[0].bar(inds + bar_width, bids[0,1,:].numpy(), bar_width, color='r')
    axes[0].set_title('Bids')

    axes[1].bar(inds, hospital_results[0,:].numpy(), bar_width, color='b')
    axes[1].bar(inds, hospital_results[1,:].numpy(), bar_width, color='r')
    axes[1].set_title('Allocation')

    plt.show()

def print_misreport_differences(model, truthful_bids, verbose=False, tolerance=1e-2):
    p1_misreports = torch.tensor(all_possible_misreports(truthful_bids[0, :].numpy()))
    p2_misreports = torch.tensor(all_possible_misreports(truthful_bids[1, :].numpy()))

    if p1_misreports.shape[0] > p2_misreports.shape[0]:
        to_pad = truthful_bids[1, :].repeat(p1_misreports.shape[0] - p2_misreports.shape[0], 1)
        p2_misreports = torch.cat( (p2_misreports, to_pad ))

    elif p2_misreports.shape[0] > p1_misreports.shape[0]:
        to_pad = truthful_bids[0, :].repeat(p2_misreports.shape[0] - p1_misreports.shape[0], 1)
        p1_misreports = torch.cat( (p1_misreports, to_pad ))

    batched_misreports = torch.cat( (p1_misreports.unsqueeze(1), p2_misreports.unsqueeze(1)), dim=1)

    # given batched misreports, we want to calc mis util for each one against truthful bids
    self_mask = torch.zeros(N_HOS, 1, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    found_regret = False
    for batch_ind in range(batched_misreports.shape[0]):
        curr_mis = batched_misreports[batch_ind, :, :].unsqueeze(0)
        mis_input = model.create_combined_misreport(curr_mis, truthful_bids)
        output = model.forward(mis_input, 1 * model.n_hos)
        p = truthful_bids.unsqueeze(0)
        mis_util = model.calc_mis_util(output, p)

        central_util, internal_util = model.calc_util(model.forward(p, 1), p)
        pos_regret = torch.clamp(mis_util - (central_util + internal_util), min=0)
        if verbose and (pos_regret > tolerance).any().item():
            print("Misreport: ", curr_mis)
            print("Regret: ", pos_regret)
        found_regret = found_regret or (pos_regret > tolerance).any().item()
    print('found large positive regret: ', found_regret)
    return found_regret

def full_regret_check(model, test_batches, verbose=False):
    """For each sample given batches checks all possible misreports for regret

    :param model: MatchNet object
    :param test_batches: test_samples
    :param verbose: boolean option for verbose print output
    :return: None
    """
    high_regrets = []
    for batch in range(test_batches.shape[0]):
        for sample in range(test_batches.shape[1]):
            if print_misreport_differences(model, test_batches[batch, sample, :, :]):
                high_regrets.append(test_batches[batch, sample, :, :])
    return high_regrets

# Command line argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--main-lr', type=float, default=5e-2, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=10, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=16, help='batch size')
parser.add_argument('--nbatch', type=int, default=2, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=10.0, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=5.0, help='control strength in cvxpy objective')
args = parser.parse_args()

N_HOS = 2
N_TYP = 4

internal_s = torch.tensor([
                           [1., 0.],
                           [1., 1.],
                           [0., 1.],
                           [0., 1.]
                           ])
central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
# probability for generator
prob_lst1 = [.5, .5, 0.0, 0.0]
prob_lst2 = [0.0, 0.0, .5, .5]

hospital_generator_list = [gens.GenericTypeHospital(prob_lst1, 30), gens.GenericTypeHospital(prob_lst2, 30)]
report_gen = gens.ReportGenerator(hospital_generator_list, (N_HOS, N_TYP))
batches = util.create_train_sample(report_gen, args.nbatch, batch_size=args.batchsize)
test_batches = util.create_train_sample(report_gen, 2, batch_size=10)

example_batch= torch.tensor([
    [
        [[10.0,10.0,0.0,0.0],
         [0.0,0.0,10.0,10.0]],
    ]
])


num_structures = central_s.shape[1]
batch_size = batches.shape[1]
int_structures = internal_s.shape[1]

# Weights matrix for central structures
internal_weight_value = 2.0

individual_weights = torch.zeros(num_structures, N_HOS)
for h in range(N_HOS):
    for col in range(num_structures):
        # Check how many allocated pairs
        allocated = central_s[h * N_TYP: (h+1) * N_TYP, col].sum().item()
        individual_weights[col, h] = allocated

# total value for structures
struct_weights = individual_weights.sum(dim=-1)

# Weights matrix of internal structures
internal_weights = torch.ones(int_structures) * internal_weight_value

model = MatchNet(N_HOS, N_TYP, central_s, internal_s, individual_weights, internal_weights,
                 control_strength=args.control_strength)
# Create experiment
ashlagi_experiment = Experiment.Experiment(args, internal_s, N_HOS, N_TYP, model)
ashlagi_experiment.run_experiment(batches, None, save=SAVE, verbose=True)

#
high_regret = full_regret_check(model, test_batches)
