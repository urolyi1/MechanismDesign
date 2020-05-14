import numpy as np
import torch
import argparse

# Custom imports
import HospitalGenerators as gens
import match_net_torch as mn
import maximum_match as mm
import Experiment
import util
from matchers import MatchNet, GreedyMatcher
from util import convert_internal_S, all_possible_misreports, find_internal_two_cycles
import matplotlib.pyplot as plt

SAVE = False
np.random.seed(1500)
torch.manual_seed(1500)


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
            if print_misreport_differences(model, test_batches[batch, sample, :, :], verbose):
                high_regrets.append(test_batches[batch, sample, :, :])
    return high_regrets

# enumerating all possible misreports.
# against 2 single truthful reports
# because we tile misreports, it is safe to put them 2 by 2 in batches, and compute utility on each in turn.
# if one list is shorter, we can just pad it out to the other list, I think.
def print_misreport_differences(model, truthful_bids, verbose=False, tolerance=1e-2):
    p1_misreports = torch.tensor(all_possible_misreports(truthful_bids[0, :].numpy()))
    p2_misreports = torch.tensor(all_possible_misreports(truthful_bids[1, :].numpy()))

    if p1_misreports.shape[0] > p2_misreports.shape[0]:
        to_pad = truthful_bids[1,:].repeat(p1_misreports.shape[0] - p2_misreports.shape[0], 1)
        p2_misreports = torch.cat( (p2_misreports, to_pad ))

    elif p2_misreports.shape[0] > p1_misreports.shape[0]:
        to_pad = truthful_bids[0,:].repeat(p2_misreports.shape[0] - p1_misreports.shape[0], 1)
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
parser.add_argument('--main-lr', type=float, default=1e-3, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=20, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--nbatch', type=int, default=4, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=10.0, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=10.0, help='control strength in cvxpy objective')
args = parser.parse_args()

N_HOS = 2
N_TYP = 7
hos1_probs = [0.25, 0, 0, 0.25, 0.25, 0.25, 0]
hos2_probs = [0, 0.33, 0.33, 0, 0, 0, 0.34]

hos_gen_lst = [gens.GenericTypeHospital(hos1_probs, 10),
               gens.GenericTypeHospital(hos2_probs, 10)]

generator = gens.ReportGenerator(hos_gen_lst, (N_HOS, N_TYP))
random_batches = util.create_train_sample(generator, num_batches=args.nbatch, batch_size=args.batchsize)
test_batches = util.create_train_sample(generator, num_batches=args.nbatch, batch_size=args.batchsize)

small_batch = torch.tensor([
    [[[3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
      [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]]]
])


# Loading/Creating structures matrix
internal_s = torch.tensor(np.load('type_matrix/ashlagi_7_type.npy'),
                          requires_grad=False, dtype=torch.float32)

central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
num_structures = central_s.shape[1]
int_structures = internal_s.shape[1]

# Weights matrix for central structures
internal_weight_value = 2.0

individual_weights = torch.zeros(num_structures, N_HOS)
for h in range(N_HOS):
    for col in range(num_structures):
        # Check how many allocated pairs
        allocated = central_s[h * N_TYP: (h+1) * N_TYP, col].sum().item()

        # Since only two-cycles this means it is an internal match
        if allocated > 1:
            allocated = internal_weight_value
        individual_weights[col, h] = allocated

# total value for structures
struct_weights = individual_weights.sum(dim=-1)

# Weights matrix of internal structures
internal_weights = torch.ones(int_structures) * internal_weight_value
# Make directory and save args
prefix = f'mix_match_{mn.curr_timestamp()}/'

model = MatchNet(N_HOS, N_TYP, central_s, internal_s, individual_weights,
                 internal_weights, control_strength=args.control_strength)

mn.init_train_loop(model, random_batches, main_iter=20, net_lr=1e-3)
print_misreport_differences(model, small_batch[0, 0, :, :])

# Create experiment
ashlagi_experiment = Experiment.Experiment(args, internal_s, N_HOS, N_TYP, model, dir=prefix)
ashlagi_experiment.run_experiment(random_batches, None, save=SAVE, verbose=True)

# Visualizations
print('allocations on batch ', small_batch)
allocs = model.forward(small_batch, 1) @ central_s.transpose(0, 1)
visualize_match_outcome(small_batch[0], allocs)
print(allocs.view(2, 7))

# Check regret of mix and match example
print_misreport_differences(model, small_batch[0,0,:,:], verbose=True)

# Exhaustive regret check on the test_batches
high_regret_samples = full_regret_check(model, test_batches, verbose=True)

compat_dict = {}
for t in range(N_TYP):
    compat_dict[t] = []
    if t - 1 >= 0:
        compat_dict[t].append(t-1)
    if t + 1 < N_TYP:
        compat_dict[t].append(t + 1)

# Compute MATCH_PI from mix and match
allocs_lst = []
for batch in range(test_batches.shape[0]):
    match_weights = mm.create_match_weights(central_s, test_batches[batch], compat_dict)  # [batch_size, n_structures]
    matchings = []
    for sample in range(test_batches.shape[1]):
        max_matching = mm.compute_max_matching(central_s, match_weights[sample], test_batches[batch, sample].view(N_TYP * N_HOS))
        matchings.append(torch.tensor(max_matching))
    matchings = torch.stack(matchings).type(torch.float32)
    allocs = (matchings @ central_s.T).view(-1, N_HOS, N_TYP)
    allocs_lst.append(allocs)
all_allocs = torch.stack(allocs_lst)

visualize_match_outcome(test_batches[0, 0].unsqueeze(0), all_allocs[0, 0].view(1, -1))

