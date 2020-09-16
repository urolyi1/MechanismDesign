import numpy as np
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

# Custom imports
import match_net.HospitalGenerators as gens
import match_net.match_net_torch as mn
import match_net.maximum_match as mm
import match_net.util as util
from match_net.match_net import MatchNet
from match_net.util import convert_internal_S, all_possible_misreports


def visualize_match_outcome(bids, allocation, title=None):
    hospital_results = allocation.detach().view(2,7)
    inds = np.arange(7)

    fig, axes = plt.subplots(2)
    if title:
        fig.suptitle(title, fontsize=16)
    bar_width = 0.25
    axes[0].bar(inds, bids[0,0,:].numpy(), bar_width, color='b')
    axes[0].bar(inds + bar_width, bids[0, 1, :].numpy(), bar_width, color='r')
    axes[0].set_title('Bids')

    axes[1].bar(inds, hospital_results[0, :].numpy(), bar_width, color='b')
    axes[1].bar(inds, hospital_results[1, :].numpy(), bar_width, color='r')
    axes[1].set_title('Allocation')

    plt.show()

SAVE = False
np.random.seed(500)
torch.manual_seed(500)

VALID_STRUCTURES_INDS = [1, 7, 10, 12, 16, 21]
small_batch = torch.tensor([
    [[[3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
      [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]]]
])




# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--main-lr', type=float, default=1e-1, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=20, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=64, help='batch size')
parser.add_argument('--nbatch', type=int, default=4, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=10.0, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=.1, help='control strength in cvxpy objective')
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




# Loading/Creating structures matrix
internal_s = torch.tensor(np.load('type_matrix/ashlagi_7_type.npy'),
                          requires_grad=False, dtype=torch.float32)

central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
num_structures = central_s.shape[1]
int_structures = internal_s.shape[1]

# Weights matrix for central structures
internal_weight_value = 2.1

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
                 internal_weights, control_strength=1.0)

allocs = model.forward(small_batch, 1) @ central_s.transpose(0, 1)
visualize_match_outcome(small_batch[0], allocs)
#mn.init_train_loop(model, random_batches, main_iter=20, net_lr=1e-2)
#print_misreport_differences(model, small_batch[0, 0, :, :])


# Create experiment
#ashlagi_experiment = Experiment.Experiment(args, internal_s, N_HOS, N_TYP, model, dir=prefix)
train_tuple = mn.train_loop(
    model, random_batches, net_lr=args.main_lr, main_iter=args.main_iter,
    misreport_iter=args.misreport_iter, misreport_lr=args.misreport_lr, verbose=True
)
#ashlagi_experiment.run_experiment(random_batches, None, save=SAVE, verbose=True)

# Visualizations
print('allocations on batch ', small_batch)
allocs = model.forward(small_batch, 1) @ central_s.transpose(0, 1)
#visualize_match_outcome(small_batch[0], allocs)
print(allocs.view(2, 7))

# Check regret of mix and match example
#print_misreport_differences(model, small_batch[0,0,:,:], verbose=True)

# Exhaustive regret check on the test_batches
high_regret_samples = util.full_regret_check(model, test_batches, verbose=True)

compat_dict = {}
for t in range(N_TYP):
    compat_dict[t] = []
    if t - 1 >= 0:
        compat_dict[t].append(t - 1)
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

#visualize_match_outcome(test_batches[0, 0].unsqueeze(0), all_allocs[0, 0].view(1, -1))

model_allocs = model.forward(test_batches.view(-1, 2, 7), 4*32) @ central_s.transpose(0, 1)
print("model mean util: ", model_allocs.sum(dim=-1).mean())
print("optimal mean util: ", all_allocs.sum(dim=-1).sum(dim=-1).mean())