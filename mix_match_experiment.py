import numpy as np
import torch
import argparse
import time

# Custom imports
import match_net.HospitalGenerators as gens
import match_net.match_net_torch as mn
from match_net import maximum_match as mm
import match_net.util as util
from match_net.match_net import MatchNet
from match_net.util import convert_internal_S, all_possible_misreports

SAVE = False

# Start time
start_time = time.time()

# Random Seeds
np.random.seed(0)
torch.manual_seed(0)

# Command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--main-lr', type=float, default=1e-1, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=10, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=16, help='batch size')
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

hos_gen_lst = [gens.GenericTypeHospital(HOS1_PROBS, 8),
               gens.GenericTypeHospital(HOS2_PROBS, 6)]

# Generating training and test batches
generator = gens.ReportGenerator(hos_gen_lst, (N_HOS, N_TYP))
random_batches = util.create_train_sample(generator, num_batches=args.nbatch, batch_size=args.batchsize)
test_batches = util.create_train_sample(generator, num_batches=args.nbatch, batch_size=args.batchsize)

SMALL_BATCH = torch.tensor([
    [[[3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
      [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]]]
])

# Loading/Creating structures matrix
internal_s = torch.tensor(
    np.load('type_matrix/ashlagi_7_type.npy'),
    requires_grad=False,
    dtype=torch.float32
)

central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
num_structures = central_s.shape[1]
int_structures = internal_s.shape[1]

# Weights matrix for central structures
internal_weight_value = 2.0

# Create matrix of value of each structure for each hospital
individual_weights = util.create_individual_weights(central_s, internal_weight_value, num_structures, N_HOS, N_TYP)

# total value for structures
struct_weights = individual_weights.sum(dim=-1)

# Weights matrix of internal structures
internal_weights = torch.ones(int_structures) * internal_weight_value

# Make directory and save args
prefix = f'mix_match_{mn.curr_timestamp()}/'

model = MatchNet(N_HOS, N_TYP, central_s, internal_s, individual_weights,
                 internal_weights, control_strength=args.control_strength)

allocs = model.forward(SMALL_BATCH, 1) @ central_s.transpose(0, 1)

# Create experiment
train_tuple = mn.train_loop(
    model, random_batches, net_lr=args.main_lr, lagr_lr=3.0, main_iter=args.main_iter,
    misreport_iter=args.misreport_iter, misreport_lr=args.misreport_lr,
    rho=10.0, benchmark_input=SMALL_BATCH, disable=True
)


# Visualizations
mn.print_allocs(SMALL_BATCH, model)

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

model_allocs = model.forward(
    test_batches.view(-1, N_HOS, N_TYP), args.nbatch * args.batchsize
) @ central_s.transpose(0, 1)

print("model mean util: ", model_allocs.sum(dim=-1).mean())
print("optimal mean util: ", all_allocs.sum(dim=-1).sum(dim=-1).mean())

# Start time
end_time = time.time()
hours = (end_time - start_time) // 3600
mins =  ((end_time - start_time) % 3600) // 60
secs = ((end_time - start_time) % 60)
print(f'Hours: {hours}, Minutes: {mins}, Seconds: {secs}')
