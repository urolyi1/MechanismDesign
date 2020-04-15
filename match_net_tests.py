import numpy as np
import torch

# local imports
import match_net_torch as mn
import Experiment
from matchers import MatchNet
from util import find_internal_two_cycles, convert_internal_S


""" Test for mix-and-match setting """
N_HOS = 2
N_TYP = 7

# Loading/Creating structures matricies for setting
internal_s = torch.tensor(np.load('type_matrix/ashlagi_7_type.npy'),
                          requires_grad=False, dtype=torch.float32)
central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                         requires_grad=False, dtype=torch.float32)
num_structures = central_s.shape[1]
int_structures = internal_s.shape[1]

# Weights matrix for central structures
internal_weight_value = 2.2
internal_inds = find_internal_two_cycles(central_s, N_HOS, N_TYP)

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

# Creating model
model = MatchNet(N_HOS, N_TYP, central_s, internal_s, individual_weights, internal_weights, control_strength=1.0)

# Batches for testing
example_batch = torch.tensor(
    [[[3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
      [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]]]
)

truthful_batch = torch.tensor(
    [[[3.0000, 0.0000, 0.0000, 5.0000, 5.0000, 3.0000, 0.0000],
      [0.0000, 5.0, 5.0000, 0.0000, 0.0000, 0.0000, 3.0000]]]
)

curr_mis = torch.tensor(
    [
      [[3.0000, 0.0000, 0.0000, 3.0000, 0.0000, 0.0000, 0.0000],
      [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 3.0000]],

    ]
)

# Testing calc_util
batch_size = example_batch.shape[0]
alloc_vec = model.forward(example_batch, batch_size).round()
central_util, internal_util = model.calc_util(alloc_vec, truthful_batch)
print('allocation: ', alloc_vec)
print(f'Central Utility: {central_util}, Internal Utility: {internal_util}')

if not torch.eq(central_util, torch.tensor([9.6, 9.6])).all():
    raise Exception("Central Utility Error")
if not torch.eq(internal_util, torch.tensor([4.4, 4.4])).all():
    raise Exception("Internal Utility Error")

# Testing calc_mis_util
batch_size = curr_mis.shape[0]
mis_input = model.create_combined_misreport(curr_mis, example_batch)

mis_alloc = model.forward(mis_input, batch_size * N_HOS).round()
mis_util = model.calc_mis_util(mis_alloc, example_batch)
print('allocation: ', alloc_vec)
print(f'Misreported Utility: {mis_util}')

if not torch.eq(mis_util, torch.tensor([12.6, 9.6])).all():
    raise Exception("Misreport Error")
