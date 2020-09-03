import numpy as np
import torch
from torch import nn as nn

# how large of a negative value we will tolerate without raising an exception
# this happens when the relaxed solution slightly over-allocates
# any negatives less than this will be clamped away
NEGATIVE_TOL = 1e-2


def valid_leftovers(minquantity, i, p, allocation):
    """
    Validates the minimum value in difference between true and allocation is
    above negative tolerance.
    """
    if minquantity <= 0:
        try:
            assert (abs(minquantity) < NEGATIVE_TOL)
        except:
            print(allocation)
            print(p[:, i, :])
            print(p[:, i, :] - allocation[:, i, :])
            print(abs(minquantity))
            assert (abs(minquantity) < NEGATIVE_TOL)


class Matcher(nn.Module):
    """
    Superclass for greedy and NN-based matchers. Implements everything that can be shared between these two.
    """

    def __init__(self, n_hos, n_types, central_s, internal_s, weights_matrix, internal_weights):
        super(Matcher, self).__init__()

        # Initializing parameters
        self.n_hos = n_hos
        self.n_types = n_types
        self.S = central_s
        self.int_S = internal_s

        # creating the central matching Cvxypy layer
        self.n_structures = central_s.shape[1]
        self.n_h_t_combos = self.n_types * self.n_hos
        self.int_structures = internal_s.shape[1]

        # Weights Matrix for structures
        self.weights_matrix = weights_matrix  # [num_structures, n_hos]
        self.internal_weights = internal_weights  # [int_structures, 1]

        # Mask matrix to only weight self and mis reports
        self.mis_mask = torch.zeros(self.n_hos, 1, self.n_hos)
        self.mis_mask[np.arange(self.n_hos), :, np.arange(self.n_hos)] = 1.0
        self.self_mask = torch.zeros(self.n_hos, 1, self.n_hos, self.n_types)
        self.self_mask[np.arange(self.n_hos), :, np.arange(self.n_hos), :] = 1.0

    def forward(self, X, batch_size):
        raise NotImplementedError

    def calc_util(self, alloc_vec, truthful):
        """
        Takes truthful allocation and computes utility
        INPUT
        ------
        alloc_vec: dim [batch_size, n_structures]
        truthful: assumed truthful bids [batch_size, n_hos, n_types]
        S: matrix of possible cycles [n_hos * n_types, n_structures]
        OUTPUT
        ------
        util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility in sample 0
        """
        # Find batch size
        batch_size = alloc_vec.shape[0]

        # Compute matrix of number of allocated pairs shape: [batch_size, n_hos , n_types]
        allocation = (alloc_vec @ torch.transpose(self.S, 0, 1)).view(-1, self.n_hos, self.n_types)
        allocation = allocation.clamp(min=0)

        # Utility received by each hospital[batch_size, n_hos]
        central_util = alloc_vec @ self.weights_matrix

        # For each hospital compute leftovers
        leftovers = []
        for i in range(self.n_hos):
            # Check that allocation does not over allocate
            minquantity = (torch.min(truthful[:, i, :] - allocation[:, i, :])).item()
            valid_leftovers(minquantity, i, truthful, allocation)

            # Compute leftovers as difference between allocated and true bid
            curr_hos_leftovers = (truthful[:, i, :] - allocation[:, i, :]).clamp(min=0)  # [batch_size, n_types]
            leftovers.append(curr_hos_leftovers)

        # Stack leftovers and run internal matchings
        leftovers = torch.stack(leftovers, dim=1).view(-1, self.n_types)  # [batch_size * n_hos, n_types]
        internal_alloc = self.internal_linear_prog(leftovers, leftovers.shape[0])  # [batch_size * n_hos, int_structs]

        # Matrix multiply with internal structure weights to get internal utility
        internal_util = (internal_alloc.view(batch_size, self.n_hos, -1) @ self.internal_weights)

        return central_util, internal_util

    def calc_mis_util(self, mis_alloc, truthful):
        """
        Takes misreport allocation and computes utility
        INPUT
        ------
        mis_alloc: dim [n_hos * batch_size, n_structures]
        S: matrix of possible cycles [n_hos * n_types, n_structures]
        n_hos: number of hospitals
        n_types: number of types
        mis_mask: [n_hos, 1, n_hos] all zero except in index (i, 0, i)
        internal_util: [batch_size, n_hos] utility from internal matching
        OUTPUT
        ------
        util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility from misreporting in sample 0
        """
        batch_size = int(mis_alloc.size()[0] / self.n_hos)

        # Compute allocations in terms of people [n_hos, batch_size, n_hos * n_types]
        alloc_counts = mis_alloc.view(self.n_hos, batch_size, -1) @ self.S.transpose(0, 1)

        # multiply by mask to only count misreport util
        raw_central_util = (mis_alloc @ self.weights_matrix)  # [n_hos, batch_size, n_hos]
        central_util = raw_central_util.view(self.n_hos, batch_size, self.n_hos) * self.mis_mask

        central_util, _ = torch.max(central_util, dim=-1, keepdim=False, out=None)
        central_util = central_util.transpose(0, 1)  # [batch_size, n_hos]

        # Computing leftovers
        leftovers = []
        for i in range(self.n_hos):
            # Grab hospital i's allocations and clamp to remove negative [batch_size, n_types]
            allocated = (alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types))[i, :, i, :]
            allocated = allocated.clamp(min=0)

            # Check if there are over allocations from central mechanism
            minquantity = (torch.min(truthful[:, i, :] - allocated)).item()
            valid_leftovers(minquantity, i, truthful, allocated)

            # Compute difference between
            curr_hos_leftovers = (truthful[:, i, :] - allocated).clamp(min=0)  # [batch_size, n_types]
            leftovers.append(curr_hos_leftovers)

        leftovers = torch.stack(leftovers, dim=1).view(-1, self.n_types)  # [batch_size * n_hos, n_types]
        internal_alloc = self.internal_linear_prog(leftovers, leftovers.shape[0])  # [batch_size * n_hos, int_structs]
        internal_util = internal_alloc @ self.internal_weights  # [batch_size * n_hos, 1]

        # sum utility from central mechanism and internal matching
        return central_util + internal_util.view(batch_size, self.n_hos)

    def create_combined_misreport(self, curr_mis, true_rep):
        """
        Tiles and combines curr misreport and true rep to create output tensor
        where only one hospital is misreporting at a time

        INPUT
        ------
        curr_mis: current misreports dim [batch_size, n_hos, n_type]
        true_rep: true report dim [batch_size, n_hos, n_type]
        self_mask: tensor of all zeros except in index [i, :, i, :]

        OUTPUT
        ------
        combined: dim [n_hos, batch_size, n_hos, n_type]

        """
        only_mis = curr_mis.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * self.self_mask
        other_hos = true_rep.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * (1 - self.self_mask)
        result = only_mis + other_hos
        return result