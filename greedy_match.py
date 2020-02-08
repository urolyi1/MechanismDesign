import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from maximum_match import cvxpy_max_matching


class GreedyMatcher(nn.Module):
    # do we need this to be a module?

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None):

        super(GreedyMatcher, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types

        self.S = S
        self.int_S = int_S

        # creating the central matching cvxypy layer
        self.n_structures = num_structs  # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos

        x1 = cp.Variable(self.n_structures)
        w = cp.Parameter(self.n_structures)  # structure weight
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        self.control_strength = 0.0

        constraints = [x1 >= 0, self.S @ x1 <= b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize((w.T @ x1))
        problem = cp.Problem(objective, constraints)

        self.l_prog_layer = CvxpyLayer(problem, parameters=[w, b], variables=[x1])

        # INTERNAL MATCHING CVXPY LAYER
        self.int_structures = int_structs

        x_int = cp.Variable(self.int_structures)
        int_w = cp.Parameter(self.int_structures)
        int_b = cp.Parameter(self.n_types)

        int_constraints = [x_int >= 0,
                           self.int_S @ x_int <= int_b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize((int_w.T @ x_int) )
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_w, int_b], variables=[x_int])

        if W is not None:
            self.W = W
        else:
            self.W = torch.ones(self.n_structures)
        if internalW is not None:
            self.internalW = internalW
        else:
            self.internalW = torch.ones(self.int_structures)


    def linear_program_forward(self, X, batch_size):
        """
        INPUT
        ------
        X: given bids [batch_size, n_hos, n_types]
        z: neural network output [batch_size, n_structures]
        batch_size: number of samples in batch

        OUTPUT
        ------
        x1_out: allocation vector [batch_size, n_structures]
        """

        # tile S matrix to accomodate batch_size
        tiled_S = self.S.view(1, self.n_h_t_combos, self.n_structures).repeat(batch_size, 1, 1)
        W = self.W.unsqueeze(0).repeat(batch_size, 1)
        B = X.view(batch_size, self.n_hos * self.n_types)  # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        x1_out, = self.l_prog_layer(W, B, solver_args={'max_iters': 50000, 'verbose': False})
        return x1_out

    def internal_linear_prog(self, X, batch_size):
        """
        INPUT
        ------
        X: All internal bids [batch_size * n_hos, n_types]
        batch_size: number of samples in batch

        x1_out: allocation vector [batch_size * n_hos, n_structures]
        """
        tiled_S = self.int_S.view(1, self.n_types, self.int_structures).repeat(batch_size, 1, 1)
        W = self.internalW.unsqueeze(0).repeat(batch_size, 1)
        B = X.view(batch_size, self.n_types)

        x_int_out, = self.int_layer(W, B, solver_args={'max_iters': 50000, 'verbose': False})

        return x_int_out

    def forward(self, X, batch_size):
        """
        Feed-forward output of network

        INPUT
        ------
        X: bids tensor [batch_size, n_hos, n_types]

        OUTPUT
        ------
        x_1: allocation vector [batch_size, n_structures]
        """

        x_1 = self.linear_program_forward(X, batch_size)
        return x_1

    def create_combined_misreport(self, curr_mis, true_rep, self_mask):
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
        only_mis = curr_mis.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * self_mask
        other_hos = true_rep.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * (1 - self_mask)
        result = only_mis + other_hos
        return result

    def integer_forward(self, X, batch_size):
        w = torch.ones(self.n_structures).numpy() # currently weight all structurs same
        x1_out = torch.zeros(batch_size, self.n_structures)
        for batch in range(batch_size):
            curr_X = X[batch].view(self.n_hos * self.n_types).detach().numpy()
            curr_z = np.zeros_like(w)
            resulting_vals = cvxpy_max_matching(self.S.numpy(), w, curr_X, curr_z, self.control_strength)
            x1_out[batch,:] = torch.tensor(resulting_vals)
        return x1_out

    def internal_integer_forward(self, X, batch_size):
        """
        X: [batch_size * n_hos, n_types]
        """
        w = torch.ones(self.int_structures).numpy()
        x2_out = torch.zeros(batch_size, self.int_structures)
        for batch in range(batch_size):
            curr_B = X[batch].view(self.n_types).detach().numpy()
            curr_z = np.zeros_like(w)
            resulting_vals = cvxpy_max_matching(self.int_S.numpy(), w, curr_B, curr_z, 0.0)
            x2_out[batch, :] = torch.tensor(resulting_vals)
        return x2_out

    def calc_mis_util(self, p, mis_alloc, S, mis_mask):
        """
        Takes misreport allocation and computes utility

        INPUT
        ------
        mis_alloc: dim [n_hos * batch_size, n_possible_cycles]
        S: matrix of possible cycles [n_hos * n_types, n_possible_cycles]
        n_hos: number of hospitals
        n_types: number of types
        mis_mask: [n_hos, 1, n_hos] all zero except in index (i, 1, i)
        internal_util: [batch_size, n_hos] utility from internal matching

        OUTPUT
        ------
        util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility from misreporting in sample 0
        """
        batch_size = int(mis_alloc.size()[0] / self.n_hos)
        alloc_counts = mis_alloc.view(self.n_hos, batch_size, -1) @ S.transpose(0,
                                                                                1)  # [n_hos, batch_size, n_hos * n_types]

        # multiply by mask to only count misreport util
        central_util = torch.sum(alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types),
                                 dim=-1) * mis_mask  # [n_hos, batch_size, n_hos]
        central_util, _ = torch.max(central_util, dim=-1, keepdim=False, out=None)
        central_util = central_util.transpose(0, 1)  # [batch_size, n_hos]

        # TODO possibly replace later if slow
        utils = []
        for i in range(self.n_hos):
            allocated = (alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types))[i, :, i, :]
            minquantity = (torch.min(p[:, i, :] - allocated)).item()
            if minquantity <= 0:
                try:
                    assert (abs(minquantity) < 1e-3)
                except:
                    print(allocated)
                    print(p[:, i, :])
                    print(p[:, i, :] - allocated)
                    print(abs(minquantity))
                    assert (abs(minquantity) < 1e-3)

            curr_hos_leftovers = (p[:, i, :] - allocated).clamp(min=0)
            curr_hos_alloc = self.internal_linear_prog(curr_hos_leftovers, curr_hos_leftovers.shape[0])
            counts = curr_hos_alloc @ torch.transpose(self.int_S, 0, 1)  # [batch_size, n_types]
            utils.append(torch.sum(counts, dim=1))
        internal_util = torch.stack(utils, dim=1)  # [batch_size, n_hos]
        return central_util + internal_util  # sum utility from central mechanism and internal matching

    def calc_util(self, alloc_vec, S):
        """
        Takes truthful allocation and computes utility

        INPUT
        ------
        alloc_vec: dim [batch_size, n_possible_cycles]
        S: matrix of possible cycles [n_hos * n_types, n_possible_cycles]

        OUTPUT
        ------
        util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility in sample 0
        """

        allocation = alloc_vec @ torch.transpose(S, 0, 1)  # [batch_size, n_hos * n_types]

        return torch.sum(allocation.view(-1, self.n_hos, self.n_types), dim=-1)


