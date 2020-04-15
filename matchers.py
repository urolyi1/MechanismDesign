import pickle
import time

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn as nn

from maximum_match import cvxpy_max_matching

# how large of a negative value we will tolerate without raising an exception
# this happens when the relaxed solution slightly over-allocates
# any negatives less than this will be clamped away
NEGATIVE_TOL = 5e-1


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

        self.weights_matrix = weights_matrix  # [num_structures, n_hos]
        self.internal_weights = internal_weights  # [int_structures, 1]

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


class MatchNet(Matcher):
    def __init__(self, n_hos, n_types, central_s, internal_s, weights_matrix, internal_weights, control_strength=5.0):
        # Matcher initialization to set parameters
        super(MatchNet, self).__init__(n_hos, n_types, central_s, internal_s, weights_matrix, internal_weights)
        self.control_strength = control_strength
        self.total_weights = self.weights_matrix.sum(-1)

        # Central linear program CVXPY problem definition
        x1 = cp.Variable(self.n_structures)
        w = cp.Parameter(self.n_structures)  # structure weight
        z = cp.Parameter(self.n_structures)  # control parameter
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        constraints = [x1 >= 0, self.S.numpy() @ x1 <= b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize((w.T @ x1) - self.control_strength * cp.norm(x1 - z, 1))
        problem = cp.Problem(objective, constraints)

        self.l_prog_layer = CvxpyLayer(problem, parameters=[w, b, z], variables=[x1])

        # Internal matching CVXPY problem definition
        x_int = cp.Variable(self.int_structures)
        int_w = cp.Parameter(self.int_structures)
        int_b = cp.Parameter(self.n_types)

        # constraint for positive allocation and less than true bid
        int_constraints = [x_int >= 0, self.int_S.numpy() @ x_int <= int_b]
        objective = cp.Maximize((int_w.T @ x_int))
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_w, int_b], variables=[x_int])

        # Defining the neural network
        self.neural_net = nn.Sequential(nn.Linear(self.n_h_t_combos, 128), nn.Tanh(), nn.Linear(128, 128),
                                        nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_structures))

    def save(self, filename_prefix='./'):

        torch.save(self.neural_net.state_dict(), filename_prefix+'matchnet.pytorch')
        params_dict = {
            'n_hos': self.n_hos,
            'n_types': self.n_types,
            'n_structures': self.n_structures,
            'int_structs': self.int_structures,
            'S': self.S,
            'int_S': self.int_S,
            'W': self.W,
            'internalW': self.internalW
        }
        with open(filename_prefix+'matchnet_classvariables.pickle', 'wb') as f:
            pickle.dump(params_dict, f)

    @staticmethod
    def load(filename_prefix):
        with open(filename_prefix + 'matchnet_classvariables.pickle', 'rb') as f:
            params_dict = pickle.load(f)

        result = MatchNet(
            params_dict['n_hos'],
            params_dict['n_types'],
            params_dict['n_structures'],
            params_dict['int_structs'],
            params_dict['S'],
            params_dict['int_S'],
            W=params_dict['W'],
            internalW=params_dict['internalW']
        )

        result.neural_net.load_state_dict(torch.load(filename_prefix+'matchnet.pytorch'))

        return result

    def neural_net_forward(self, X):
        """
        INPUT
        ------
        X: input [batch_size, n_hos, n_types]
        OUTPUT
        ------
        Z: output [batch_size, n_structures]
        """
        Z = X.view(-1, self.n_types * self.n_hos)

        return self.neural_net(Z)

    def linear_program_forward(self, X, z, batch_size):
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
        W = self.total_weights.unsqueeze(0).repeat(batch_size, 1)
        B = X.view(batch_size, self.n_hos * self.n_types)  # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        t0 = time.time()
        x1_out, = self.l_prog_layer(W, B, z, solver_args={'max_iters': 50000, 'verbose': False, 'scale': 5.0})
        # print(f'central match took {time.time() - t0}', flush=True)

        return x1_out

    def internal_linear_prog(self, X, batch_size):
        """
        INPUT
        ------
        X: All internal bids [batch_size * n_hos, n_types]
        batch_size: number of samples in batch (confusing name ambiguity)

        x1_out: allocation vector [batch_size * n_hos, n_structures]
        """

        W = self.internal_weights.unsqueeze(0).repeat(batch_size, 1)
        B = X.view(batch_size, self.n_types)

        # feed all parameters through cvxpy layer
        t0 = time.time()
        x1_out, = self.int_layer(W, B, solver_args={'max_iters': 50000, 'verbose': False, 'eps': 1e-3, 'scale': 5.0})
        # print(f'internal match took {time.time() - t0}')

        return x1_out

    def internal_integer_forward(self, X, batch_size):
        """
        X: [batch_size * n_hos, n_types]
        """
        w = self.internal_weights.numpy()
        x2_out = torch.zeros(batch_size, self.int_structures)
        for batch in range(batch_size):
            curr_B = X[batch].view(self.n_types).detach().numpy()
            curr_z = np.zeros_like(w)
            resulting_vals = cvxpy_max_matching(self.int_S.numpy(), w, curr_B, curr_z, 0.0)
            x2_out[batch, :] = torch.tensor(resulting_vals)
        return x2_out

    def integer_forward(self, X, batch_size):
        """
        INPUT:
        X: [batch_size, n_hos, n_types]

        OUTPUT:
        x1_out: [batch_size, n_hos, n_types]
        """
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types))  # [batch_size, n_structures]
        w = self.internal_weights.numpy()# currently weight all structures same
        x1_out = torch.zeros(batch_size, self.n_structures)
        for batch in range(batch_size):
            curr_X = X[batch].view(self.n_hos * self.n_types).detach().numpy()
            curr_z = z[batch].detach().numpy()
            resulting_vals = cvxpy_max_matching(self.S.numpy(), w, curr_X, curr_z, self.control_strength)
            x1_out[batch,:] = torch.tensor(resulting_vals)
        return x1_out

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

        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types))  # [batch_size, n_structures]

        x_1 = self.linear_program_forward(X, z, batch_size)
        return x_1


class GreedyMatcher(Matcher):
    # do we need this to be a module?

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None):

        super(GreedyMatcher, self).__init__(n_hos, n_types, num_structs, int_structs, S, int_S, W=W, internalW=internalW)

        x1 = cp.Variable(self.n_structures)
        w = cp.Parameter(self.n_structures)  # structure weight
        b = cp.Parameter(self.n_h_t_combos)  # max bid

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
        batch_size = X.shape[0]
        x_1 = self.linear_program_forward(X, batch_size)
        return x_1

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

