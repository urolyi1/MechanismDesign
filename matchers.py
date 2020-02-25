import pickle
import time

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from torch import nn as nn

from maximum_match import cvxpy_max_matching

# how large of a negative value we will tolerate without raising an exception
# this happens when the relaxed solution slightly overallocates
# any negatives less than this will be clamped away
NEGATIVE_TOL = 5e-1

class Matcher(nn.Module):
    """
    Superclass for greedy and NN-based matchers. Implements everything that can be shared between these two.
    """

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None):

        super(Matcher, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types

        self.S = S
        self.int_S = int_S

        # creating the central matching cvxypy layer
        self.n_structures = num_structs  # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos
        self.int_structures = int_structs
        if W is not None:
            self.W = W
        else:
            self.W = torch.ones(self.n_structures)
        if internalW is not None:
            self.internalW = internalW
        else:
            self.internalW = torch.ones(self.int_structures)

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

        allocation = alloc_vec @ torch.transpose(S, 0, 1) # [batch_size, n_hos * n_types]

        return torch.sum(allocation.view(-1, self.n_hos, self.n_types), dim=-1)

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

    def forward(self, X, batch_size):
        raise NotImplementedError

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
        alloc_counts = mis_alloc.view(self.n_hos, batch_size, -1) @ S.transpose(0, 1)  # [n_hos, batch_size, n_hos * n_types]

        # multiply by mask to only count misreport util
        central_util = torch.sum(alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types), dim=-1) * mis_mask  # [n_hos, batch_size, n_hos]
        central_util, _ = torch.max(central_util, dim=-1, keepdim=False, out=None)
        central_util = central_util.transpose(0, 1)  # [batch_size, n_hos]

        leftovers = []
        for i in range(self.n_hos):
            allocated = (alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types))[i, :, i, :] # [batch_size, n_types]
            minquantity = (torch.min(p[:, i, :] - allocated)).item()
            if minquantity <= 0:
                try:
                    assert (abs(minquantity) < NEGATIVE_TOL)
                except:
                    print(allocated)
                    print(p[:, i, :])
                    print(p[:, i, :] - allocated)
                    print(abs(minquantity))
                    assert (abs(minquantity) < NEGATIVE_TOL)
            curr_hos_leftovers = (p[:, i, :] - allocated).clamp(min=0)  # [batch_size, n_types]
            leftovers.append(curr_hos_leftovers)
        leftovers = torch.stack(leftovers, dim=1).view(-1, self.n_types)  # [batch_size * n_hos, n_types]
        internal_alloc = self.internal_linear_prog(leftovers, leftovers.shape[0])  # [batch_size * n_hos, int_structs]
        counts = internal_alloc.view(batch_size, self.n_hos, -1) @ torch.transpose(self.int_S, 0, 1)  # [batch_size, n_hos, n_types]
        internal_util = torch.sum(counts, dim=-1)

        return central_util + internal_util  # sum utility from central mechanism and internal matching


class MatchNet(Matcher):

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None, control_strength=5):

        super(MatchNet, self).__init__(n_hos, n_types, num_structs, int_structs, S, int_S, W=W, internalW=internalW)
        self.control_strength = control_strength

        x1 = cp.Variable(self.n_structures)
        w = cp.Parameter(self.n_structures)  # structure weight
        z = cp.Parameter(self.n_structures)  # control parameter
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        constraints = [x1 >= 0, self.S @ x1 <= b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (w.T @ x1) - self.control_strength * cp.norm(x1 - z, 1) )
        problem = cp.Problem(objective, constraints)

        self.l_prog_layer = CvxpyLayer(problem, parameters=[w, b, z], variables=[x1])

        x_int = cp.Variable( self.int_structures )
        int_w = cp.Parameter( self.int_structures )
        int_b = cp.Parameter( self.n_types )

        int_constraints = [x_int >= 0, self.int_S @ x_int <= int_b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (int_w.T @ x_int) )
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_w, int_b], variables=[x_int])

        self.neural_net = nn.Sequential(nn.Linear(self.n_h_t_combos, 128), nn.Tanh(), nn.Linear(128, 128),
                                        nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_structures))

    def save(self, filename_prefix='./'):

        torch.save(self.neural_net.state_dict(), filename_prefix+'matchnet.pytorch')

    #n_hos, n_types, num_structs, int_structs, S, int_S, W = None, internalW = None):
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

        W = torch.ones(batch_size, self.n_structures)  # currently weight all structurs same
        B = X.view(batch_size, self.n_hos * self.n_types) # max bids to make sure not over allocated

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

        W = torch.ones(batch_size, self.int_structures)
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
        w = torch.ones(self.int_structures).numpy()
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
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types)) # [batch_size, n_structures]
        w = torch.ones(self.n_structures).numpy() # currently weight all structurs same
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

        #batch_size = X.shape[0]
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types)) # [batch_size, n_structures]

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

