import time
import pickle
import numpy as np
import torch
from torch import nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from match_net.matchers import Matcher
from match_net.maximum_match import cvxpy_max_matching

class Misreporter(nn.Module):
    def __init__(self, n_hos, n_types, layer_size=64, n_layers=2):
        super(Misreporter, self).__init__()

        self.n_hos = n_hos
        self.n_types = n_types

        self.net = nn.Sequential(
            nn.Linear(self.n_hos*self.n_types, layer_size),
            *[nn.Linear(layer_size, layer_size) for _ in range(n_layers)],
            nn.Linear(layer_size, self.n_hos*self.n_types),
            nn.Sigmoid()
        )

    def forward(self, x):
        truthful_bids = x
        shape = x.shape
        x = torch.flatten(x, start_dim=-2)
        x = self.net(x)
        x = x.view(shape)*truthful_bids # always make it a fraction of truthful
        return x

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
        objective = cp.Maximize((w + z).T @ x1 - self.control_strength * cp.norm(x1, 2))
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
            'weights_matrix': self.weights_matrix,
            'internal_weights': self.internal_weights,
            'control_strength': self.control_strength
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
            params_dict['S'],
            params_dict['int_S'],
            params_dict['weights_matrix'],
            params_dict['internal_weights'],
            params_dict['control_strength']
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
        w = self.internal_weights.numpy()  # currently weight all structures same
        x1_out = torch.zeros(batch_size, self.n_structures)
        for batch in range(batch_size):
            curr_X = X[batch].view(self.n_hos * self.n_types).detach().numpy()
            curr_z = z[batch].detach().numpy()
            resulting_vals = cvxpy_max_matching(self.S.numpy(), w, curr_X, curr_z, self.control_strength)
            x1_out[batch, :] = torch.tensor(resulting_vals)
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