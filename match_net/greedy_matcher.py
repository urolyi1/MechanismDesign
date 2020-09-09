import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from match_net.matchers import Matcher
from match_net.maximum_match import cvxpy_max_matching


class GreedyMatcher(Matcher):
    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None):

        super(GreedyMatcher, self).__init__(
            n_hos, n_types, num_structs, int_structs, S, int_S, W=W, internalW=internalW
        )

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

