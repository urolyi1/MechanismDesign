import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class CEEI(nn.Module):

    def __init__(self, resource_constraints):
        super(CEEI, self).__init__()

        self.resource_constraints = resource_constraints
        assert len(resource_constraints.shape) == 1, "resource constraints should be 1D list of m resources"
        self.m = resource_constraints.shape[0]
        self.n = 2 # two players (for now?)

        self.x = cp.Variable(self.n) # tasks per player in solution
        self.demand_vectors = cp.Parameter( (self.n, self.m)) # demand vectors in matrix form
        self.control_strength = 0.1
        self.control_term = cp.Parameter(self.n) # control term same shape as x
        self.constraints = [(self.x @ self.demand_vectors) <= self.resource_constraints]
        self.objective = cp.Maximize(cp.sum(cp.log(self.x)) - self.control_strength*cp.norm(self.x - self.control_term, 1))
        self.problem = cp.Problem(self.objective, self.constraints)

        self.problem_layer = CvxpyLayer(self.problem, parameters=[self.demand_vectors, self.control_term], variables=[self.x])

    def forward(self, demand_vectors):
        # should take valuations from players 1 and 2 and compute how many tasks they get.
        flat_demand_vectors = demand_vectors.view(-1, demand_vectors.shape[1]*demand_vectors.shape[2])
        # compute neural network output, but make it zero for now
        nn_output = torch.zeros(demand_vectors.shape[0], self.n, requires_grad=True)

        resulting_x, = self.problem_layer(demand_vectors, nn_output)

        return resulting_x


if __name__ == '__main__':
    resource_constraints = torch.tensor([10.0,10.0])
    ceei = CEEI(resource_constraints)
    print(ceei.forward(torch.tensor([ [ [2.0,3.0], [2.0,3.0]]])))
