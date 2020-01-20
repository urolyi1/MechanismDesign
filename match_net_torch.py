import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class MatchNet(nn.Module):

    def __init__(self,  n_hos, n_types, num_structs, S):
        
        super(MatchNet, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types
        
        self.S = S

        # creating the cvxypy layer
        self.n_structures = num_structs # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos

        x1 = cp.Variable(self.n_structures)
        s = cp.Parameter( (self.n_h_t_combos, self.n_structures) ) # valid structures
        w = cp.Parameter(self.n_structures) # structure weight
        z = cp.Parameter(self.n_structures) # control parameter
        b = cp.Parameter(self.n_h_t_combos) # max bid
    
        constraints = [x1 >= 0, s @ x1 <= b]
        objective = cp.Maximize( (w.T @ x1) - cp.norm(x1 - z, 2) )
        problem = cp.Problem(objective, constraints)
        
        self.l_prog_layer = CvxpyLayer(problem, parameters = [s, w, b, z], variables=[x1])

        self.neural_net = nn.Sequential(nn.Linear(6, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 8), nn.Tanh())

    def neural_net_forward(self, X):
        '''
        INPUT
        ------
        X: input [batch_size, n_hos, n_types]
        OUTPUT
        ------
        Z: output [batch_size, n_structures]
        '''
        Z = X.view(-1, self.n_types * self.n_hos)

        return self.neural_net(Z)

    def linear_program_forward(self, X, z, batch_size):
        '''
        INPUT
        ------
        X: given bids [batch_size, n_hos, n_types]
        z: neural network output [batch_size, n_structures]
        batch_size: number of samples in batch
        
        OUTPUT
        ------
        x1_out: allocation vector [batch_size, n_structures]
        '''

        # tile S matrix to accomodate batch_size
        tiled_S = self.S.view(1, self.n_h_t_combos, self.n_structures).repeat(batch_size, 1, 1)
        W = torch.ones(batch_size, self.n_structures) # currently weight all structurs same
        B = X.view(batch_size, self.n_hos * self.n_types) # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        x1_out, = self.l_prog_layer(tiled_S, W, B, z)
        return x1_out
        
    def forward(self, X, batch_size):
        '''
        Feed-forward output of network

        INPUT
        ------
        X: bids tensor [batch_size, n_hos, n_types]

        OUTPUT
        ------
        x_1: allocation vector [batch_size, n_structures]
        '''
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types)) # [batch_size, n_structures]

        x_1 = self.linear_program_forward(X, z, batch_size)
        return x_1

    def create_combined_misreport(self, curr_mis, true_rep, self_mask):
        ''' 
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

        '''
        only_mis = curr_mis.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * self_mask 
        other_hos = true_rep.view(1, -1, self.n_hos, self.n_types).repeat(self.n_hos, 1, 1, 1) * (1 - self_mask)
        result = only_mis + other_hos
        return result.clone().detach().requires_grad_(True)

    def calc_internal_util(self, p, mis_x):
        '''
        Calculate internal utility

        INPUT
        ------
        curr_mis: [batch_size, n_hos, n_types]
        og: [batch_size, n_hos, n_types]

        OUTPUT
        ------
        internal_util: the utility from the internal matching for each hospital [batch_size, n_hos]
        '''
        return 0

    def calc_mis_util(self, mis_alloc, S, n_hos, n_types, mis_mask, internal_util):
        '''
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
        '''
        batch_size = int(mis_alloc.size()[0] / self.n_hos)
        alloc_counts = mis_alloc.view(self.n_hos, batch_size, -1) @ S.transpose(0, 1) # [n_hos, batch_size, n_hos * n_types]
        
        # multiply by mask to only count misreport util
        central_util = torch.sum(alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types), dim=-1) * mis_mask # [n_hos, batch_size, n_hos]
        central_util, _ = torch.max(central_util, dim=-1, keepdim=False, out=None)
        central_util = central_util.transpose(0, 1) # [batch_size, n_hos]
        
        return central_util + internal_util # sum utility from central mechanism and internal matching

    def calc_util(self, alloc_vec, S, n_hos, n_types):
        '''
        Takes truthful allocation and computes utility
        
        INPUT
        ------
        alloc_vec: dim [batch_size, n_possible_cycles]
        S: matrix of possible cycles [n_hos * n_types, n_possible_cycles]

        OUTPUT
        ------
        util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility in sample 0
        '''

        allocation = alloc_vec @ torch.transpose(S, 0, 1) # [batch_size, n_hos * n_types]
        return torch.sum(allocation.view(-1, self.n_hos, self.n_types), dim=-1)


def optimize_misreports(model, curr_mis, p, min_bids, max_bids, mis_mask, self_mask, batch_size, iterations=10, lr=1e-3):
    '''
    Inner optimization to find best misreports

    INPUT
    ------
    model: MatchNet object
    curr_mis: current misreports (duplicate of truthful bids)
    p: truthful bids
    min_bids: lowest amount a hospital can misreport
    max_mid: ceiling of hospital misreport
    iterations: number of iterations to optimize misreports
    lr: learning rate

    OUTPUT
    -------
    None
    '''
    # not convinced this method is totally correct but sketches out what we want to do
    mis_input = model.create_combined_misreport(curr_mis, p, self_mask)
    for i in range(iterations):
        output = model.forward(mis_input.view(-1, model.n_hos * model.n_types), batch_size * model.n_hos)
        model.zero_grad()
        mis_util = model.calc_mis_util(output, model.S, model.n_hos, model.n_types, mis_mask, 0) # FIX inputs
        mis_tot_util = torch.sum(mis_util)
        mis_tot_util.backward()
        with torch.no_grad(): # do we need no_grad() here?
            print(mis_input.requires_grad)
            mis_input.data += lr * mis_input.grad
            mis_input.clamp_(min_bids, max_bids) # Probably can't use clamp and will need to use min and max
            mis_input.grad.zero_()
    return mis_input.detach()

# parameters
N_HOS = 2
N_TYP = 3
num_structures = 8
batch_size = 10

# MASKS
self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0

mis_mask = torch.zeros(N_HOS, 1, N_HOS)
mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

main_iter = 2 # number of training iterations

single_s = torch.tensor([[1.0,1.0,0.0,0.0,0.0,0.0],
                         [0.0,0.0,0.0,1.0,1.0,0.0],
                         [1.0,0.0,0.0,0.0,1.0,0.0],
                         [0.0,1.0,0.0,1.0,0.0,0.0],
                         [0.0,1.0,1.0,0.0,0.0,0.0],
                         [0.0,0.0,0.0,1.0,1.0,0.0],
                         [0.0,1.0,0.0,0.0,0.0,1.0],
                         [0.0,0.0,1.0,0.0,1.0,0.0]], requires_grad=False).t()


rho = 1.0

# true input by batch dim [batch size, n_hos, n_types]
p = torch.tensor(np.arange(batch_size * N_HOS * N_TYP)).view(batch_size, N_HOS, N_TYP).float() 
# initializing lagrange multipliers to 1
lagr_mults = torch.ones(N_HOS) #TODO: Maybe better initilization?

min_bids = 0 # these are values to clamp bids, i.e. must be above 0 and below true pool
max_bids = 100

# Making model

model = MatchNet(N_HOS, N_TYP, num_structures, single_s)

model_optim = optim.Adam(params=model.parameters(), lr=1e-3)
# Training loop
for c in range(main_iter):
    curr_mis = p.clone().detach().requires_grad_(True)
    
    mis_input = optimize_misreports(model, curr_mis, p, min_bids, max_bids, mis_mask, self_mask, batch_size)

    model.zero_grad()
    output = model.forward(mis_input.view(-1, model.n_hos * model.n_types), batch_size * model.n_hos)
    mis_util = model.calc_mis_util(output, model.S, model.n_hos, model.n_types, mis_mask, 0)
    util = model.calc_util(model.forward(p, batch_size), single_s, N_HOS, N_TYP)

    mis_diff = nn.functional.relu(util - mis_util) # [batch_size, n_hos]

    rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

    rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
    lagr_loss = torch.sum(torch.mul(rgt, lagr_mults))
    total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(util, dim=1))
    total_loss.backward()
    model_optim.step()
