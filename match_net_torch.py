import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class MatchNet(nn.Module):

    def __init__(self, h_layers, act_funs, n_hos, n_types, num_structs, S):
        
        super(MatchNet, self).__init__()
        self.hid_layers = h_layers
        self.act_funs = act_funs
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
        for layer, act_f in zip(self.hid_layers, self.act_funs):
            Z = layer(Z)
            Z = act_f(Z)
        return Z

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
        '''
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types)) # [batch_size, n_structures]

        x_1 = self.linear_program_forward(X, z, batch_size)
        return x_1

def create_combined_misreport(curr_mis, true_rep, self_mask):
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
    only_mis = curr_mis.view(1, -1, n_hos, n_type).repeat(n_hos, 1, 1, 1) * self_mask 
    other_hos = true_rep.view(1, -1, n_hos, n_type).repeat(n_hos, 1, 1, 1) * (1 - self_mask)
    return only_mis + other_hos

def calc_internal_util(p, mis_x):
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
    raise NotImplementedError

def calc_mis_util(mis_alloc, S, n_hos, n_types, mis_mask, internal_util):
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
    batch_size = mis_alloc.size()[0] / n_hos
    alloc_counts = mis_alloc.view(n_hos, batch_size, -1) @ S.transpose(0, 1) # [n_hos, batch_size, n_hos * n_types]

    # multiply by mask to only count misreport util
    central_util = torch.sum(alloc_counts.view(n_hos, -1, n_hos, n_types), dim=-1) * mis_mask # [n_hos, batch_size, n_hos]
    central_util = torch.max(central_util, dim=-1).transpose(0, 1) # [batch_size, n_hos]

    return central_util + internal_util # sum utility from central mechanism and internal matching


def calc_util(alloc_vec, S, n_hos, n_types):
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
    return torch.sum(allocation.view(-1, n_hos, n_types), dim=-1)


def optimize_misreports(model, curr_mis, p, min_bids, max_bids, iterations=10, lr=1e-3):
    # not convinced this method is totally correct but sketches out what we want to do
    mis_input = create_combined_misreport(curr_mis, p)
    for i in range(iterations):
        mis_input.requires_grad_(True)
        output = model.forward(mis_input.view(-1, n_hos * n_types)) 
        model.zero_grad()
        mis_util = calc_mis_util(output)
        mis_tot_util = torch.sum(mis_util, dim)
        mis_tot_util.backward()


        with torch.no_grad(): # do we need no_grad() here?
            mis_input.data += lr * mis_input.grad
            mis_input.clamp_(min_bids, max_bids)
        mis_input.detach()

for c in range(main_iter):
    p # true input by batch dim [batch size, n_hos, n_types]
    curr_mis = p.clone().detach().requires_grad_(True)


    min_bids = None # these are values to clamp bids, i.e. must be above 0 and below true pool
    max_bids = None
    optimize_misreports(model, curr_mis, p, min_bids, max_bids)

    util = calc_util(model.forward(p))

    mis_diff = nn.functional.ReLU(util - mis_util) # [batch_size, n_hos]

    rgt = torch.mean(mis_diff, dim=0) 

    rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
    lagr_loss = torch.sum(torch.mul(rgt, lagr_mults)) # TODO: need to initialize lagrange mults
    total_loss = rgt_loss + lagr_loss - torch.mean(tf.sum(util, dim=1))

    # TODO: define total loss optimizer








