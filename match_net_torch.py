import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MatchNet(nn.Module):

    def __init__(self, h_layers, act_funs, n_hos, n_types):
        
        super(MatchNet, self).__init__()
        self.hid_layers = h_layers
        self.act_funs = act_funs



    def forward(self):
        raise NotImplementedError

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

def calc_mis_util(mis_alloc, S, n_hos, n_types, mis_mask):
    '''
    Takes misreport allocation and computes utility
    
    INPUT
    ------
    mis_alloc: dim [n_hos * batch_size, n_possible_cycles]
    S: matrix of possible cycles [n_hos * n_types, n_possible_cycles]

    OUTPUT
    ------
    util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility from misreporting in sample 0
    '''
    raise NotImplementedError

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

for c  in range(main_iter):
    p # true input by batch dim [batch size, n_hos, n_types]
    curr_mis = p.clone().detach().requires_grad_(True)


    min_bids = None # these are values to clamp bids, i.e. must be above 0 and below true pool
    max_bids = None
    optimize_misreports(model, curr_mis, p, min_bids, max_bids)

    util = calc_util(model.forward(p))

    mis_diff = nn.functional.ReLU(util - mis_util) # [batch_size, n_hos]

    rgt = torch.mean(mis_diff, dim=0) 

    rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
    lagr_loss = torch.sum(torch.mul(rgt , lagr_mults)) # TODO: need to initialize lagrange mults
    total_loss = rgt_loss + lagr_loss - torch.mean(tf.sum(util, dim=1))

    # TODO: define total loss optimizer








