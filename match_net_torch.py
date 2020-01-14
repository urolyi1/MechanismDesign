import torch
import torch.nn as nn
import numpy as np

class MatchNet(nn.Module):

    def __init__(self, h_layers, act_funs, n_hos, n_types):
        
        super(MatchNet, self).__init__()
        self.hid_layers = h_layers
        self.act_funs = act_funs



    def forward(self):
        raise NotImplementedError

def create_combined_misreport(curr_mis, true_rep):
    ''' 
    Tiles and combines curr misreport and true rep to create output tensor
    where only one hospital is misreporting at a time

    INPUT
    ------
    curr_mis: current misreports dim [batch_size, n_hos, n_type]
    true_rep: true report dim [batch_size, n_hos, n_type]


    OUTPUT
    ------
    combined: dim [n_hos, batch_size, n_hos, n_type]

    '''
    raise NotImplementedError

def calc_mis_util(mis_alloc):
    '''
    Takse misreport allocation and computes utility
    
    INPUT
    ------
    mis_alloc: dim [n_hos * batch_size, n_possible_cycles]

    OUTPUT
    ------
    util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility from misreporting in sample 0
    '''
    raise NotImplementedError

def calc_util(alloc):
    '''
    Takse truthful allocation and computes utility
    
    INPUT
    ------
    mis_alloc: dim [batch_size, n_possible_cycles]

    OUTPUT
    ------
    util: dim [batch_size, n_hos] where util[0, 1] would be hospital 1's utility in sample 0
    '''
    raise NotImplementedError


for c  in range(main_iter):
    p # true input by batch dim [batch size, n_hos, n_types]
    curr_mis = p.clone().detach().requires_grad_(True)

    mis_optimizer # TODO: Define misreport Optimizer 
    for i in range(inner_iter):
        mis_input = create_combined_misreport(curr_mis, p) 
        output = model.forward(mis_input.view(-1, n_hos * n_types))
        mis_util = calc_mis_util(output)
        mis_tot_util = torch.sum(mis_util, dim)
        mis_tot_util.backward()
        mis_optimizer.step()

    util = calc_util(model.forward(p))

    mis_diff = nn.functional.ReLU(util - mis_util) # [batch_size, n_hos]

    rgt = torch.mean(mis_diff, dim=0) 

    rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
    lagr_loss = torch.sum(torch.mul(rgt , lagr_mults)) # TODO: need to initialize lagrange mults
    total_loss = rgt_loss + lagr_loss - torch.mean(tf.sum(util, dim=1))

    # TODO: define total loss optimizer








