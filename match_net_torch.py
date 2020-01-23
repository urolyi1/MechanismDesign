import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import HospitalGenerators as gens
import diffcp

class GreedyMatcher(nn.Module):
    # do we need this to be a module?

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S):

        super(GreedyMatcher, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types

        self.S = S
        self.int_S = int_S

        # creating the central matching cvxypy layer
        self.n_structures = num_structs  # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos

        x1 = cp.Variable(self.n_structures)
        s = cp.Parameter((self.n_h_t_combos, self.n_structures))  # valid structures
        w = cp.Parameter(self.n_structures)  # structure weight
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        self.control_strength = 10.0

        constraints = [x1 >= 0, s @ x1 <= b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize((w.T @ x1))
        problem = cp.Problem(objective, constraints)

        self.l_prog_layer = CvxpyLayer(problem, parameters=[s, w, b], variables=[x1])

        # INTERNAL MATCHING CVXPY LAYER
        self.int_structures = int_structs

        x_int = cp.Variable(self.int_structures)
        int_s = cp.Parameter((self.n_types, self.int_structures))
        int_w = cp.Parameter(self.int_structures)
        int_b = cp.Parameter(self.n_types)

        int_constraints = [x_int >= 0,
                           int_s @ x_int <= int_b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize((int_w.T @ x_int))
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_s, int_w, int_b], variables=[x_int])


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
        W = torch.ones(batch_size, self.n_structures)  # currently weight all structurs same
        B = X.view(batch_size, self.n_hos * self.n_types)  # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        x1_out, = self.l_prog_layer(tiled_S, W, B, solver_args={'max_iters': 50000})
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
        W = torch.ones(batch_size, self.int_structures)
        B = X.view(batch_size, self.n_types)

        x_int_out, = self.int_layer(tiled_S, W, B, solver_args={'max_iters': 50000})

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


class MatchNet(nn.Module):

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S):
        
        super(MatchNet, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types
        
        self.S = S
        self.int_S = int_S

        # creating the central matching cvxypy layer
        self.n_structures = num_structs  # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos

        x1 = cp.Variable(self.n_structures)
        s = cp.Parameter( (self.n_h_t_combos, self.n_structures) ) # valid structures
        w = cp.Parameter(self.n_structures)  # structure weight
        z = cp.Parameter(self.n_structures)  # control parameter
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        self.control_strength = 10.0
    
        constraints = [x1 >= 0, s @ x1 <= b] # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (w.T @ x1) - self.control_strength*cp.norm(x1 - z, 2) )
        problem = cp.Problem(objective, constraints)
        
        self.l_prog_layer = CvxpyLayer(problem, parameters=[s, w, b, z], variables=[x1])

        # INTERNAL MATCHING CVXPY LAYER
        self.int_structures = int_structs

        x_int = cp.Variable( self.int_structures )
        int_s = cp.Parameter( (self.n_types, self.int_structures) )
        int_w = cp.Parameter( self.int_structures )
        int_b = cp.Parameter( self.n_types )

        int_constraints = [x_int >= 0, int_s @ x_int <= int_b ] # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (int_w.T @ x_int) )
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_s, int_w, int_b], variables=[x_int])

        self.neural_net = nn.Sequential(nn.Linear(self.n_h_t_combos, 20), nn.Tanh(), nn.Linear(20, 20),
                                        nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, self.n_structures))

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

        # tile S matrix to accommodate batch_size
        tiled_S = self.S.view(1, self.n_h_t_combos, self.n_structures).repeat(batch_size, 1, 1)
        W = torch.ones(batch_size, self.n_structures) # currently weight all structurs same
        B = X.view(batch_size, self.n_hos * self.n_types) # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        x1_out, = self.l_prog_layer(tiled_S, W, B, z, solver_args={'max_iters': 50000})
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
        W = torch.ones(batch_size, self.int_structures)
        B = X.view(batch_size, self.n_types)

        x_int_out, = self.int_layer(tiled_S, W, B, solver_args={'max_iters': 50000})

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
        z = self.neural_net_forward(X.view(-1, self.n_hos * self.n_types)) # [batch_size, n_structures]

        x_1 = self.linear_program_forward(X, z, batch_size)
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

        # TODO possibly replace later if slow
        utils = []
        for i in range(self.n_hos):
            allocated = (alloc_counts.view(self.n_hos, -1, self.n_hos, self.n_types))[i, :, i, :]
            minquantity = (torch.min(p[:,i,:] - allocated)).item()
            if minquantity <= 0:
                try:
                    assert(abs(minquantity) < 1e-3)
                except:
                    print(allocated)
                    print(p[:, i, :])
                    print(p[:,i,:] - allocated)
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

        allocation = alloc_vec @ torch.transpose(S, 0, 1) # [batch_size, n_hos * n_types]

        return torch.sum(allocation.view(-1, self.n_hos, self.n_types), dim=-1)


def optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=10, lr=1e-1):
    """
    Inner optimization to find best misreports

    INPUT
    ------
    model: MatchNet object
    curr_mis: current misreports (duplicate of truthful bids) [batch_size, n_hos, n_types]
    p: truthful bids [batch_size, n_hos, n_types]
    min_bids: lowest amount a hospital can misreport
    max_mid: ceiling of hospital misreport
    iterations: number of iterations to optimize misreports
    lr: learning rate

    OUTPUT
    -------
    curr_mis: current best misreport for each hospital when others report truthfully [batch_size, n_hos, n_types]
    """
    # not convinced this method is totally correct but sketches out what we want to do
    for i in range(iterations):
        # tile current best misreports into valid inputs
        mis_input = model.create_combined_misreport(curr_mis, p, self_mask)
        
        model.zero_grad()

        # push tiled misreports through network
        output = model.forward(mis_input.view(-1, model.n_hos * model.n_types), batch_size * model.n_hos)

        # calculate utility from output only weighting utility from misreporting hospital
        mis_util = model.calc_mis_util(p, output, model.S, mis_mask)  # FIX inputs
        mis_tot_util = torch.sum(mis_util)
        mis_tot_util.backward()

        # Gradient descent
        with torch.no_grad():
            curr_mis = curr_mis + lr * curr_mis.grad
            curr_mis = torch.max( torch.min(curr_mis, p), torch.zeros_like(curr_mis) ) # clamping misreports to be valid
        curr_mis.requires_grad_(True)
    #print(torch.sum(torch.abs(orig_mis_input - mis_input)))
    return curr_mis.detach()

def greedy_experiment():
    print('estimating regret for greedy match...')
    N_HOS = 2
    N_TYP = 3
    num_structures = 8
    batch_size = 10
    # MASKS
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0
    main_iter = 30  # number of training iterations
    # Large compatibility matrix [n_hos_pair_combos, n_structures]
    single_s = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], requires_grad=False).t()
    # Internal compatbility matrix [n_types, n_int_structures]
    internal_s = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], requires_grad=False).t()
    # regret quadratic term weight
    # true input by batch dim [batch size, n_hos, n_types]
    p = torch.tensor(np.arange(batch_size * N_HOS * N_TYP)).view(batch_size, N_HOS, N_TYP).float()
    # initializing lagrange multipliers to 1
    # Making model
    model = GreedyMatcher(N_HOS, N_TYP, num_structures, 2, single_s, internal_s)
    tot_loss_lst = []
    rgt_loss_lst = []
    # Training loop
    curr_mis = p.clone().detach().requires_grad_(True)

    curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=5)

    mis_input = model.create_combined_misreport(curr_mis, p, self_mask)

    output = model.forward(mis_input, batch_size * model.n_hos)
    mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
    util = model.calc_util(model.forward(p, batch_size), single_s)

    mis_diff = (mis_util - util)  # [batch_size, n_hos]

    rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

    # computes losses
    print('mean regret', torch.mean(rgt).item())
    print('mean util', torch.mean(torch.sum(util, dim=1)).item())



def basic_matchnet_experiment():
    N_HOS = 2
    N_TYP = 3
    num_structures = 8
    batch_size = 10
    # MASKS
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0
    main_iter = 30  # number of training iterations
    # Large compatibility matrix [n_hos_pair_combos, n_structures]
    single_s = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], requires_grad=False).t()
    # Internal compatbility matrix [n_types, n_int_structures]
    internal_s = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], requires_grad=False).t()
    # regret quadratic term weight
    rho = 1.0
    # true input by batch dim [batch size, n_hos, n_types]
    p = torch.tensor(np.arange(batch_size * N_HOS * N_TYP)).view(batch_size, N_HOS, N_TYP).float()
    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS)  # TODO: Maybe better initilization?
    # Making model
    model = MatchNet(N_HOS, N_TYP, num_structures, 2, single_s, internal_s)
    model_optim = optim.Adam(params=model.parameters(), lr=1e-1)
    lagr_optim = optim.Adam(params=[lagr_mults], lr=1e-2)
    tot_loss_lst = []
    rgt_loss_lst = []
    # Training loop
    for c in range(main_iter):
        curr_mis = p.clone().detach().requires_grad_(True)

        curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=5)

        mis_input = model.create_combined_misreport(curr_mis, p, self_mask)

        output = model.forward(mis_input, batch_size * model.n_hos)
        mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
        util = model.calc_util(model.forward(p, batch_size), single_s)

        mis_diff = (mis_util - util)  # [batch_size, n_hos]

        rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

        # computes losses
        rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
        lagr_loss = torch.sum(torch.mul(rgt, lagr_mults))
        total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(util, dim=1))

        tot_loss_lst.append(total_loss.item())
        rgt_loss_lst.append(rgt_loss.item())

        print('total loss', total_loss.item())
        print('rgt_loss', rgt_loss.item())
        print('mean regret', torch.mean(rgt).item())
        print('lagr_loss', lagr_loss.item())

        if c % 5 == 0:
            lagr_optim.zero_grad()
            (-lagr_loss).backward(retain_graph=True)
            lagr_optim.step()

        model_optim.zero_grad()
        total_loss.backward()
        model_optim.step()
    print(tot_loss_lst)
    print(rgt_loss_lst)

def two_two_experiment():
    lower_lst = [[10, 20], [30, 60]]
    upper_lst = [[20, 40], [50, 100]]

    generator = gens.create_simple_generator(lower_lst, upper_lst, 2, 2)
    # parameters
    N_HOS = 2
    N_TYP = 2
    num_structures = 4
    int_structues = 1
    batch_size = 10

    # MASKS
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0

    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    main_iter = 50  # number of training iterations

    # Large compatibility matrix [n_hos_pair_combos, n_structures]
    single_s = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                             [1.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0, 1.0],
                             [0.0, 1.0, 0.0, 1.0]], requires_grad=False)

    # Internal compatbility matrix [n_types, n_int_structures]
    internal_s = torch.tensor([[1.0],
                               [1.0]], requires_grad=False)

    # regret quadratic term weight
    rho = 1.0

    # true input by batch dim [batch size, n_hos, n_types]
    # p = torch.tensor(np.arange(batch_size * N_HOS * N_TYP)).view(batch_size, N_HOS, N_TYP).float()

    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS)  # TODO: Maybe better initilization?

    # Making model
    model = MatchNet(N_HOS, N_TYP, num_structures, int_structues, single_s, internal_s)

    model_optim = optim.Adam(params=model.parameters(), lr=1e-1)
    lagr_optim = optim.Adam(params=[lagr_mults], lr=1e-2)

    tot_loss_lst = []
    rgt_loss_lst = []
    # Training loop
    for c in range(main_iter):
        p = torch.tensor(next(generator.generate_report(10))).float()
        curr_mis = p.clone().detach().requires_grad_(True)

        curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=50, lr=1.0)

        mis_input = model.create_combined_misreport(curr_mis, p, self_mask)

        output = model.forward(mis_input, batch_size * model.n_hos)
        mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
        util = model.calc_util(model.forward(p, batch_size), single_s)

        mis_diff = (mis_util - util)  # [batch_size, n_hos]

        rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

        # computes losses
        rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
        lagr_loss = torch.sum(torch.mul(rgt, lagr_mults))
        total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(util, dim=1))

        tot_loss_lst.append(total_loss.item())
        rgt_loss_lst.append(rgt_loss.item())

        print('total loss', total_loss.item())
        print('rgt_loss', rgt_loss.item())
        print('lagr_loss', lagr_loss.item())

        if c % 5 == 0:
            lagr_optim.zero_grad()
            (-lagr_loss).backward(retain_graph=True)
            lagr_optim.step()

        model_optim.zero_grad()
        total_loss.backward()
        model_optim.step()

    print(tot_loss_lst)
    print(rgt_loss_lst)

    # Actually look at the allocations to see if they make sense
    print((model.forward(p, batch_size) @ single_s.transpose(0, 1)).view(10, 2, 2))
    print(p)



# parameters
if __name__ == '__main__':
    greedy_experiment()
    basic_matchnet_experiment()