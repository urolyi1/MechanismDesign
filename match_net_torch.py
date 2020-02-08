import itertools
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
import argparse
from cvxpylayers.torch import CvxpyLayer
import HospitalGenerators as gens
from datetime import datetime
import diffcp
from tqdm import tqdm as tqdm
import time
import os
import matplotlib.pyplot as plt
import maximum_match as max_match
from greedy_match import GreedyMatcher
from maximum_match import cvxpy_max_matching

# ensures problem is not unbounded
# no matching should ever include more than MAX_STRUCTURES of any structure
MAX_STRUCTURES = 1000

# how large of a negative value we will tolerate without raising an exception
# this happens when the relaxed solution slightly overallocates
# any negatives less than this will be clamped away
NEGATIVE_TOL = 5e-1


def curr_timestamp():
    return datetime.strftime(datetime.now(), format='%Y-%m-%d_%H-%M-%S')

class MatchNet(nn.Module):

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None, control_strength=5):
        
        super(MatchNet, self).__init__()
        self.n_hos = n_hos
        self.n_types = n_types
        
        self.S = S
        self.int_S = int_S

        # Creating the central matching cvxypy layer
        self.n_structures = num_structs  # TODO: figure out how many cycles
        self.n_h_t_combos = self.n_types * self.n_hos

        x1 = cp.Variable(self.n_structures)
        s = cp.Parameter( (self.n_h_t_combos, self.n_structures) )  # valid structures
        w = cp.Parameter(self.n_structures)  # structure weight
        z = cp.Parameter(self.n_structures)  # control parameter
        b = cp.Parameter(self.n_h_t_combos)  # max bid

        self.control_strength = control_strength
    
        constraints = [x1 >= 0, self.S @ x1 <= b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (w.T @ x1) - self.control_strength * cp.norm(x1 - z, 1) )
        problem = cp.Problem(objective, constraints)
        
        self.l_prog_layer = CvxpyLayer(problem, parameters=[w, b, z], variables=[x1])

        # INTERNAL MATCHING CVXPY LAYER
        self.int_structures = int_structs

        x_int = cp.Variable( self.int_structures )
        int_s = cp.Parameter( (self.n_types, self.int_structures) )
        int_w = cp.Parameter( self.int_structures )
        int_b = cp.Parameter( self.n_types )

        int_constraints = [x_int >= 0, self.int_S @ x_int <= int_b]  # constraint for positive allocation and less than true bid
        objective = cp.Maximize( (int_w.T @ x_int) )
        problem = cp.Problem(objective, int_constraints)

        self.int_layer = CvxpyLayer(problem, parameters=[int_w, int_b], variables=[x_int])

        self.neural_net = nn.Sequential(nn.Linear(self.n_h_t_combos, 128), nn.Tanh(), nn.Linear(128, 128),
                                        nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_structures))
        if W is not None:
            self.W = W
        else:
            self.W = torch.ones(self.n_structures)
        if internalW is not None:
            self.internalW = internalW
        else:
            self.internalW = torch.ones(self.int_structures)

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

        # tile S matrix to accommodate batch_size
        tiled_S = self.S.view(1, self.n_h_t_combos, self.n_structures).repeat(batch_size, 1, 1)
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
        #print(torch.norm(curr_mis.grad))

        # Gradient descent
        with torch.no_grad():
            curr_mis = curr_mis + lr * curr_mis.grad
            curr_mis = torch.max( torch.min(curr_mis, p), torch.zeros_like(curr_mis) ) # clamping misreports to be valid
        curr_mis.requires_grad_(True)
    #print(torch.sum(torch.abs(orig_mis_input - mis_input)))
    return curr_mis.detach()


def create_train_sample(generator, num_batches, batch_size=16):
    """
    Generate num_batches batches and stack them into a single tensor

    :param generator: hospital true bid generator
    :param num_batches: number of batches to generate
    :return: tensor of batches [num_batches, batch_size, n_hos, n_types]
    """
    batches = []
    for i in range(num_batches):
        batches.append(torch.tensor(next(generator.generate_report(batch_size))).float())
    return torch.stack(batches, dim=0)


def blow_up_column(col_vector, num_hospitals):
    """
    Takes a single structure vector and blows it up to all possible version of that structure specifying hospital

    :param col_vector: vector of a possible structure within the S matrix
    :param num_hospitals: number of hospitals to blow up column to
    :return: tensor of all possible version of structure with specific hospitals
    """
    # find which pairs are in the structure
    nonzero_inds, = np.nonzero(col_vector)
    all_inds = []
    for ind in nonzero_inds:
        count = int(col_vector[ind])
        all_inds.extend(count*[ind])
    num_types = len(col_vector)
    all_hospitals = list(range(num_hospitals))
    num_outcomes = len(all_inds)
    new_columns = []
    for hospital_outcome in itertools.product(all_hospitals, repeat=num_outcomes):
        new_column = np.zeros(num_types * num_hospitals)
        for i, ind in enumerate(all_inds):
            curr_ind_hospital = hospital_outcome[i]
            new_column[(curr_ind_hospital * num_types) + ind] += 1.0
        new_columns.append(new_column)
    return np.stack(new_columns).transpose()


def convert_internal_S(single_S, num_hospitals):
    """
    Blowup a single S matrix with all possible structures between just pairs to be all possible structures
    from hospital and pair.

    :param single_S: structures matrix between just pairs
    :param num_hospitals: number of hospitals in exchange
    :return: Large structure matrix with all possible structures accounting for hospital and type
    """
    all_cols = []
    # Blow up all the columns in single structure matrix
    for col in range(single_S.shape[1]):
        all_cols.append(blow_up_column(single_S[:, col], num_hospitals))
    return np.concatenate(all_cols, axis=1)


def internal_central_bloodtypes(num_hospitals):
    """
    :param num_hospitals: number of hospitals involved
    :return: tuple of internal structure matrix and full structure matrix

    "O-O","O-B","O-AB","O-A","B-O","B-B","B-AB","B-A","AB-O","AB-B","AB-AB","AB-A","A-O","A-B","A-AB","A-A"

    """
    internal_s = np.load('bloodtypematrix.npy')
    central_s = convert_internal_S(internal_s, num_hospitals)
    return torch.tensor(internal_s, dtype=torch.float32, requires_grad=False), torch.tensor(central_s, dtype=torch.float32, requires_grad=False)

def benchmark_example():
    """
    Runs model on simple example for benchmark and profiling purposes
    :param args:
    :return:
    """
    lower_lst = [[10, 20], [30, 60]]
    upper_lst = [[20, 40], [50, 100]]
    generator = gens.create_simple_generator(lower_lst, upper_lst, 2, 2)
    batches = create_train_sample(generator, 2, batch_size=3)

    N_HOS = 2
    N_TYP = 2
    num_structures = 4
    int_structures = 1
    batch_size = batches.shape[1]

    # single structure matrix
    internal_s = torch.tensor([[1.0],
                               [1.0]], requires_grad=False)
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), 2), requires_grad=False, dtype=torch.float32)

    model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s,
                     control_strength=3.0)
    # initial_train_loop(batches, model, batch_size, central_s, init_iter=args.init_iter, net_lr=args.main_lr)
    final_p, rgt_loss_lst, tot_loss_lst, util_loss_lst = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                                                                    main_iter=3,
                                                                    net_lr=1e-1,
                                                                    misreport_iter=10,
                                                                    misreport_lr=5.0)




def ashlagi_7_type_single(args):
    N_HOS = 2
    N_TYP = 7

    batches = torch.tensor([
        [
            [[10.0,0.0,0.0,10.0,10.0,10.0,0.0],
             [0.0,10.0,10.0,0.0,0.0,0.0,10.0]]
        ]
    ])
    prefix = f'ashlagi_7_type{curr_timestamp()}/'
    os.mkdir(prefix)
    print(vars(args))
    with open(prefix + 'argfile.json', 'w') as f:
        json.dump(vars(args), f)
    internal_s = torch.tensor(np.load('ashlagi_7_type.npy'),
                              requires_grad=False, dtype=torch.float32)
    int_structures = internal_s.shape[1]
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                             requires_grad=False, dtype=torch.float32)
    num_structures = central_s.shape[1]
    batch_size = batches.shape[1]
    model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s,
                     control_strength=args.control_strength)

    greedy_matcher = GreedyMatcher(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s)

    greedy_regrets = test_model_performance(batches, greedy_matcher, batch_size, greedy_matcher.S, N_HOS, N_TYP)
    print('regrets on greedy match after 100 iters', greedy_regrets)
    opt_util_mean = 0.0
    for batch in range(batches.shape[0]):
        # create mix and match weights
        #weights_batch = max_match.create_match_weights(central_s, batches[batch, :])
        for inst in range(batches.shape[1]):
            optimal_matching = cvxpy_max_matching(central_s.numpy(),
                                                  torch.ones(central_s.shape[1]).numpy(),
                                                  batches[batch,inst,:].view(N_HOS * N_TYP).numpy(),
                                                  torch.zeros(central_s.shape[1]).numpy(), 0)
            opt_match_util = torch.sum(central_s @ optimal_matching).item()
            opt_util_mean += opt_match_util / (batches.shape[0] * batches.shape[1])
            print('max matching value', opt_match_util)
            #mix_and_matching = cvxpy_max_matching(central_s.numpy(), weights_batch[inst, :],
            #                                            batches[batch, inst, :].view(N_HOS * N_TYP).numpy(),
            #                                            torch.zeros(central_s.shape[1]).numpy(), 0)
            #print('mix and match value', torch.sum(central_s @ optimal_matching).item())
    print('max matching mean util', opt_util_mean)
    train_tuple = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                             main_iter=args.main_iter,
                             net_lr=args.main_lr,
                             misreport_iter=args.misreport_iter,
                             misreport_lr=args.misreport_lr)

    print('greedy --- internal util: {}; central util: {}; int internal util: {}; int central util: {}'.format(*compare_central_internal_utils(batches, greedy_matcher)))
    print('learned --- internal util: {}; central util: {}; int internal util: {}; int central util: {}'.format(*compare_central_internal_utils(batches, model)))
    #save_experiment(prefix, train_tuple, args, model, batches, test_batches, test_mis_iter=50)

def ashlagi_7_type_experiment(args):
    N_HOS = 2
    N_TYP = 7
    hos1_probs = [0.25, 0, 0, 0.25, 0.25, 0.25, 0]
    hos2_probs = [0, 0.33, 0.33, 0, 0, 0, 0.34]
    hos_gen_lst = [gens.AshlagiHospital(hos1_probs, 1),
                   gens.AshlagiHospital(hos2_probs, 1)]

    generator = gens.ReportGenerator(hos_gen_lst, (N_HOS, N_TYP))
    batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
    test_batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
    
    ashlagi_compat_dict = {}
    for i in range(1, N_TYP - 1):
        ashlagi_compat_dict[i] = []
        ashlagi_compat_dict[i].append(i - 1)
        ashlagi_compat_dict[i].append(i + 1)
    ashlagi_compat_dict[0] = [1]
    ashlagi_compat_dict[N_TYP - 1] = [N_TYP - 2]

    # Make directory and save args
    prefix = f'ashlagi_7_type{curr_timestamp()}/'
    os.mkdir(prefix)
    print(vars(args))
    with open(prefix + 'argfile.json', 'w') as f:
        json.dump(vars(args), f)
    internal_s = torch.tensor(np.load('ashlagi_7_type.npy'),
                                      requires_grad=False, dtype=torch.float32)
    int_structures = internal_s.shape[1]
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                             requires_grad=False, dtype=torch.float32)
    num_structures = central_s.shape[1]
    batch_size = batches.shape[1]
    model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s,
                     control_strength=args.control_strength)
    opt_util_mean = 0.0
    mix_util_mean = 0.0
    print(batches[0, 0, :])
    for batch in range(batches.shape[0]):
        # create mix and match weights
        weights_batch = max_match.create_match_weights(central_s, batches[batch, :], ashlagi_compat_dict)
        for inst in range(batches.shape[1]):
            weights = weights_batch[inst, :]
            optimal_matching = cvxpy_max_matching(central_s.numpy(),
                                                  torch.ones(central_s.shape[1]).numpy(),
                                                  batches[batch,inst,:].view(N_HOS * N_TYP).numpy(),
                                                  torch.zeros(central_s.shape[1]).numpy(), 0)
            opt_match_util = torch.sum(central_s @ optimal_matching).item()
            opt_util_mean += opt_match_util / (batches.shape[0] * batches.shape[1])
            print('max matching value', opt_match_util)
            mix_and_matching = cvxpy_max_matching(central_s.numpy(), weights,
                                                        batches[batch, inst, :].view(N_HOS * N_TYP).numpy(),
                                                        torch.zeros(central_s.shape[1]).numpy(), 0)
            mix_match_util = torch.sum(central_s @ mix_and_matching).item()
            mix_util_mean += mix_match_util / (batches.shape[0] * batches.shape[1])
            print('mix and match value', mix_match_util)
    print('max matching mean util', opt_util_mean)
    print('mix and match matching mean util', mix_util_mean)
    train_tuple = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                             main_iter=args.main_iter,
                             net_lr=args.main_lr,
                             misreport_iter=args.misreport_iter,
                             misreport_lr=args.misreport_lr)

    greedy_matcher = GreedyMatcher(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s)
    greedy_utils = compare_central_internal_utils(batches, greedy_matcher)
    learned_utils = compare_central_internal_utils(batches, model)
    print('greedy util difference', greedy_utils[1] - greedy_utils[0])
    print('greedy int util difference', greedy_utils[3] - greedy_utils[2])
    print('learned util difference', learned_utils[1] - learned_utils[0])
    print('learned int util difference', learned_utils[3] - learned_utils[2])
    save_experiment(prefix, train_tuple, args, model, batches, test_batches, test_mis_iter=50)

def realistic_experiment(args):
    """
    Runs model on two hospitals with real blood types

    :param args:
    """
    hos_gen_lst = [gens.RealisticHospital(3), gens.RealisticHospital(3)]
    generator = gens.ReportGenerator(hos_gen_lst, (2, 16))
    batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
    test_batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)

    # Make directory and save args
    prefix = f'real_two_test{curr_timestamp()}/'
    os.mkdir(prefix)
    print(vars(args))
    with open(prefix + 'argfile.json', 'w') as f:
        json.dump(vars(args), f)

    N_HOS = 2
    N_TYP = 16

    internal_s = torch.tensor(np.load('two_cycle_bloodtype_matrix.npy'),
                                      requires_grad=False, dtype=torch.float32)
    int_structures = internal_s.shape[1]
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), N_HOS),
                             requires_grad=False, dtype=torch.float32)
    num_structures = central_s.shape[1]
    batch_size = batches.shape[1]
    model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s,
                     control_strength=args.control_strength)
    for batch in range(batches.shape[0]):
        # create mix and match weights
        weights_batch = max_match.create_match_weights(central_s, batches[batch, :], None, max_match.calc_edges_AB)
        for inst in range(batches.shape[1]):
            optimal_matching = cvxpy_max_matching(central_s.numpy(),
                                                  torch.ones(central_s.shape[1]).numpy(),
                                                  batches[batch,inst,:].view(N_HOS * N_TYP).numpy(),
                                                  torch.zeros(central_s.shape[1]).numpy(), 0)

            print('max matching value', torch.sum(central_s @ optimal_matching).item())
            mix_and_matching = cvxpy_max_matching(central_s.numpy(), weights_batch[inst, :],
                                                        batches[batch, inst, :].view(N_HOS * N_TYP).numpy(),
                                                        torch.zeros(central_s.shape[1]).numpy(), 0)
            print('mix and match value', torch.sum(central_s @ optimal_matching).item())
    
    train_tuple = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                             main_iter=args.main_iter,
                             net_lr=args.main_lr,
                             misreport_iter=args.misreport_iter,
                             misreport_lr=args.misreport_lr)
    save_experiment(prefix, train_tuple, args, model, batches, test_batches, test_mis_iter=50)

def two_two_experiment(args):
    """
    Runs model on two hospital two type case

    :param args: Model parameters
    """

    # lower and upper bound for hospital bid generator
    lower_lst = [[10, 20], [30, 60]]
    upper_lst = [[20, 40], [50, 100]]

    # Create generator and batches
    generator = gens.create_simple_generator(lower_lst, upper_lst, 2, 2)
    batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
    test_batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)


    # Make directory and save args
    prefix = f'two_two_test{curr_timestamp()}/'
    os.mkdir(prefix)
    print(vars(args))
    with open(prefix+'argfile.json', 'w') as f:
        json.dump(vars(args), f)


    # parameters
    N_HOS = 2
    N_TYP = 2
    num_structures = 4
    int_structures = 1
    batch_size = batches.shape[1]

    # single structure matrix
    internal_s = torch.tensor([[1.0],
                               [1.0]], requires_grad=False)
    np.save(prefix+'internal_s.npy', internal_s.numpy())  # save structure matrix
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), 2), requires_grad=False, dtype=torch.float32)

    # Create the model and train using hyperparameters from args
    model = MatchNet(N_HOS, N_TYP, num_structures, int_structures, central_s, internal_s, control_strength=args.control_strength)
    train_tuple = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                                                     main_iter=args.main_iter,
                                                     net_lr=args.main_lr,
                                                     misreport_iter=args.misreport_iter,
                                                     misreport_lr=args.misreport_lr)
    save_experiment(prefix, train_tuple, args, model, batches, test_batches)


def save_experiment(prefix, train_tuple, args, model, batches, test_batches, test_mis_iter=50):
    """
    Saves results of experiment
    :param prefix: directory prefix with /
    :param train_tuple: tuple from train_loop()
    :param args: args passed into experiment
    :param model: model that was trained
    :param batches: training batches
    :param test_batches: test_batches
    :param test_mis_iter: number of iterations in the misreport optimization for the test batches

    :return: None
    """
    batch_size = batches.shape[1]

    final_p, rgt_loss_lst, tot_loss_lst, util_loss_lst = train_tuple
    np.save(prefix + 'util_loss.npy', util_loss_lst)
    np.save(prefix + 'rgt_loss.npy', rgt_loss_lst)
    np.save(prefix + 'tot_loss.npy', tot_loss_lst)

    # Actually look at the allocations to see if they make sense
    # print((model.forward(final_p[0], batch_size) @ central_s.transpose(0, 1)).view(batch_size, 2, 2))
    # print(final_p[0])

    # Save model and results on train/test batches
    model.save(filename_prefix=prefix)
    np.save(prefix + 'train_batches.npy', batches.numpy())

    np.save(prefix + 'test_batches.npy', test_batches.numpy())

    final_train_regrets = test_model_performance(batches, model,
                                                 batch_size, model.S, model.n_hos, model.n_types,
                                                 misreport_iter=args.misreport_iter, misreport_lr=1.0)
    test_regrets = test_model_performance(test_batches, model, batch_size, model.S,
                                          model.n_hos, model.n_types, misreport_iter=test_mis_iter, misreport_lr=1.0)
    #print('test batch regrets', test_regrets)
    #print('train batch regrets', final_train_regrets)

    torch.save(test_regrets, prefix + 'test_batch_regrets.pytorch')
    torch.save(final_train_regrets, prefix + 'train_batch_regrets.pytorch')


def initial_train_loop(train_batches, model, batch_size, single_s, net_lr=1e-1, init_iter=50):
    """
    Train model without misreporting to learning optimal matching first
    """
    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)
    for i in range(init_iter):
        epoch_mean_loss = 0.0
        for c in range(train_batches.shape[0]):
            p = train_batches[c,:,:,:]
            util = model.calc_util(model.forward(p, batch_size), single_s)
            total_loss = -torch.mean(torch.sum(util, dim=1))
            epoch_mean_loss += total_loss.item()
            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()
        print('mean loss', epoch_mean_loss / train_batches.shape[0])


def test_model_performance(test_batches, model, batch_size, single_s, N_HOS, N_TYP, misreport_iter=20, misreport_lr=0.5):
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    all_misreports = test_batches.clone().detach()
    regrets = []
    for c in range(test_batches.shape[0]):

        p = test_batches[c, :, :, :]
        curr_mis = all_misreports[c, :, :, :].clone().detach().requires_grad_(True)
        print(curr_mis)

        curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=misreport_iter,
                                       lr=misreport_lr)

        integer_truthful = model.integer_forward(p, batch_size)
        integer_misreports = model.integer_forward(curr_mis, batch_size)
        print('integer on truthful', integer_truthful)
        print((model.S @ integer_truthful[0]).view(2,-1))
        print('integer on misreports', integer_misreports)
        print((model.S @ integer_misreports[0]).view(2,-1))
        print(curr_mis)
        
        with torch.no_grad():
            mis_input = model.create_combined_misreport(curr_mis, p, self_mask)

            output = model.forward(mis_input, batch_size * model.n_hos)
            mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
            util = model.calc_util(model.forward(p, batch_size), single_s)

            mis_diff = (mis_util - util)  # [batch_size, n_hos]

            regrets.append(mis_diff.detach())
            all_misreports[c, :, :, :] = curr_mis


        all_misreports.requires_grad_(True)
    return regrets


# what we want to know is, what's the difference btwn our allocation total utility (on central matching only,
# because maximal we assume no leftovers are matched internally) and the alternative of doing ONLY internal matching
# in both the integer and relaxed cases

def compare_central_internal_utils(batches, model):
    batch_size = batches.shape[1]
    for c in range(batches.shape[0]):
        p = batches[c,:,:,:]
        model_alloc = model.forward(p, batch_size)
        model_alloc_integer = model.integer_forward(p, batch_size)
        central_util = model.calc_util(model_alloc, model.S)
        central_integer_util = model.calc_util(model_alloc_integer, model.S)

        flat_p = p.view(p.shape[0]*p.shape[1], p.shape[2])
        internal_matches_only = model.internal_linear_prog(flat_p, p.shape[0]*p.shape[1])
        internal_matches_only_integer = model.internal_integer_forward(flat_p, p.shape[0]*p.shape[1])

        internal_counts = internal_matches_only.view(batch_size, model.n_hos, -1) @ torch.transpose(model.int_S, 0, 1)
        internal_util = torch.sum(internal_counts, dim=2)

        internal_counts_integer = internal_matches_only_integer.view(batch_size, model.n_hos, -1) @ torch.transpose(model.int_S, 0, 1)
        internal_util_integer = torch.sum(internal_counts_integer, dim=2)
        return internal_util, central_util, internal_util_integer, central_integer_util

def train_loop(train_batches, model, batch_size, single_s, N_HOS, N_TYP, net_lr=1e-2, lagr_lr=1.0, main_iter=50,
               misreport_iter=50, misreport_lr=1.0, rho=10.0, verbose=False):
    # MASKS
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS)  # TODO: Maybe better initilization?
    lagr_update_counter = 0

    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)
    lagr_optim = optim.SGD(params=[lagr_mults], lr=lagr_lr)
    all_tot_loss_lst = []
    all_rgt_loss_lst = []
    all_util_loss_lst = []

    # Training loop
    all_misreports = train_batches.clone().detach()
    for i in range(main_iter):
        tot_loss_lst = []
        rgt_loss_lst = []
        util_loss_lst = []
        # For each batch in training batches
        for c in tqdm(range(train_batches.shape[0])):
            # true input by batch dim [batch size, n_hos, n_types]
            p = train_batches[c,:,:,:]
            curr_mis = all_misreports[c,:,:,:].clone().detach().requires_grad_(True)

            curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=misreport_iter, lr=misreport_lr)

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
            util_loss_lst.append(torch.mean(torch.sum(util, dim=1)).item())

            # Update Lagrange multipliers every 5 iterations
            if lagr_update_counter % 5 == 0:
                lagr_optim.zero_grad()
                (-lagr_loss).backward(retain_graph=True)
                lagr_optim.step()
            lagr_update_counter += 1

            # Update model weights
            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()
            with torch.no_grad():
                all_misreports[c,:,:,:] = curr_mis
            all_misreports.requires_grad_(True)
        all_tot_loss_lst.append(tot_loss_lst)
        all_rgt_loss_lst.append(rgt_loss_lst)
        all_util_loss_lst.append(util_loss_lst)

        # Print current allocations and difference between allocations and internal matching
        '''
        if i % 5 == 0 and verbose:
            # TODO: ONLY WORKS FOR TWO TWO!!!!!
            counts = (model.forward(p, batch_size) @ model.S.transpose(0, 1)).view(batch_size, 2, 2)
            internal_match = p.clone().detach()
            internal_match[:, :, 1] = internal_match[:, :, 0]
            print(counts)
            print(counts - internal_match)
        '''
        print('total loss', total_loss.item())
        print('rgt_loss', rgt_loss.item())
        print('non-quadratic regret', rgt)
        print('lagr_loss', lagr_loss.item())
        print('mean util', torch.mean(torch.sum(util, dim=1)))

    return train_batches, all_rgt_loss_lst, all_tot_loss_lst, all_util_loss_lst


def create_basic_plots(dir_name):
    """
    Creates basic plots for result of model

    :param dir_name: name of directory with model
    """
    # load hyper parameters from json
    with open(dir_name + 'argfile.json') as args_file:
        args = json.load(args_file)

    # load losses
    tot_loss = np.load(dir_name + 'tot_loss.npy')
    util_loss = np.load(dir_name + 'util_loss.npy')
    rgt_loss = np.load(dir_name + 'rgt_loss.npy')
    training_batches = np.load(dir_name + 'train_batches.npy')

    # calculate optimal internal match mean in the batches
    optimal_train_matching_util = 2 * training_batches.min(axis=-1).sum(axis=-1).mean()
    optimal_test_matching_util = 2 * training_batches.min(axis=-1).sum(axis=-1).mean()

    # Plot total loss and loss from regret
    plt.figure()
    plt.plot(np.arange(1, args['main_iter'] + 1), tot_loss.mean(axis=1), 'o--')
    plt.plot(np.arange(1, args['main_iter'] + 1), rgt_loss.mean(axis=1), 'x--')
    plt.legend(['Average Total loss', 'Average Regret loss'])

    # Plot utility gained from matching
    plt.figure()
    plt.plot(np.arange(1, args['main_iter'] + 1), util_loss.mean(axis=1), 'o--')
    plt.hlines(optimal_train_matching_util, linestyles='solid', xmin=0, xmax=args['main_iter'], color='red')
    plt.legend(['MatchNet', 'Optimal strategy proof matching'], loc='lower right')


parser = argparse.ArgumentParser()

parser.add_argument('--main-lr', type=float, default=1e-1, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=5, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--nbatch', type=int, default=4, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=20, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=.1, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=5.0, help='control strength in cvxpy objective')



# parameters
if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    #two_two_experiment(args)
    #realistic_experiment(args)
    ashlagi_7_type_experiment(args)
    #ashlagi_7_type_single(args)
