import itertools
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cvxpy as cp
import argparse
from cvxpylayers.torch import CvxpyLayer
import HospitalGenerators as gens
import diffcp
from tqdm import tqdm as tqdm
import time



class MatchNet(nn.Module):

    def __init__(self, n_hos, n_types, num_structs, int_structs, S, int_S, W=None, internalW=None):
        
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
        if W is not None:
            self.W = W
        else:
            self.W = torch.ones(self.n_structures)
        if internalW is not None:
            self.internalW = internalW
        else:
            self.internalW = torch.ones(self.int_structures)

    def save(self, filename_prefix=None):
        if filename_prefix is None:
            filename_prefix = f'matchnet_{time.time()}'

        torch.save(self.neural_net.state_dict(), filename_prefix + '.pytorch')

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

        with open(filename_prefix+'_classvariables.pickle', 'wb') as f:
            pickle.dump(params_dict, f)

    @staticmethod
    def load(filename_prefix):
        with open(filename_prefix + '_classvariables.pickle', 'rb') as f:
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

        result.neural_net.load_state_dict(torch.load(filename_prefix+'.pytorch'))

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
        W = torch.ones(batch_size, self.n_structures) # currently weight all structurs same
        B = X.view(batch_size, self.n_hos * self.n_types) # max bids to make sure not over allocated

        # feed all parameters through cvxpy layer
        x1_out, = self.l_prog_layer(tiled_S, W, B, z, solver_args={'max_iters': 50000, 'verbose': False})
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

        x_int_out, = self.int_layer(tiled_S, W, B, solver_args={'max_iters': 50000, 'verbose': False})

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


def create_train_sample(generator, num_batches, batch_size=16):
    """
    :param generator:
    :param num_batches:
    :return:
    """
    batches = []
    for i in range(num_batches):
        batches.append(torch.tensor(next(generator.generate_report(batch_size))).float())
    return torch.stack(batches, dim=0)


def blow_up_column(col_vector, num_hospitals):
    nonzero_inds,  = np.nonzero(col_vector)
    all_inds = []
    for ind in nonzero_inds:
        count = int(col_vector[ind])
        all_inds.extend( count*[ind])
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


def convert_internal_S(internal_S, num_hospitals):
    all_cols = []
    for col in range(internal_S.shape[1]):
        all_cols.append(blow_up_column(internal_S[:,col], num_hospitals))
    return np.concatenate(all_cols, axis=1)


def internal_central_bloodtypes(num_hospitals):
    internal_s = np.load('bloodtypematrix.npy')
    central_s = convert_internal_S(internal_s, num_hospitals)
    return torch.tensor(internal_s, dtype=torch.float32, requires_grad=False), torch.tensor(central_s, dtype=torch.float32, requires_grad=False)
def two_two_experiment(args):
    lower_lst = [[10, 20], [30, 60]]
    upper_lst = [[20, 40], [50, 100]]

    generator = gens.create_simple_generator(lower_lst, upper_lst, 2, 2)
    batches = create_train_sample(generator, args.nbatch, batch_size=args.batchsize)
    # parameters
    N_HOS = 2
    N_TYP = 2
    num_structures = 4
    int_structues = 1
    batch_size = batches.shape[1]

    internal_s = torch.tensor([[1.0],
                               [1.0]], requires_grad=False)
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), 2), requires_grad = False, dtype=torch.float32)
    # Internal compatbility matrix [n_types, n_int_structures]

    model = MatchNet(N_HOS, N_TYP, num_structures, int_structues, central_s, internal_s)
    initial_train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP, init_iter=args.init_iter, net_lr=args.main_lr)
    final_p, rgt_loss_lst, tot_loss_lst = train_loop(batches, model, batch_size, central_s, N_HOS, N_TYP,
                                                     main_iter=args.main_iter,
                                                     net_lr=args.main_lr,
                                                     misreport_iter=args.misreport_iter,
                                                     misreport_lr=args.misreport_lr)

    print(tot_loss_lst)
    print(rgt_loss_lst)

    # Actually look at the allocations to see if they make sense
    print((model.forward(final_p[0], batch_size) @ central_s.transpose(0, 1)).view(batch_size, 2, 2))
    print(final_p[0])
    model.save(filename_prefix='test')



def initial_train_loop(train_batches, model, batch_size, single_s, N_HOS, N_TYP, net_lr=1e-2, init_iter=50):
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



def train_loop(train_batches, model, batch_size, single_s, N_HOS, N_TYP, net_lr=1e-2, lagr_lr=1.0, main_iter=50,
               misreport_iter=50, misreport_lr=1.0, rho=10.0):
    # MASKS
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    # Large compatibility matrix [n_hos_pair_combos, n_structures]
    # regret quadratic term weight
    # true input by batch dim [batch size, n_hos, n_types]
    # p = torch.tensor(np.arange(batch_size * N_HOS * N_TYP)).view(batch_size, N_HOS, N_TYP).float()
    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS)  # TODO: Maybe better initilization?
    # Making model
    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)
    lagr_optim = optim.SGD(params=[lagr_mults], lr=lagr_lr)
    tot_loss_lst = []
    rgt_loss_lst = []
    # Training loop
    all_misreports = train_batches.clone().detach()
    for i in range(main_iter):
        for c in tqdm(range(train_batches.shape[0])):
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

            print('total loss', total_loss.item())
            print('rgt_loss', rgt_loss.item())
            print('lagr_loss', lagr_loss.item())
            print('mean util', torch.mean(torch.sum(util, dim=1)))

            if i % 5 == 0:
                lagr_optim.zero_grad()
                (-lagr_loss).backward(retain_graph=True)
                lagr_optim.step()

            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()
            with torch.no_grad():
                all_misreports[c,:,:,:] = curr_mis
            all_misreports.requires_grad_(True)

    print(all_misreports)
    return train_batches, rgt_loss_lst, tot_loss_lst

parser = argparse.ArgumentParser()

parser.add_argument('--main-lr', type=float, default=1e-1, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=25, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=16, help='batch size')
parser.add_argument('--nbatch', type=int, default=3, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=10, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=5.0, help='misreport learning rate')

# parameters
if __name__ == '__main__':
    args = parser.parse_args()
    two_two_experiment(args)
