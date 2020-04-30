import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch import optim
from tqdm import tqdm as tqdm

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

        self.neural_net = nn.Sequential(nn.Linear(self.m * self.n, 128), nn.Tanh(), nn.Linear(128, 128),
                                        nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n))


        self.mis_mask = torch.zeros(self.n, 1, self.n)
        self.mis_mask[np.arange(self.n), :, np.arange(self.n)] = 1.0
        self.self_mask = torch.zeros(self.n, 1, self.n, self.m)
        self.self_mask[np.arange(self.n), :, np.arange(self.n), :] = 1.0

    def forward(self, demand_vectors):
        # should take valuations from players 1 and 2 and compute how many tasks they get.
        flat_demand_vectors = demand_vectors.view(-1, demand_vectors.shape[1]*demand_vectors.shape[2])
        # compute neural network output, but make it zero for now
        nn_output = self.neural_net(demand_vectors.view(-1, self.n*self.m))

        resulting_x, = self.problem_layer(demand_vectors, nn_output)

        resources_allocated = (demand_vectors * resulting_x.unsqueeze(2))

        # not currently returning resulting_x

        return resources_allocated

    def create_combined_misreport(self, curr_mis, true_rep):
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
        only_mis = curr_mis.view(1, -1, self.n, self.m).repeat(self.n, 1, 1, 1) * self.self_mask
        other_hos = true_rep.view(1, -1, self.n, self.m).repeat(self.n, 1, 1, 1) * (1 - self.self_mask)
        result = only_mis + other_hos
        return result, true_rep.view(1, -1, self.n, self.m).repeat(self.n, 1, 1, 1)

    def calc_mis_util(self, tiled_resources_alloc, tiled_demand_vectors):
        # recreate calc_util with mask
        # return full-size tiled central util

        batch_size = tiled_resources_alloc.shape[0] // self.n
        flat_demand_vectors = tiled_demand_vectors.view(-1, self.n, self.m)
        job_scores = tiled_resources_alloc / flat_demand_vectors
        raw_util = torch.min(job_scores, dim=2)[0] # number of jobs for each player in each misreport
        util = raw_util.view(self.n, batch_size, self.n) * self.mis_mask
        util, _ = torch.max(util, dim=-1, keepdim=False)
        util = util.transpose(0, 1)
        return util

    def calc_util(self, resources_alloc, true_demand_vectors):
        '''
        Input:
        resources_alloc: allocated resources (not tasks) [batch_size, n_agents, n_items]
        true_demand_vectors: demand vectors to use for computing utility [batch_size, n_agents, n_items]

        Output:
        [batch_size, n_agents]: utility in terms of number of tasks
        '''

        # divide true demand vectors by resources allocated and take elementwise min (to find limiting resource).
        job_scores = resources_alloc / true_demand_vectors

        return torch.min(job_scores, dim=2)[0]


def optimize_misreports(model, curr_mis, truthful, batch_size, iterations=10, lr=1e-1):
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
        mis_input, truthful_tiled = model.create_combined_misreport(curr_mis, truthful)

        # zero out gradients
        model.zero_grad()

        # push tiled misreports through network
        output = model.forward(mis_input.view(-1, model.n, model.m))

        # calculate utility from output only weighting utility from misreporting hospital
        mis_util = model.calc_mis_util(output, truthful_tiled)
        mis_tot_util = torch.sum(mis_util)
        mis_tot_util.backward()

        # Gradient descent
        with torch.no_grad():
            curr_mis = curr_mis + lr * curr_mis.grad

            # no need to clamp misreports
        curr_mis.requires_grad_(True)
    #print(torch.sum(torch.abs(orig_mis_input - mis_input)))
    return curr_mis.detach()


def train_loop(model, train_batches, net_lr=1e-2, lagr_lr=1.0, main_iter=50, misreport_iter=50, misreport_lr=1.0,
               rho=10.0, verbose=False):
    # Getting certain model parameters
    N_AGENTS = model.n
    batch_size = train_batches.shape[1]

    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_AGENTS)  # TODO: Maybe better initilization?
    lagr_update_counter = 0

    # Model optimizers
    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)
    lagr_optim = optim.SGD(params=[lagr_mults], lr=lagr_lr)

    # Lists to track total loss, regret, and utility
    all_tot_loss_lst = []
    all_rgt_loss_lst = []
    all_util_loss_lst = []

    # Initialize best misreports to just truthful
    all_misreports = train_batches.clone().detach()

    # Training loop
    for i in range(main_iter):
        # Lists to track loss over iterations
        tot_loss_lst = []
        rgt_loss_lst = []
        util_loss_lst = []

        # For each batch in training batches
        for c in tqdm(range(train_batches.shape[0])):
            # true input by batch dim [batch size, n_hos, n_types]
            p = train_batches[c,:,:,:]
            curr_mis = all_misreports[c,:,:,:].clone().detach().requires_grad_(True)

            # Run misreport optimization step
            # TODO: Print better info about optimization
            #  at last starting utility vs ending utility maybe also net difference?

            # Print best misreport pre-misreport optimization
            if verbose and lagr_update_counter % 5 == 0:
                print('best misreport pre-optimization', curr_mis[0])

            curr_mis = optimize_misreports(model, curr_mis, p, batch_size, iterations=misreport_iter, lr=misreport_lr)

            # Print best misreport post optimization
            if verbose and lagr_update_counter % 5 == 0:
                print('best misreport post-optimization', curr_mis[0])

            # Calculate utility from best misreports
            mis_input, tiled_p = model.create_combined_misreport(curr_mis, p)
            output = model.forward(mis_input.view(-1, model.n, model.m))
            mis_util = model.calc_mis_util(output, tiled_p)
            central_util = model.calc_util(model.forward(p), p)

            # Difference between truthful utility and best misreport util
            mis_diff = mis_util - central_util   # [batch_size, n_hos]
            mis_diff = torch.max(mis_diff, torch.zeros_like(mis_diff))
            rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

            # computes losses
            rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
            lagr_loss = torch.sum(torch.mul(rgt, lagr_mults))
            total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(central_util, dim=1))

            # Add performance to lists
            tot_loss_lst.append(total_loss.item())
            rgt_loss_lst.append(rgt_loss.item())
            util_loss_lst.append(torch.mean(torch.sum(central_util , dim=1)).item())

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

            # Save current best misreport
            with torch.no_grad():
                all_misreports[c,:,:,:] = curr_mis
            all_misreports.requires_grad_(True)

        all_tot_loss_lst.append(tot_loss_lst)
        all_rgt_loss_lst.append(rgt_loss_lst)
        all_util_loss_lst.append(util_loss_lst)

        # Print current allocations and difference between allocations and internal matching
        print('total loss', total_loss.item())
        print('rgt_loss', rgt_loss.item())
        print('non-quadratic regret', rgt)
        print('lagr_loss', lagr_loss.item())
        print('mean util', torch.mean(torch.sum(central_util, dim=1)))

    return train_batches, all_rgt_loss_lst, all_tot_loss_lst, all_util_loss_lst

if __name__ == '__main__':
    resource_constraints = torch.tensor([10.0,10.0])
    ceei = CEEI(resource_constraints)
    bids = torch.tensor([[[2.0, 3.0], [2.0, 3.0]],[ [1.0, 1.0], [1.0, 1.0]], [[2.0,2.0],[2.0,2.0]]])
    resources = ceei.forward(bids)

    true_util = ceei.calc_util(resources, bids)
    tiled, tiled_true = ceei.create_combined_misreport(bids, bids)

    tiled_resources = ceei.forward(tiled.view(-1, ceei.n, ceei.m))

    mis_util = ceei.calc_mis_util(tiled_resources, tiled_true)

    train_loop(ceei, bids.unsqueeze(0))




