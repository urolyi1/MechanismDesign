import json
import pickle

import torch
import torch.optim as optim
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt


def curr_timestamp():
    return datetime.strftime(datetime.now(), format='%Y-%m-%d_%H-%M-%S')


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



def test_model_performance(test_batches, model, batch_size, N_HOS, N_TYP, misreport_iter=1000, misreport_lr=0.1):
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    all_misreports = test_batches.clone().detach()
    regrets = []
    for c in range(test_batches.shape[0]):
        # Truthful bid
        p = test_batches[c, :, :, :]

        # Best misreport
        curr_mis = all_misreports[c, :, :, :].clone().detach().requires_grad_(True)
        print('truthful bids', p)


        curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=misreport_iter,
                                       lr=misreport_lr)
        print('Optimized best misreport on final mechanism', curr_mis)
        integer_truthful = model.integer_forward(p, batch_size)
        integer_misreports = model.integer_forward(curr_mis, batch_size)
        print('integer on truthful', integer_truthful)
        print('Integer allocation on truthful first sample', (model.S @ integer_truthful[0]).view(2,-1))
        print('integer on misreports', integer_misreports)
        print('Integer allocation on misreported first sample', (model.S @ integer_misreports[0]).view(2,-1))
        print('truthful first sample', p[0])

        with torch.no_grad():
            mis_input = model.create_combined_misreport(curr_mis, p, self_mask)

            output = model.forward(mis_input, batch_size * model.n_hos)
            mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
            util = model.calc_util(model.forward(p, batch_size), model.S)

            mis_diff = (mis_util - util)  # [batch_size, n_hos]

            regrets.append(mis_diff.detach())
            all_misreports[c, :, :, :] = curr_mis


        all_misreports.requires_grad_(True)
    return regrets, all_misreports


def train_loop(train_batches, model, batch_size, S, N_HOS, N_TYP, net_lr=1e-2, lagr_lr=1.0, main_iter=50,
               misreport_iter=50, misreport_lr=1.0, rho=10.0, verbose=False):
    # MASKS
    # self_mask only has 1's for indices of form [i, :, i, :]
    self_mask = torch.zeros(N_HOS, batch_size, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0

    # Misreport mask that only has 1's for indices [i, :, i]
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS)  # TODO: Maybe better initilization?
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
            # TODO: Print better info about optimization at last starting utility vs ending utility maybe also net difference?

            # Print best misreport pre-misreport optimization
            if verbose and lagr_update_counter % 5 == 0:
                print('best misreport pre-optimization', curr_mis[0])

            curr_mis = optimize_misreports(model, curr_mis, p, mis_mask, self_mask, batch_size, iterations=misreport_iter, lr=misreport_lr)

            # Print best misreport post optimization
            if verbose and lagr_update_counter % 5 == 0:
                print('best misreport post-optimization', curr_mis[0])

            # Calculate utility from best misreports
            mis_input = model.create_combined_misreport(curr_mis, p, self_mask)
            output = model.forward(mis_input, batch_size * model.n_hos)
            mis_util = model.calc_mis_util(p, output, model.S, mis_mask)
            util = model.calc_util(model.forward(p, batch_size), S)

            # Difference between truthful utility and best misreport util
            mis_diff = (mis_util - util)  # [batch_size, n_hos]
            mis_diff = torch.max(mis_diff, torch.zeros_like(mis_diff))
            rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

            # computes losses
            rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
            lagr_loss = torch.sum(torch.mul(rgt, lagr_mults))
            total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(util, dim=1))

            # Add performance to lists
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

parser.add_argument('--main-lr', type=float, default=5e-2, help='main learning rate')
parser.add_argument('--main-iter', type=int, default=8, help='number of outer iterations')
parser.add_argument('--init-iter', type=int, default=100, help='number of outer iterations')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--nbatch', type=int, default=2, help='number of batches')
parser.add_argument('--misreport-iter', type=int, default=100, help='number of misreport iterations')
parser.add_argument('--misreport-lr', type=float, default=.25, help='misreport learning rate')
parser.add_argument('--random-seed', type=int, default=0, help='random seed')
parser.add_argument('--control-strength', type=float, default=5.0, help='control strength in cvxpy objective')



# parameters
if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    two_two_experiment(args)
    #realistic_experiment(args)
    #ashlagi_7_type_experiment(args)
    #ashlagi_7_type_single(args)
