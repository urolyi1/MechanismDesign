import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm as tqdm


def curr_timestamp():
    return datetime.strftime(datetime.now(), format='%Y-%m-%d_%H-%M-%S')


def optimize_misreports(model, curr_mis, truthful, batch_size, iterations=10, lr=1e-1):
    """Otimization to find best misreports on current model

    :param model: MatchNet object
    :param curr_mis: current misreports (duplicate of truthful bids) [batch_size, n_hos, n_types]
    :param truthful: truthful bids [batch_size, n_hos, n_types]
    :param batch_size: number of samples
    :param iterations: number of iterations to optimize misreports
    :param lr: learning rate for gradient descent
    :return: current best misreport for each hospital when others report truthfully [batch_size, n_hos, n_types]
    """
    for i in range(iterations):
        # tile current best misreports into valid inputs
        mis_input = model.create_combined_misreport(curr_mis, truthful)

        # zero out gradients
        model.zero_grad()

        # push tiled misreports through network
        output = model.forward(mis_input.view(-1, model.n_hos * model.n_types), batch_size * model.n_hos)

        # calculate utility from output only weighting utility from misreporting hospital
        mis_util = model.calc_mis_util(output, truthful)
        mis_tot_util = torch.sum(mis_util)
        mis_tot_util.backward()

        # Gradient descent
        with torch.no_grad():
            curr_mis = curr_mis + lr * curr_mis.grad

            # clamping misreports to be valid
            curr_mis = torch.max(torch.min(curr_mis, truthful), torch.zeros_like(curr_mis))
        curr_mis.requires_grad_(True)
    #print(torch.sum(torch.abs(orig_mis_input - mis_input)))
    return curr_mis.detach()


def test_model_performance(model, test_batches, misreport_iter=1000, misreport_lr=0.1):
    """Given a model calculate various performance metrics

    :param model: MatchNet object being tested
    :param test_batches: batches to use when calculating performance
    :param misreport_iter: number of iterations to optimize misreport for
    :param misreport_lr: learning rate when advesarially optimizing misreport
    :return: (tuple) calculated training regret, best misreports for each sample in batch
    """
    batch_size = test_batches.shape[1]
    all_misreports = test_batches.clone().detach() * 0.0
    regrets = []

    # For each batch run evaluation metrics
    for c in range(test_batches.shape[0]):
        # Truthful bid
        p = test_batches[c, :, :, :]
        curr_mis = all_misreports[c, :, :, :].clone().detach().requires_grad_(True)
        print('truthful bids', p)

        # Optimize misreports
        curr_mis = optimize_misreports(model, curr_mis, p, batch_size, iterations=misreport_iter, lr=misreport_lr)
        print('Optimized best misreport on final mechanism', curr_mis)

        integer_truthful = model.integer_forward(p, batch_size)
        integer_misreports = model.integer_forward(curr_mis, batch_size)

        # Print integer allocations on truthful and misreport
        print('integer on truthful', integer_truthful)
        print('Integer allocation on truthful first sample', (model.S @ integer_truthful[0]).view(2, -1))
        print('integer on misreports', integer_misreports)
        print('Integer allocation on misreported first sample', (model.S @ integer_misreports[0]).view(2, -1))
        print('truthful first sample', p[0])

        with torch.no_grad():
            mis_input = model.create_combined_misreport(curr_mis, p)
            output = model.forward(mis_input, batch_size * model.n_hos)
            mis_util = model.calc_mis_util(output, p)
            central_util, internal_util = model.calc_util(model.forward(p, batch_size), p)

            mis_diff = (mis_util - (central_util + internal_util))  # [batch_size, n_hos]

            regrets.append(mis_diff.detach())
            all_misreports[c, :, :, :] = curr_mis

        all_misreports.requires_grad_(True)
    return regrets, all_misreports


def init_train_loop(model, train_batches, main_iter, net_lr=1e-2):
    batch_size = train_batches.shape[1]

    # Model optimizers
    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)
    print('INITIAL TRAIN LOOP')
    for i in range(main_iter):
        for c in range(train_batches.shape[0]):
            p = train_batches[c, :, :, :]
            central_util, internal_util = model.calc_util(model.forward(p, batch_size), p)
            total_loss = -1.0 * torch.mean(torch.sum(central_util + internal_util, dim=1))

            # Update model weights
            model_optim.zero_grad()
            total_loss.backward()
            model_optim.step()


def single_train_step(
    model, p, curr_mis, batch_size, model_optim,
    lagr_mults, misreport_iter, misreport_lr, rho
):
    # Print best misreport pre-misreport optimization
    print(f'best misreport pre-optimization', curr_mis[0])

    # Run misreport optimization step
    curr_mis = optimize_misreports(model, curr_mis, p, batch_size, iterations=misreport_iter, lr=misreport_lr)

    # Print best misreport post optimization
    print(f'best misreport post-optimization', curr_mis[0])

    # Calculate utility from best misreports
    mis_input = model.create_combined_misreport(curr_mis, p)
    output = model.forward(mis_input, batch_size * model.n_hos)
    mis_util = model.calc_mis_util(output, p)
    central_util, internal_util = model.calc_util(model.forward(p, batch_size), p)

    # Difference between truthful utility and best misreport util
    mis_diff = (mis_util - (central_util + internal_util))  # [batch_size, n_hos]
    mis_diff = torch.max(mis_diff, torch.zeros_like(mis_diff))
    rgt = torch.mean(mis_diff, dim=0)  # [n_hos]

    # computes losses
    rgt_loss = rho * torch.sum(torch.mul(rgt, rgt))
    lagr_loss = torch.sum(rgt * lagr_mults)
    total_loss = rgt_loss + lagr_loss - torch.mean(torch.sum(central_util + internal_util, dim=1))
    mean_util = torch.mean(torch.sum(central_util + internal_util, dim=1))

    # Update model weights
    model_optim.zero_grad()
    total_loss.backward()
    model_optim.step()

    # Return total loss, regret loss, and mean utility of batch
    return total_loss.item(), rgt_loss.item(), mean_util.item(), lagr_loss.item()


def train_loop(
    model, train_batches, net_lr=1e-2, lagr_lr=2.0, main_iter=50,
    misreport_iter=50, misreport_lr=1.0, rho=10.0, disable=False
):
    # Getting certain model parameters
    N_HOS = model.n_hos
    batch_size = train_batches.shape[1]

    # initializing lagrange multipliers to 1
    lagr_mults = torch.ones(N_HOS).requires_grad_(True)
    lagr_update_counter = 0

    # Model optimizers
    model_optim = optim.Adam(params=model.parameters(), lr=net_lr)

    # Lists to track total loss, regret, and utility
    all_tot_loss_lst = []
    all_rgt_loss_lst = []
    all_util_loss_lst = []

    # Initialize best misreports to just truthful
    all_misreports = train_batches.clone().detach() * 0.5

    # Training loop
    for i in range(main_iter):
        print("Iteration: ", i)
        print('lagrange multipliers: ', lagr_mults)

        # Lists to track loss over iterations
        tot_loss_lst = []
        rgt_loss_lst = []
        util_loss_lst = []

        # For each batch in training batches
        for c in tqdm(range(train_batches.shape[0]), disable=disable):
            # true input by batch dim [batch size, n_hos, n_types]
            p = train_batches[c, :, :, :]
            curr_mis = all_misreports[c, :, :, :].clone().detach().requires_grad_(True)
            tot_loss, rgt_loss, util, lagr_loss = single_train_step(
                model, p, curr_mis, batch_size, model_optim, lagr_mults, misreport_iter, misreport_lr, rho
            )

            # Add performance to lists
            tot_loss_lst.append(tot_loss)
            rgt_loss_lst.append(rgt_loss)
            util_loss_lst.append(util)

            # Update Lagrange multipliers every 5 iterations
            if lagr_update_counter % 5 == 0:
                with torch.no_grad():
                    lagr_mults += lagr_lr * lagr_mults.grad
                    lagr_mults.grad.zero_()
            lagr_update_counter += 1

            # Save current best misreport
            with torch.no_grad():
                all_misreports[c, :, :, :] = curr_mis
            all_misreports.requires_grad_(True)

        all_tot_loss_lst.append(tot_loss_lst)
        all_rgt_loss_lst.append(rgt_loss_lst)
        all_util_loss_lst.append(util_loss_lst)

        # Print current allocations and difference between allocations and internal matching
        print('total loss', tot_loss)
        print('rgt_loss', rgt_loss)
        print('lagr_loss', lagr_loss)
        print('mean util', util)
        print("----------------------")

    return train_batches, all_rgt_loss_lst, all_tot_loss_lst, all_util_loss_lst

