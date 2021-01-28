import pickle
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from double_net import utils_misreport as utils
from double_net.sinkhorn import generate_marginals, log_sinkhorn_plan, generate_additive_marginals, generate_exact_one_marginals
from double_net import datasets as ds


class DoubleNet(nn.Module):
    def __init__(self, n_agents, n_items, item_ranges, sinkhorn_epsilon, sinkhorn_rounds, marginal_choice='unit'):
        super(DoubleNet, self).__init__()
        self.n_agents = n_agents
        self.n_items = n_items
        self.item_ranges = item_ranges
        self.clamp_op = ds.get_clamp_op(item_ranges)
        self.marginal_choice = marginal_choice
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_rounds = sinkhorn_rounds
        
        self.alloc_net = nn.Sequential(
            nn.Linear(self.n_agents * self.n_items, 128), nn.Tanh(), nn.Linear(128, 128),
            nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_agents * self.n_items)
        )
        self.payment_net = nn.Sequential(
            nn.Linear(self.n_agents * self.n_items, 128), nn.Tanh(), nn.Linear(128, 128),
            nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_agents), nn.Sigmoid()
        )

        if marginal_choice == 'unit':
            agents_marginal, items_marginal = generate_marginals(self.n_agents, self.n_items)
        elif marginal_choice == 'additive':
            agents_marginal, items_marginal = generate_additive_marginals([self.n_items] * self.n_agents, [1] * self.n_items)
        elif marginal_choice == 'exact_one':
            agents_marginal, items_marginal = generate_exact_one_marginals(self.n_agents, self.n_items)
        else:
            raise NotImplementedError(f"{marginal_choice} demand structure not implemented")

        self.register_buffer('agents_marginal', agents_marginal)
        self.register_buffer('items_marginal', items_marginal)

    def neural_network_forward(self, bids):
        """Augments bids in neural network

        :param bids: bids from bidders on items [batch_size, n_agents * n_items]
        :return: augmented bids [batch_size, n_agents * n_items]
        """
        augmented = self.alloc_net(bids)
        return augmented

    def bipartite_matching(self, bids):
        """Given bids finds max-weight bipartite matching

        :param bids: bids that will be used as edge weights [batch_size, n_agents * n_items]
        :return: allocations [batch_size, n_agents * n_items]
        """
        batch_size = bids.shape[0]
        bids_matrix = bids.view(-1, self.n_agents, self.n_items)
        padded = -torch.nn.functional.pad(
            bids_matrix,
            [0, 1, 0, 1],
            mode='constant',
            value=0
        )  # pads column on right and row on bottom of zeros
        agent_tiled_marginals = self.agents_marginal.repeat(batch_size, 1)
        item_tiled_marginals = self.items_marginal.repeat(batch_size, 1)

        plan = log_sinkhorn_plan(padded,
                                 agent_tiled_marginals,
                                 item_tiled_marginals,
                                 rounds=self.sinkhorn_rounds, epsilon=self.sinkhorn_epsilon)

        # chop off dummy allocations
        plan_without_dummies = plan[..., 0:-1, 0:-1]

        return plan_without_dummies

    def forward(self, bids):
        """
        :param bids: bids from bidders on items [batch_size, n_agents, n_items]
        :return: allocations tensor [batch_size, n_agents, n_items], payments tensor [batch_size, n_agents]
        """
        X = bids.view(-1, self.n_agents * self.n_items)
        augmented = self.neural_network_forward(X)

        allocs = self.bipartite_matching(augmented)
        payments = self.payment_net(X) * ((allocs * X.view(-1, self.n_agents, self.n_items)).sum(dim=-1))

        return allocs.view(-1, self.n_agents, self.n_items), payments

    def save(self, filename_prefix='./'):
        torch.save(self.alloc_net.state_dict(), filename_prefix + 'alloc_net.pytorch')
        torch.save(self.payment_net.state_dict(), filename_prefix + 'payment_net.pytorch')
        
        params_dict = {
            'n_agents': self.n_agents,
            'n_items': self.n_items,
            'item_ranges': self.item_ranges,
            'sinkhorn_epsilon': self.sinkhorn_epsilon,
            'sinkhorn_rounds': self.sinkhorn_rounds,
            'marginal_choice': self.marginal_choice,
        }
        with open(filename_prefix + 'doublenet_classvariables.pickle', 'wb') as f:
            pickle.dump(params_dict, f)

    @staticmethod
    def load(filename_prefix):
        with open(filename_prefix + 'doublenet_classvariables.pickle', 'rb') as f:
            params_dict = pickle.load(f)

        result = DoubleNet(
            params_dict['n_agents'],
            params_dict['n_items'],
            params_dict['item_ranges'],
            params_dict['sinkhorn_epsilon'],
            params_dict['sinkhorn_rounds'],
            params_dict['marginal_choice'],
        )
        result.alloc_net.load_state_dict(torch.load(filename_prefix + 'alloc_net.pytorch'))
        result.payment_net.load_state_dict(torch.load(filename_prefix + 'payment_net.pytorch'))
        return result
    
    
def train_loop(
    model, train_loader, args, device='cpu'
):
    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho

    mean_regrets = []
    mean_payments = []
    lagrange_mults = []
    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)

        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = batch.clone().detach().to(device)
            utils.optimize_misreports(
                model, batch, misreport_batch, misreport_iter=args.misreport_iter, lr=args.misreport_lr
            )

            allocs, payments = model(batch)
            truthful_util = utils.calc_agent_util(batch, allocs, payments)
            misreport_util = utils.tiled_misreport_util(misreport_batch, batch, model)
            regrets = torch.clamp(misreport_util - truthful_util, min=0)
            positive_regrets = torch.clamp_min(regrets, 0)

            payment_loss = payments.sum(dim=1).mean() * payment_mult

            if epoch < args.rgt_start:
                regret_loss = 0
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)

            # Calculate loss
            loss_func = regret_loss + regret_quad - payment_loss

            # update model
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            # update various fancy multipliers
            # if epoch >= args.rgt_start:
            if iter % args.lagr_update_iter == 0:
                with torch.no_grad():
                    regret_mults += rho * positive_regrets.mean(dim=0)
            if iter % args.rho_incr_iter == 0:
                rho += args.rho_incr_amount

        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            "regret_mean": regrets_epoch.mean().item(),
            "regret_mults": regret_mults,
            "payment": payments_epoch.sum(dim=1).mean().item(),
        }

        # append metrics to lists
        mean_regrets.append(regrets_epoch.mean().item())
        mean_payments.append(payments_epoch.sum(dim=1).mean().item())
        lagrange_mults.append(regret_mults)
        print(train_stats)

    return mean_regrets, mean_payments, lagrange_mults


def train_loop_no_lagrange(
    model, train_loader, args, device='cpu'
):
    payment_mult = 1
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)

        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = batch.clone().detach().to(device)
            utils.optimize_misreports(
                model, batch, misreport_batch, misreport_iter=args.misreport_iter, lr=args.misreport_lr
            )

            allocs, payments = model(batch)
            truthful_util = utils.calc_agent_util(batch, allocs, payments)
            misreport_util = utils.tiled_misreport_util(misreport_batch, batch, model)
            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)

            payment_loss = payments.sum(dim=1).mean() * payment_mult

            if epoch < args.rgt_start:
                regret_loss = 0
            else:
                regret_loss = torch.sqrt(positive_regrets.mean()) + positive_regrets.mean()

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)

            # Calculate loss
            loss_func = regret_loss - payment_loss

            # update model
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            "regret_mean": regrets_epoch.mean().item(),
            "payment": payments_epoch.sum(dim=1).mean().item(),
        }
        print(train_stats)


def test_loop(model, loader, args, device='cpu'):
    # regrets and payments are 2d: n_samples x n_agents; unfairs is 1d: n_samples.
    test_regrets = torch.Tensor().to(device)
    test_payments = torch.Tensor().to(device)

    for i, batch in tqdm(enumerate(loader)):
        batch = batch.to(device)
        misreport_batch = batch.clone().detach()
        utils.optimize_misreports(model, batch, misreport_batch,
                                  misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model(batch)
        truthful_util = utils.calc_agent_util(batch, allocs, payments)
        misreport_util = utils.tiled_misreport_util(misreport_batch, batch, model)

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)

    mean_regret = test_regrets.sum(dim=1).mean(dim=0).item()
    result = {
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        "regret_mean": mean_regret,
        "regret_max": test_regrets.sum(dim=1).max().item(),
    }
    return result
