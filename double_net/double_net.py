import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from double_net import utils_misreport as utils


class DoubleNet(nn.module):
    def __init__(self, n_agents, n_items):
        super(DoubleNet, self).__init__()
        self.n_agents = n_agents
        self.n_items = n_items

        self.neural_net = nn.Sequential(
            nn.Linear(self.n_agents * self.n_items, 128), nn.Tanh(), nn.Linear(128, 128),
            nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, self.n_agents * self.n_items)
        )

    def neural_network_forward(self, bids):
        """Augments bids in neural network

        :param bids: bids from bidders on items [batch_size, n_agents * n_items]
        :return: augmented bids [batch_size, n_agents * n_items]
        """
        augmented = self.neural_net(bids)
        clamped_bids = torch.min(bids, augmented)  # Making sure neural network does not increase bid of bidders

        return clamped_bids

    def bipartite_matching(self, bids):
        """Given bids finds max-weight bipartite matching

        :param bids: bids that will be used as edge weights [batch_size, n_agents * n_items]
        :return: allocations [batch_size, n_agents * n_items]
        """
        return torch.zeros(bids.shape)

    def forward(self, bids):
        """
        :param bids: bids from bidders on items [batch_size, n_agents, n_items]
        :return: allocations tensor [batch_size, n_agents, n_items], payments tensor [batch_size, n_agents]
        """
        X = bids.view(-1, self.n_agents * self.n_items)
        augmented = self.neural_network_forward(X)

        allocs = self.bipartite_matching(augmented)
        payments = (allocs * augmented).view(-1, self.n_agents, self.n_items).sum(dim=-1)

        return allocs, payments


def train_loop(
    model, train_loader, args, device='cpu'
):
    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho

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
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()
                # regret_loss = (regret_mults * (positive_regrets + positive_regrets.max(dim=0).values) / 2).mean()
                # regret_quad = (rho / 2.0) * ((positive_regrets ** 2).mean() +
                #                              (positive_regrets.max(dim=0).values ** 2).mean()) / 2


            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            # price_of_fair_epoch = torch.cat((price_of_fair_epoch, price_of_fair), dim=0)

            # Calculate loss
            loss_func = regret_loss \
                        + regret_quad \
                        - payment_loss \

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
            # "regret_min": regrets_epoch.min().item(),
            "regret_mean": regrets_epoch.mean().item(),

            # "payment_max": payments_epoch.sum(dim=1).max().item(),
            # "payment_min": payments_epoch.sum(dim=1).min().item(),
            "payment": payments_epoch.sum(dim=1).mean().item(),
        }
        print(train_stats)

        mult_stats = {
            "regret_mult": regret_mults.mean().item(),
            # "regret_rho": rho,
            "payment_mult": payment_mult,
        }
