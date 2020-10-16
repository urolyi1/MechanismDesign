import torch
from torch import nn


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
    pass
