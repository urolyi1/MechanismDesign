import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from double_net import utils_misreport as utils

class RegretNetUnitDemand(nn.Module):
    def __init__(
            self, n_agents, n_items, clamp_op=None, hidden_layer_size=128,
            n_hidden_layers=2, activation='tanh', separate=False
    ):
        super(RegretNetUnitDemand, self).__init__()
        self.activation = activation
        if activation == 'tanh':
            self.act = nn.Tanh
        else:
            self.act = nn.ReLU

        self.clamp_op = clamp_op

        self.n_agents = n_agents
        self.n_items = n_items

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.separate = separate

        # outputs are agents (+dummy agent) per item (+ dummy item), plus payments per agent
        self.allocations_size = (self.n_agents + 1) * (self.n_items + 1)
        self.payments_size = self.n_agents

        self.nn_model = nn.Sequential(
            *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
              [l for i in range(self.n_hidden_layers)
               for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
        )

        self.allocation_head = nn.Linear(self.hidden_layer_size, self.allocations_size * 2)
        self.fractional_payment_head = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()
        )

    def glorot_init(self):
        """
        reinitializes with glorot (aka xavier) uniform initialization
        """

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)

        alloc_scores = self.allocation_head(x)
        alloc_first = F.softmax(alloc_scores[:, 0:self.allocations_size].view(-1, self.n_agents + 1, self.n_items + 1),
                                dim=1)
        alloc_second = F.softmax(
            alloc_scores[:, self.allocations_size:self.allocations_size * 2].view(-1, self.n_agents + 1,
                                                                                  self.n_items + 1), dim=2)
        allocs = torch.min(alloc_first, alloc_second)

        payments = self.fractional_payment_head(x) * torch.sum(
            allocs[:, :-1, :-1] * reports, dim=2
        )

        return allocs[:, :-1, :-1], payments


def train_loop(model, train_loader, args, device="cpu"):

    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho

    # local_optimum_model = None

    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)

        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = batch.clone().detach().to(device)
            utils.optimize_misreports(model, batch, misreport_batch, misreport_iter=args.misreport_iter, lr=args.misreport_lr)

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
            # if epoch >= args.fair_start:
                # if local_optimum_model is None:
                #     local_optimum_model = RegretNet(args.n_agents, args.n_items, activation='relu',
                #                                     hidden_layer_size=args.hidden_layer_size,
                #                                     n_hidden_layers=args.n_hidden_layers,
                #                                     separate=args.separate).to(device)
                #     local_optimum_model.load_state_dict(model.state_dict())


        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            # "regret_min": regrets_epoch.min().item(),
            "regret_mean": regrets_epoch.mean().item(),

            # "payment_max": payments_epoch.sum(dim=1).max().item(),
            # "payment_min": payments_epoch.sum(dim=1).min().item(),
            "payment": payments_epoch.sum(dim=1).mean().item(),

            # "fairprice_max": price_of_fair_epoch.max().item(),
            # "fairprice_min": price_of_fair_epoch.min().item(),
            # "fairprice_mean": price_of_fair_epoch.mean().item(),
        }
        print(train_stats)

        mult_stats = {
            "regret_mult": regret_mults.mean().item(),
            # "regret_rho": rho,
            "payment_mult": payment_mult,
            # "fair_rho": rho_fair
        }
