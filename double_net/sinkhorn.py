import torch
import numpy as np


def sinkhorn_plan(dist_mat, a, b, epsilon=1e-1, rounds=2, debug=False):
    K = torch.exp(-dist_mat / epsilon)
    Kt = K.transpose(-1, -2)
    v = torch.ones_like(b)
    # these einsums do batched matrix multiply K @ v, Kt @ u
    u = a / torch.einsum('...ij,...j->...i', K, v)
    v = b / torch.einsum('...ij,...j->...i', Kt, u)

    if debug:
        u_diffs = []
        v_diffs = []

    for i in range(rounds):
        if debug:
            old_us = u.clone()
            old_vs = v.clone()

        u = a / torch.einsum('...ij,...j->...i', K, v)
        v = b / torch.einsum('...ij,...j->...i', Kt, u)

        if debug:
            u_diff = u - old_us
            v_diff = v - old_vs
            u_diffs.append( u_diff.flatten(start_dim=1).norm(dim=1).mean().item() )
            v_diffs.append( v_diff.flatten(start_dim=1).norm(dim=1).mean().item() )


    # this einsum does batched torch.diag(u) @ K @ torch.diag(v)

    if debug:
        print("mean norm of differences in u at each iteration", u_diffs)
        print("mean norm of differences in v at each iteration", v_diffs)

    return torch.einsum('...i,...ik,...k->...ik', u, K, v)

def generate_marginals_demands(agent_demand_list, item_supply_list):
    agent_demand_list = np.array([float(x) for x in agent_demand_list])
    item_supply_list = np.array([float(x) for x in item_supply_list])

    total_agent_demand = agent_demand_list.sum()
    total_item_supply = item_supply_list.sum()

    if total_agent_demand > total_item_supply:
        item_supply_list = np.append(item_supply_list, [ (total_agent_demand - total_item_supply) + 1.0])
        agent_demand_list = np.append(agent_demand_list, [1.0])
    else:
        agent_demand_list = np.append(agent_demand_list, [ (total_item_supply - total_agent_demand) + 1.0])
        item_supply_list = np.append(item_supply_list, [1.0])

    # assert agent_demand_list.sum() == item_supply_list.sum()

    return torch.tensor(agent_demand_list), torch.tensor(item_supply_list)




def log_sinkhorn_plan(dist_mat, a, b, epsilon, rounds):
    v = torch.ones_like(b)
    g = epsilon * torch.log(v)
    for i in range(rounds):
        f = -epsilon * torch.logsumexp(-(dist_mat - g[..., None]) / epsilon, dim=-1) + \
            epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp(-(dist_mat - f[..., None, :]) / epsilon, dim=1) + \
            epsilon * torch.log(b)

    return torch.exp((-dist_mat + f[..., None] + g[..., None, :]) / epsilon)


def generate_marginals(n_agents, n_items):
    main_agents = [1.0 for _ in range(n_agents)]
    main_items = [1.0 for _ in range(n_items)]
    if n_agents > n_items:
        main_items.append((n_agents - n_items) + 1.0)
        main_agents.append(1.0)
    else:
        main_agents.append((n_items - n_agents) + 1.0)
        main_items.append(1.0)

    assert sum(main_agents) == sum(main_items)

    return torch.tensor(main_agents), torch.tensor(main_items)
