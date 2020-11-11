import torch
import numpy as np


def sinkhorn_plan(dist_mat, a, b, epsilon=1e-1, rounds=2):
    K = torch.exp(-dist_mat / epsilon)
    Kt = K.transpose(-1, -2)
    v = torch.ones_like(b)
    # these einsums do batched matrix multiply K @ v, Kt @ u
    u = a / torch.einsum('...ij,...j->...i', K, v)
    v = b / torch.einsum('...ij,...j->...i', Kt, u)

    for i in range(rounds):
        u = a / torch.einsum('...ij,...j->...i', K, v)
        v = b / torch.einsum('...ij,...j->...i', Kt, u)

    # this einsum does batched torch.diag(u) @ K @ torch.diag(v)
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
