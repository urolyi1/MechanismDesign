import torch
import numpy as np
import logging


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

def generate_additive_marginals(agent_demand_list, item_supply_list):
    agent_demand_list = np.array([float(x) for x in agent_demand_list])
    item_supply_list = np.array([float(x) for x in item_supply_list])

    total_agent_demand = agent_demand_list.sum()
    total_item_supply = item_supply_list.sum()

    agent_demand_list = np.append(agent_demand_list, [total_item_supply])
    item_supply_list = np.append(item_supply_list, [total_agent_demand])

    assert agent_demand_list.sum() == item_supply_list.sum()

    return torch.tensor(agent_demand_list).float(), torch.tensor(item_supply_list).float()


def generate_exact_one_marginals(n_agents, n_items):
    agent_demands = [1.0 for _ in range(n_agents)]
    item_supplies = [1.0 for _ in range(n_items)]
    if n_agents <= n_items:
        agent_demands.append((n_items - n_agents))
        item_supplies.append(0.0)
    else:
        raise ValueError("There must be least as many items as agents")
    
    assert sum(agent_demands) == sum(item_supplies)
    
    return torch.tensor(agent_demands).float(), torch.tensor(item_supplies).float()


def compute_sinkhorn_max_error(plan: torch.Tensor, a: torch.Tensor, b: torch.Tensor, tol: float) -> float:
    a_max_err = torch.max(torch.abs(plan.sum(dim=-1) - a) / a).item()
    b_max_err = torch.max(torch.abs(plan.sum(dim=-2) - b) / b).item()

    return max(a_max_err, b_max_err)


def sinkhorn_error(dist_mat, f, g, a, epsilon):
    plan_marginals = torch.exp((-dist_mat + f[..., None] + g[..., None, :]) / epsilon).sum(dim=-1)
    return torch.max(torch.abs(plan_marginals - a) / a).item()


def sinkhorn_eps_scale(dist_mat, a, b, start_eps=1.0, end_eps=1e-1, eps_steps=10, tol=1e-1):
    v = torch.ones_like(b)
    g = start_eps * torch.log(v)
    f = torch.zeros_like(a)

    epsilon_vals = np.linspace(start_eps, end_eps, eps_steps)
    total_iters = 0
    for epsilon in epsilon_vals:
        with torch.no_grad():
            err = sinkhorn_error(dist_mat, f, g, a, epsilon)
        iters = 0
        while err >= tol:
            f = -epsilon * torch.logsumexp(-(dist_mat - g[..., None, :]) / epsilon, dim=-1) + \
                epsilon * torch.log(a)
            g = -epsilon * torch.logsumexp(-(dist_mat - f[..., None]) / epsilon, dim=-2) + \
                epsilon * torch.log(b)
            with torch.no_grad():
                err = sinkhorn_error(dist_mat, f, g, a, epsilon)
            iters += 1
            total_iters += 1

        logging.info(f"scaled sinkhorn with {epsilon} took {iters} iterations to hit tolerance of {tol} on batch of size {dist_mat.shape[0]}")

    logging.info(
        f"scaled sinkhorn with end eps {end_eps} took {total_iters} in total to hit {tol} on batch of size {dist_mat.shape[0]}")
    return torch.exp((-dist_mat + f[..., None] + g[..., None, :]) / end_eps)


def log_sinkhorn_plan_tolerance(dist_mat, a, b, epsilon=1e-1, tol=3):
    v = torch.ones_like(b)
    g = epsilon * torch.log(v)
    f = torch.zeros_like(a)
    err = sinkhorn_error(dist_mat, f, g, a, epsilon)
    iters = 0
    while err >= tol:
        f = -epsilon * torch.logsumexp(-(dist_mat - g[..., None, :]) / epsilon, dim=-1) + \
            epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp(-(dist_mat - f[..., None]) / epsilon, dim=-2) + \
            epsilon * torch.log(b)
        with torch.no_grad():
            err = sinkhorn_error(dist_mat, f, g, a, epsilon)
        iters += 1

    logging.info(f"sinkhorn took {iters} iterations to hit tolerance of {tol} on batch of size {dist_mat.shape[0]}")

    return torch.exp((-dist_mat + f[..., None] + g[..., None, :]) / epsilon)


def log_sinkhorn_plan(dist_mat, a, b, epsilon=1e-1, rounds=3):
    v = torch.ones_like(b)
    g = epsilon * torch.log(v)
    for i in range(rounds):
        f = -epsilon * torch.logsumexp(-(dist_mat - g[..., None, :]) / epsilon, dim=-1) + \
            epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp(-(dist_mat - f[..., None]) / epsilon, dim=-2) + \
            epsilon * torch.log(b)

    return torch.exp((-dist_mat + f[..., None] + g[..., None, :]) / epsilon)


def generate_marginals(n_agents, n_items):
    main_agents = [1.0 for _ in range(n_agents)]
    main_items = [1.0 for _ in range(n_items)]
    main_agents.append(float(n_items))
    main_items.append(float(n_agents))

    assert sum(main_agents) == sum(main_items)

    return torch.tensor(main_agents), torch.tensor(main_items)
