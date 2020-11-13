import torch


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


def log_sinkhorn_plan(dist_mat, a, b, epsilon, rounds):
    batch_size = dist_mat.shape[0]
    v = torch.ones_like(b)
    g = epsilon * torch.log(v)
    for i in range(rounds):
        f = -epsilon * torch.logsumexp(-(dist_mat - g.view(batch_size, 1, -1)) / epsilon, dim=-1) + \
            epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp(-(dist_mat - f.view(batch_size, -1, 1)) / epsilon, dim=1) + \
            epsilon * torch.log(b)

    return torch.exp((-dist_mat + f.view(batch_size, -1, 1) + g.view(batch_size, 1, -1)) / epsilon)


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
