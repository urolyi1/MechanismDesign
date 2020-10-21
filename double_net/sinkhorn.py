import torch

def sinkhorn_plan(dist_mat, a, b, epsilon=1e-1, rounds=2):
    K = torch.exp(-dist_mat / epsilon)
    v = torch.ones_like(b)
    v /= v.sum(dim=-1)
    u = a / (K @ v)
    v = b / (K.t() @ u)


    for i in range(rounds):
        u = a / (K @ v)
        v = b / (K.t() @ u)
    
    return torch.diag(u) @ K @ torch.diag(v)
