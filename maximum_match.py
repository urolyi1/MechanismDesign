import gurobi as grb
import numpy as np
import torch

from match_net_torch import convert_internal_S


def gurobi_max_matching(S_matrix, w, b):
    n_types = S_matrix.shape[0]
    n_structures = S_matrix.shape[1]
    m = grb.Model("match")
    x_vars = [m.addVar(vtype=grb.GRB.INTEGER, lb=0, name=f"x_{i}") for i in range(n_structures)]
    row_sums = [grb.LinExpr(S_matrix[i,:], x_vars) for i in range(n_types)]
    m.setObjective(grb.quicksum([wi * xi for (wi, xi) in zip(w, x_vars)]), grb.GRB.MAXIMIZE)
    for i, row_sum in enumerate(row_sums):
        m.addConstr(row_sum <= b[i], name=f"rowsum_{i}")
    m.update()
    m.optimize()

    solns = [x.x for x in x_vars]
    return solns


if __name__ == '__main__':
    internal_s = torch.tensor([[1.0],
                               [1.0]], requires_grad=False)
    central_s = torch.tensor(convert_internal_S(internal_s.numpy(), 2), requires_grad = False, dtype=torch.float32)

    w = torch.ones(central_s.shape[1]).numpy()
    b = np.array([2.0, 1.0, 1.0, 2.0])
    result = gurobi_max_matching(central_s.numpy(), w, b)
    print(result)



