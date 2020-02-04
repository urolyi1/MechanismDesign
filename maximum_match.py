import cvxpy as cp


def cvxpy_max_matching(S_matrix, w, b, z, control_strength):
    n_types = S_matrix.shape[0]
    n_structures = S_matrix.shape[1]
    x1 = cp.Variable(n_structures, integer=True)
    _s = cp.Parameter((n_types, n_structures))  # valid structures
    _w = cp.Parameter(n_structures)  # structure weight
    _z = cp.Parameter(n_structures)  # control parameter
    _b = cp.Parameter(n_types)  # max bid

    constraints = [x1 >= 0, S_matrix @ x1 <= b]  # constraint for positive allocation and less than true bid
    objective = cp.Maximize((w.T @ x1) - control_strength * cp.norm(x1 - z, 1))
    problem = cp.Problem(objective, constraints)
    _s.value = S_matrix
    _w.value = w
    _z.value = z
    _b.value = b
    problem.solve(solver=cp.GUROBI)
    return x1.value
