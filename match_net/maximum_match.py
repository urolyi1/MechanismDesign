import cvxpy as cp
import numpy as np
import match_net.HospitalGenerators as hg


def is_blood_compat(p_type, d_type):
    """

    :param p_type: patient blood type e.g "A" "B" "AB"
    :param d_type: donor blood type
    :return: boolean
    """
    for ch in d_type:
        if ch not in p_type:
            return False
    return True


def calc_edges_AB(truthful_bids, compat_dict):
    """
    Calculates the total number of edges.

    :param truthful_bids: the true number of each type hospitals have [n_hos, n_types]
    :return: total number of edges in the graph
    """
    blood_types = ['', 'B', 'AB', 'A']

    type_tot = truthful_bids.sum(axis=0)

    n_edges = 0
    for p_idx in range(4):
        for d_idx in range(4):
            patient = blood_types[p_idx]
            donor = blood_types[d_idx]

            # Check all possible other pairs
            for p, other_p in enumerate(blood_types):
                for d, other_d in enumerate(blood_types):

                    # If compatible with other pair
                    if is_blood_compat(patient, other_d) and is_blood_compat(other_p, donor):

                        # Self compatible pairs
                        if p_idx == p and d == d_idx:
                            n_edges += type_tot[4 * p_idx + d_idx] * (type_tot[4 * p_idx + d_idx] - 1)
                        # Otherwise
                        else:
                            n_edges += type_tot[4 * p + d] * type_tot[4 * p_idx + d_idx]

    return n_edges / 2.0

def calc_n_edges(bids, compat_dict):
    """
    bids: [n_hos, n_types]
    """
    n_hos = bids.shape[0]
    n_types = bids.shape[1]
    type_totals = bids.sum(axis=0)
    n_edges = 0
    for t in range(n_types):
        num_patients = type_totals[t]
        compat_types = compat_dict[t]
        # compatible types
        for compat_t in compat_types:
            if t == compat_t:
                n_edges += type_totals[t] * (type_totals[t] - 1)
            else:
                n_edges += type_totals[t] * type_totals[compat_t]
    return n_edges / 2.0

def create_match_weights(central_S, batch, compat_dict, calc_edges=calc_n_edges):
    """

    :param central_S: S matrix for all structures and hospitals
    :param batch: batch of bids [batch_size, n_hos, n_types
    :return: The weights matrix for each batch [batch_size, n_structure]
    """
    # Parameters
    batch_size = batch.shape[0]
    n_structs = central_S.shape[1]
    n_hos = batch.shape[1]
    n_types = int(central_S.shape[0] / n_hos)

    weights_batch = np.zeros((batch_size, n_structs))
    # for each sample in batch
    for i in range(batch_size):
        w = np.zeros(n_structs)
        n_edges = calc_edges(batch[i, :, :], compat_dict)
        internal_weight = n_edges + 3
        external_weight = 1 + 1 / (n_edges ** 2) + 1 / (n_edges ** (3)) * 1 / (n_edges ** (n_hos + 1))
        # for each structure in s matrix
        for col in range(n_structs):
            struct = central_S[:, col].numpy().reshape((n_hos, n_types))
            nonzero_inds = struct.nonzero()
            if len(nonzero_inds[0]) == 2:
                hos0 = nonzero_inds[0][0]
                hos1 = nonzero_inds[0][1]
            else:
                hos0 = nonzero_inds[0][0]
                hos1 = hos0

            # TODO: Add zero weight for crossing the bipartition.
            # Internal edge
            if hos0 == hos1:
                w[col] = internal_weight
            # External edge
            else:
                w[col] = external_weight
        weights_batch[i] = w
    return weights_batch

def cvxpy_max_matching(S_matrix, w, b, z, control_strength):
    """

    :param S_matrix: Structures matrix [n_types, n_structures]
    :param w: weights matrix [1, n_structures]
    :param b:
    :param z:
    :param control_strength:
    :return:
    """
    n_types = S_matrix.shape[0]
    n_structures = S_matrix.shape[1]
    x1 = cp.Variable(n_structures, integer=True)  # [n_structures, 1]
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

def compute_max_matching(S_matrix, w, b):
    n_types = S_matrix.shape[0]
    n_structures = S_matrix.shape[1]
    x1 = cp.Variable(n_structures, integer=True)  # [n_structures, 1]
    _s = cp.Parameter((n_types, n_structures))  # valid structures
    _w = cp.Parameter(n_structures)  # structure weight
    _b = cp.Parameter(n_types)  # max bid

    constraints = [x1 >= 0, S_matrix.numpy() @ x1 <= b]  # constraint for positive allocation and less than true bid
    objective = cp.Maximize((w.T @ x1))
    problem = cp.Problem(objective, constraints)
    _s.value = S_matrix.numpy()
    _w.value = w
    _b.value = b.numpy()
    problem.solve(solver=cp.GUROBI)
    return x1.value

if __name__ == "main":
    # Constructing weights matrix
    central_S = np.load('two_hos_two_cycle_bloodtype_matrix.npy')

    hos_gen_lst = [hg.RealisticHospital(100), hg.RealisticHospital(100)]
    generator = hg.ReportGenerator(hos_gen_lst, (2, 16))
    batch = next(generator.generate_report(4))
    w_batch = create_match_weights(central_S, batch)
    #for i in range(batch.shape[0]):
       # cvxpy_max_matching(central_S, w_batch[i, :], batch[i], )



