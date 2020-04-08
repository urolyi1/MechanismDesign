import itertools

import numpy as np
import torch


def convert_internal_S(single_S, num_hospitals):
    """
    Blowup a single S matrix with all possible structures between just pairs to be all possible structures
    from hospital and pair.

    :param single_S: structures matrix between just pairs
    :param num_hospitals: number of hospitals in exchange
    :return: Large structure matrix with all possible structures accounting for hospital and type
    """
    all_cols = []
    # Blow up all the columns in single structure matrix
    for col in range(single_S.shape[1]):
        all_cols.append(blow_up_column(single_S[:, col], num_hospitals))
    return np.concatenate(all_cols, axis=1)


def blow_up_column(col_vector, num_hospitals):
    """
    Takes a single structure vector and blows it up to all possible version of that structure specifying hospital

    :param col_vector: vector of a possible structure within the S matrix
    :param num_hospitals: number of hospitals to blow up column to
    :return: tensor of all possible version of structure with specific hospitals
    """
    # find which pairs are in the structure
    (nonzero_inds,) = np.nonzero(col_vector)
    all_inds = []
    for ind in nonzero_inds:
        count = int(col_vector[ind])
        all_inds.extend(count * [ind])
    num_types = len(col_vector)
    all_hospitals = list(range(num_hospitals))
    num_outcomes = len(all_inds)
    new_columns = []
    for hospital_outcome in itertools.product(all_hospitals, repeat=num_outcomes):
        new_column = np.zeros(num_types * num_hospitals)
        for i, ind in enumerate(all_inds):
            curr_ind_hospital = hospital_outcome[i]
            new_column[(curr_ind_hospital * num_types) + ind] += 1.0
        new_columns.append(new_column)
    return np.stack(new_columns).transpose()


def all_possible_misreports(true_bid):
    # gives iterators from 0 to max
    all_iters = [range(int(i) + 1) for i in true_bid]
    results = [np.array(x, dtype=np.float32) for x in itertools.product(*all_iters)]
    return results


def internal_central_bloodtypes(num_hospitals):
    """
    :param num_hospitals: number of hospitals involved
    :return: tuple of internal structure matrix and full structure matrix

    "O-O","O-B","O-AB","O-A","B-O","B-B","B-AB","B-A","AB-O","AB-B","AB-AB","AB-A","A-O","A-B","A-AB","A-A"

    """
    internal_s = np.load("bloodtypematrix.npy")
    central_s = convert_internal_S(internal_s, num_hospitals)
    return (
        torch.tensor(internal_s, dtype=torch.float32, requires_grad=False),
        torch.tensor(central_s, dtype=torch.float32, requires_grad=False),
    )
