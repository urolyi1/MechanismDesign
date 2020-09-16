import itertools

import numpy as np
import torch

def find_internal_two_cycles(S, n_hos, n_types):
    """
    Given a structures matrix finds the indices of internal structures
    """

    # iterate over all columns
    ind_list = []
    for col in range(S.shape[1]):
        for h in range(n_hos):
            internal_count = 0
            for t in range(n_types):
                if S[h * n_types + t, col] == 1:
                    internal_count += 1
            if internal_count > 1:
                ind_list.append(col)
                break
    return ind_list


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


def create_train_sample(generator, num_batches, batch_size=16):
    """
    Generate num_batches batches and stack them into a single tensor

    :param generator: hospital true bid generator
    :param num_batches: number of batches to generate
    :return: tensor of batches [num_batches, batch_size, n_hos, n_types]
    """
    batches = []
    for i in range(num_batches):
        batches.append(torch.tensor(next(generator.generate_report(batch_size))).float())
    return torch.stack(batches, dim=0)


def create_individual_weights(central_s, internal_weight_value, num_structures, N_HOS, N_TYP):
    """
    Create matrix of value of each structure for each hospital.

    :return: tensor of value of structure split by hospital
    """

    individual_weights = torch.zeros(num_structures, N_HOS)
    for h in range(N_HOS):
        for col in range(num_structures):
            # Check how many allocated pairs
            allocated = central_s[h * N_TYP: (h + 1) * N_TYP, col].sum().item()

            # Since only two-cycles this means it is an internal match
            if allocated > 1:
                allocated = internal_weight_value
            individual_weights[col, h] = allocated
    return individual_weights

def full_regret_check(model, test_batches, verbose=False):
    """For each sample given batches checks all possible misreports for regret

    :param model: MatchNet object
    :param test_batches: test_samples
    :param verbose: boolean option for verbose print output
    :return: None
    """
    high_regrets = []
    for batch in range(test_batches.shape[0]):
        for sample in range(test_batches.shape[1]):
            if all_misreport_regret(model, test_batches[batch, sample, :, :], verbose):
                high_regrets.append(test_batches[batch, sample, :, :])
    return high_regrets


def all_misreport_regret(model, truthful_bids, verbose=False, tolerance=1e-2):
    """Given truthful inputs checks checks all possible misreports and determins whether regret
    is above the threshold for any of the misreports

    :param model: MatchNet model
    :param truthful_bids: tensor of truthful bids
    :param verbose: boolean to print when high regret misreport if found
    :param tolerance: threshold value that distinguishes negligible regret from high regret
    :return:
    """

    # params
    N_HOS = model.n_hos
    N_TYP = model.n_types

    # All possible misreports from each hospital
    p1_misreports = torch.tensor(all_possible_misreports(truthful_bids[0, :].numpy()))
    p2_misreports = torch.tensor(all_possible_misreports(truthful_bids[1, :].numpy()))

    if p1_misreports.shape[0] > p2_misreports.shape[0]:
        to_pad = truthful_bids[1, :].repeat(p1_misreports.shape[0] - p2_misreports.shape[0], 1)
        p2_misreports = torch.cat((p2_misreports, to_pad))

    elif p2_misreports.shape[0] > p1_misreports.shape[0]:
        to_pad = truthful_bids[0, :].repeat(p2_misreports.shape[0] - p1_misreports.shape[0], 1)
        p1_misreports = torch.cat((p1_misreports, to_pad))

    # Combine all possible misreports
    batched_misreports = torch.cat((p1_misreports.unsqueeze(1), p2_misreports.unsqueeze(1)), dim=1)

    # given batched misreports, we want to calc mis util for each one against truthful bids
    self_mask = torch.zeros(N_HOS, 1, N_HOS, N_TYP)
    self_mask[np.arange(N_HOS), :, np.arange(N_HOS), :] = 1.0
    mis_mask = torch.zeros(N_HOS, 1, N_HOS)
    mis_mask[np.arange(N_HOS), :, np.arange(N_HOS)] = 1.0

    high_regret_misreports = []

    # For every possible misreport check regret
    for batch_ind in range(batched_misreports.shape[0]):

        # Get specific misreport and run through model
        curr_mis = batched_misreports[batch_ind, :, :].unsqueeze(0)
        mis_input = model.create_combined_misreport(curr_mis, truthful_bids)
        output = model.forward(mis_input, 1 * model.n_hos)
        p = truthful_bids.unsqueeze(0)
        mis_util = model.calc_mis_util(output, p)

        central_util, internal_util = model.calc_util(model.forward(p, 1), p)
        pos_regret = torch.clamp(mis_util - (central_util + internal_util), min=0)

        if verbose and (pos_regret > tolerance).any().item():
            print("Misreport: ", curr_mis)
            print("Regret: ", pos_regret)
            high_regret_misreports.append(curr_mis)

    if verbose:
        print('found large positive regret: ', len(high_regret_misreports) > 0)

    return high_regret_misreports

