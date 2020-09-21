import torch

from match_net import util


def test_combine_misreports_dimensions():
    truthful_bids = torch.tensor([
        [3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
        [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]
    ])

    # All possible misreports from each hospital
    p1_misreports = torch.tensor(util.all_possible_misreports(truthful_bids[0, :].numpy()))
    p2_misreports = torch.tensor(util.all_possible_misreports(truthful_bids[1, :].numpy()))

    batched_misreports = util.combine_misreports([p1_misreports, p2_misreports], truthful_bids)
    assert batched_misreports.shape[0] == max(p1_misreports.shape[0], p2_misreports.shape[0])
    assert batched_misreports.shape[1] == truthful_bids.shape[0]
    assert batched_misreports.shape[2] == truthful_bids.shape[1]


def test_all_possible_misreports_dimensions():
    truthful_bids = torch.tensor([
        [3.0000, 0.0000, 0.0000, 3.0000, 3.0000, 3.0000, 0.0000],
        [0.0000, 3.0, 3.0000, 0.0000, 0.0000, 0.0000, 3.0000]
    ])

    p1_misreports = torch.tensor(util.all_possible_misreports(truthful_bids[0, :].numpy()))
    assert p1_misreports.shape[0] == 256  # 4^4 possible misreports
    assert p1_misreports.shape[1] == truthful_bids.shape[1]  # Ensure type dimensions align


