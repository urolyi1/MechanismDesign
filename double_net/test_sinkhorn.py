from double_net.sinkhorn import generate_marginals_demands, generate_marginals, sinkhorn_plan, log_sinkhorn_plan
import torch

def test_log_sinkhorn():
    batch_size = 10
    bids_mat = torch.rand(batch_size, 3, 4)
    a_marginals, b_marginals = generate_marginals(2, 3)


    plan = sinkhorn_plan(bids_mat, a_marginals.repeat(batch_size, 1), b_marginals.repeat(batch_size, 1), rounds=50, epsilon=1e-1)
    log_plan = log_sinkhorn_plan(bids_mat, a_marginals.repeat(batch_size, 1), b_marginals.repeat(batch_size, 1), epsilon=1e-1, rounds=50)

    assert torch.isclose(plan, log_plan, atol=1e-4).all()



def test_demand_list():
    a, b = generate_marginals_demands([1],[1,1])
    assert list(a) == [1.0,2.0]
    assert list(b) == [1.0,1.0,1.0]

    a, b = generate_marginals_demands([2],[1,1])
    assert list(a) == [2.0,1.0]
    assert list(b) == [1.0,1.0,1.0]


    a, b = generate_marginals_demands([3],[1,1])
    assert list(a) == [3.0,1.0]
    assert list(b) == [1.0,1.0,2.0]

    a, b = generate_marginals_demands([1],[3,3])
    assert list(a) == [1.0,6.0]
    assert list(b) == [3.0,3.0,1.0]
