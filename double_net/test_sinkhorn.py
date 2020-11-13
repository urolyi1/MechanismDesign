from double_net.sinkhorn import generate_marginals_demands


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
