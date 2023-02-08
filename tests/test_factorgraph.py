import pytest
from scalib.attacks import FactorGraph, BPState
import numpy as np
import os
import copy


def normalize_distr(x):
    return x / x.sum(axis=-1, keepdims=True)


def make_distri(nc, n):
    return normalize_distr(np.random.randint(1, 10000000, (n, nc)).astype(np.float64))


def test_table():
    """
    Test Table lookup
    """
    nc = 16
    n = 100
    table = np.random.permutation(nc).astype(np.uint32)
    distri_x = np.random.randint(1, 2048, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T

    graph = f"""
            PROPERTY y = table[x]
            TABLE table
            VAR MULTI x
            VAR MULTI y
            NC {nc}
            """
    graph = FactorGraph(graph, {"table": table})
    x = distri_x.argmax(axis=1)
    graph.sanity_check({}, {"x": x, "y": table[x]})
    bp_state = BPState(graph, n)

    bp_state.set_evidence("x", distri_x)

    bp_state.bp_loopy(1, True)
    distri_y = bp_state.get_distribution("y")

    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = table[x]
        distri_y_ref[:, y] = distri_x[:, x]

    assert np.allclose(distri_y_ref, distri_y)


def test_table_non_bij():
    """
    Test non-bijective Table lookup
    """
    nc = 2
    n = 1
    table = np.array([0, 0], dtype=np.uint32)
    distri_x = np.array([[0.5, 0.5]])
    distri_y = np.array([[0.8, 0.2]])

    graph = f"""
            PROPERTY y = table[x]
            TABLE table
            VAR MULTI x
            VAR MULTI y
            NC {nc}
            """
    graph = FactorGraph(graph, {"table": table})

    graph.sanity_check({}, {"x": np.array([0, 1]), "y": np.array([0, 0])})

    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)

    bp_state.bp_loopy(1, True)
    distri_x = bp_state.get_distribution("x")
    distri_y = bp_state.get_distribution("y")

    distri_x_ref = np.array([0.5, 0.5])
    distri_y_ref = np.array([1.0, 0.0])

    assert np.allclose(distri_x_ref, distri_x)
    assert np.allclose(distri_y_ref, distri_y)


def test_not():
    """
    Test NOT operation
    """
    nc = 2
    n = 1
    distri_x = np.array([[0.6, 0.4]])
    distri_y = np.array([[0.8, 0.2]])

    graph = f"""
            PROPERTY y = !x
            VAR MULTI x
            VAR MULTI y
            NC {nc}
            """
    graph = FactorGraph(graph)

    graph.sanity_check({}, {"x": np.array([0, 1]), "y": np.array([1, 0])})

    bp_state = BPState(graph, n)

    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)

    bp_state.bp_loopy(1, True)
    distri_x_bp = bp_state.get_distribution("x")
    distri_y_bp = bp_state.get_distribution("y")

    t = np.array([1, 0])
    distri_x_ref = normalize_distr(distri_x * distri_y[0, t][np.newaxis, :])
    distri_y_ref = normalize_distr(distri_y * distri_x[0, t][np.newaxis, :])

    assert np.allclose(distri_x_ref, distri_x_bp)
    assert np.allclose(distri_y_ref, distri_y_bp)


def test_and_public():
    """
    Test AND with public data
    """
    nc = 16
    n = 100
    public = np.random.randint(0, nc, n, dtype=np.uint32)
    distri_x = np.random.randint(1, 100, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY y = x & p
        VAR MULTI y
        VAR MULTI x
        PUB MULTI p#come comments
        """
    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"p": public})
    bp_state.set_evidence("x", distri_x)

    bp_state.bp_loopy(2, False)

    distri_y = bp_state.get_distribution("y")
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = x & public
        distri_y_ref[np.arange(n), y] += distri_x[np.arange(n), x]

    assert np.allclose(distri_y_ref, distri_y)

    bp_state2 = BPState(graph, n, {"p": public})
    bp_state2.set_evidence("x", distri_x)
    bp_state2.bp_acyclic("y")
    assert np.allclose(distri_y_ref, bp_state2.get_distribution("y"))


def test_xor_public():
    """
    Test XOR with public data
    """
    nc = 16
    n = 100
    public = np.random.randint(0, nc, n, dtype=np.uint32)
    public2 = np.random.randint(0, nc, n, dtype=np.uint32)
    distri_x = np.random.randint(1, 100, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T

    graph = f"""
        PROPERTY y = x ^ p ^ p2
        VAR MULTI y
        VAR MULTI x
        PUB MULTI p
        PUB MULTI p2
        NC {nc}
        """
    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"p": public, "p2": public2})
    bp_state.set_evidence("x", distri_x)

    bp_state.bp_loopy(2, False)

    distri_y = bp_state.get_distribution("y")
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = x ^ public ^ public2
        distri_y_ref[np.arange(n), y] = distri_x[np.arange(n), x]

    distri_x = bp_state.get_distribution("x")

    print("p ^ p2")
    print(public ^ public2)
    print("y_ref")
    print(distri_y_ref)
    print("y")
    print(distri_y)
    print("x")
    print(distri_x)
    assert distri_x is not None
    assert distri_y is not None
    assert np.allclose(distri_y_ref, distri_y)


def test_AND():
    """
    Test AND between distributions
    """
    cases = [
        (
            np.array([[0.5, 0.5]]),  # uniform x
            np.array([[0.0, 1.0]]),  # y == 1
            np.array([[0.0, 1.0]]),  # z == 1
        ),
        (
            np.array([[0.1, 0.9]]),  # x == 1
            np.array([[0.5, 0.5]]),  # uniform y
            np.array([[1.0, 0.0]]),  # z == 1
        ),
        (
            np.array([[0.1, 0.9]]),  # x == 1
            np.array([[0.4, 0.6]]),  # uniform y
            np.array([[0.8, 0.2]]),  # z == 1
        ),
        (make_distri(2, 1) for _ in range(3)),
        (make_distri(4, 1) for _ in range(3)),
        (make_distri(256, 4) for _ in range(3)),
    ]

    for distri_x, distri_y, distri_z in cases:
        print("#### Test case:")
        print(distri_x)
        print(distri_y)
        print(distri_z)
        n, nc = distri_x.shape
        graph = f"""
            # some comments
            NC {nc}
            PROPERTY z = x&y
            VAR MULTI z
            VAR MULTI x
            VAR MULTI y

            """
        graph = FactorGraph(graph)
        bp_state = BPState(graph, n)
        bp_state.set_evidence("x", distri_x)
        bp_state.set_evidence("y", distri_y)
        bp_state.set_evidence("z", distri_z)

        distri_x_ref = np.zeros(distri_x.shape)
        distri_y_ref = np.zeros(distri_y.shape)
        distri_z_ref = np.zeros(distri_z.shape)

        for x in range(nc):
            for y in range(nc):
                distri_x_ref[:, x] += distri_z[:, x & y] * distri_y[:, y]
                distri_y_ref[:, y] += distri_z[:, x & y] * distri_x[:, x]
                distri_z_ref[:, x & y] += distri_x[:, x] * distri_y[:, y]

        distri_x_ref = normalize_distr(distri_x_ref * distri_x)
        distri_y_ref = normalize_distr(distri_y_ref * distri_y)
        distri_z_ref = normalize_distr(distri_z_ref * distri_z)

        print("#### Ref:")
        print(distri_x_ref)
        print(distri_y_ref)
        print(distri_z_ref)

        bp_state.bp_loopy(1, True)
        distri_x = bp_state.get_distribution("x")
        distri_y = bp_state.get_distribution("y")
        distri_z = bp_state.get_distribution("z")

        print("#### Got:")
        print(distri_x)
        print(distri_y)
        print(distri_z)

        assert np.allclose(distri_z_ref, distri_z)
        assert np.allclose(distri_x_ref, distri_x)
        assert np.allclose(distri_y_ref, distri_y)


def test_and_not_or():
    nc = 16
    n = 100
    nc = 2
    n = 1
    p = np.random.randint(0, nc, n, dtype=np.uint32)
    t = np.random.randint(0, nc, n, dtype=np.uint32)
    nt = (nc - 1) ^ t
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)
    distri_z = make_distri(nc, n)

    graph = f"""
        NC {nc}
        VAR MULTI x
        VAR MULTI y
        VAR MULTI z
        PUB MULTI p
        PUB MULTI t
        PROPERTY z = !y & x & p & !t
        """
    graph2 = f"""
        NC {nc}
        VAR MULTI x
        VAR MULTI y
        VAR MULTI ny
        PROPERTY ny = !y
        VAR MULTI z
        VAR MULTI nz
        PROPERTY z = !nz
        PUB MULTI p
        PUB MULTI nt
        VAR MULTI a
        VAR MULTI na
        PROPERTY na = !a
        VAR MULTI b
        VAR MULTI nb
        PROPERTY nb = !b
        PROPERTY a = x & p
        PROPERTY b = nt & ny 
        PROPERTY nz = na | nb 
        """
    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"p": p, "t": t})
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)
    bp_state.set_evidence("z", distri_z)
    bp_state.bp_loopy(1, True)
    import sys

    print("BP2", file=sys.stderr)

    graph2 = FactorGraph(graph2)
    bp_state2 = BPState(graph2, n, {"p": p, "nt": nt})
    bp_state2.set_evidence("x", distri_x)
    bp_state2.set_evidence("y", distri_y)
    bp_state2.set_evidence("z", distri_z)
    bp_state2.bp_loopy(8, False)

    for v in ["x", "y", "z"]:
        print(v)
        d = bp_state.get_distribution(v)
        d2 = bp_state2.get_distribution(v)
        assert np.allclose(d, d2)

    bp_state3 = BPState(graph, n, {"p": p, "t": t})
    bp_state3.set_evidence("x", distri_x)
    bp_state3.set_evidence("y", distri_y)
    bp_state3.set_evidence("z", distri_z)
    for v in ["x", "y", "z"]:
        print(v)
        bp_state3.bp_acyclic(v)
        assert np.allclose(bp_state2.get_distribution(v), bp_state3.get_distribution(v))


def test_ADD():
    """
    Test ADD between distributions
    """
    nc = 251
    n = 4
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x+y
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y

        """

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)

    bp_state.bp_loopy(1, True)
    distri_z = bp_state.get_distribution("z")

    distri_z_ref = np.zeros(distri_z.shape)
    msg = np.zeros(distri_z.shape)

    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:, (x + y) % nc] += distri_x[:, x] * distri_y[:, y]

    distri_z_ref = (distri_z_ref.T / np.sum(distri_z_ref, axis=1)).T
    assert np.allclose(distri_z_ref, distri_z)


def test_add_cst():
    """
    Test ADD of a constant
    """
    nc = 251
    y_cases = [0, 1, 2, 58, 249, 250, 251, 2345]
    n = len(y_cases)
    distri_x = make_distri(nc, n)

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x+y+w
        VAR MULTI z
        VAR MULTI x
        PUB MULTI y
        PUB SINGLE w
        """
    y_cases = np.array(y_cases, dtype=np.uint32)
    graph = FactorGraph(graph)
    w = 5
    for evidence_var, target_var, sub in [("x", "z", False), ("z", "x", True)]:
        print("sub", sub)
        bp_state = BPState(graph, n, {"y": y_cases, "w": w})
        bp_state.set_evidence(evidence_var, distri_x)

        bp_state.bp_loopy(1, True)
        distri_z = bp_state.get_distribution(target_var)

        distri_z_ref = np.zeros(distri_z.shape)

        for x in range(nc):
            for j in range(n):
                if sub:
                    new_idx = (x - y_cases[j] - w) % nc
                else:
                    new_idx = (x + y_cases[j] + w) % nc
                distri_z_ref[j, new_idx] = distri_x[j, x]

        assert np.allclose(distri_z_ref, distri_z)


def test_ADD_multiple():
    """
    Test ADD between distributions
    """
    nc = 17
    n = 4
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)
    distri_w = make_distri(nc, n)

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x+y+w
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y
        VAR MULTI w
        """

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)
    bp_state.set_evidence("w", distri_w)

    bp_state.bp_loopy(1, True)
    distri_z = bp_state.get_distribution("z")

    distri_z_ref = np.zeros(distri_z.shape)
    msg = np.zeros(distri_z.shape)

    for x in range(nc):
        for y in range(nc):
            for w in range(nc):
                distri_z_ref[:, (x + y + w) % nc] += (
                    distri_x[:, x] * distri_y[:, y] * distri_w[:, w]
                )

    distri_z_ref = (distri_z_ref.T / np.sum(distri_z_ref, axis=1)).T
    assert np.allclose(distri_z_ref, distri_z)


def test_MUL():
    """
    Test MUL between distributions
    """
    nc = 251
    n = 4
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x*y
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y

        """

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)

    bp_state.bp_loopy(1, True)
    distri_z = bp_state.get_distribution("z")

    distri_z_ref = np.zeros(distri_z.shape)
    msg = np.zeros(distri_z.shape)

    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:, (x * y) % nc] += distri_x[:, x] * distri_y[:, y]

    distri_z_ref = (distri_z_ref.T / np.sum(distri_z_ref, axis=1)).T
    assert np.allclose(distri_z_ref, distri_z)


def test_xor():
    """
    Test XOR between distributions
    """
    nc = 512
    n = 10
    distri_x = np.random.randint(1, 10000000, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T
    distri_y = np.random.randint(1, 10000000, (n, nc))
    distri_y = (distri_y.T / np.sum(distri_y, axis=1)).T

    distri_b = np.random.randint(1, 10000000, (n, nc))
    distri_b[1, 0] = 1.0
    distri_b[1, 1] = 0.0
    distri_b = (distri_b.T / np.sum(distri_b, axis=1)).T
    distri_a = np.random.randint(1, 10000000, (n, nc))
    distri_a[1, :] = 1.0
    distri_a = (distri_a.T / np.sum(distri_a, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY s2: z = x^y
        PROPERTY s1: x = a^b
        VAR MULTI x # some comments too
        VAR MULTI y
        VAR MULTI a
        VAR MULTI b
        VAR SINGLE z"""

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)
    bp_state.set_evidence("a", distri_a)
    bp_state.set_evidence("b", distri_b)

    bp_state2 = copy.deepcopy(bp_state)

    # Custom optimized sequence -- equivelent to non-loopy algo, but focusing only on z.
    bp_state.propagate_var("a")
    bp_state.propagate_var("b")
    bp_state.propagate_var("x")
    bp_state.propagate_factor("s1")
    bp_state.propagate_var("x")
    bp_state.propagate_var("y")
    bp_state.propagate_factor("s2")
    bp_state.propagate_var("z")

    bp_state2.bp_loopy(2, True)

    distri_z = bp_state.get_distribution("z")
    distri_z2 = bp_state2.get_distribution("z")

    msg = np.zeros((n, nc))
    distri_z_ref_multi = np.zeros((n, nc))
    distri_z_ref = np.ones(distri_z.shape)

    for a in range(nc):
        for b in range(nc):
            msg[:, a ^ b] += distri_a[:, a] * distri_b[:, b]

    distri_x *= msg
    for x in range(nc):
        for y in range(nc):
            distri_z_ref_multi[:, x ^ y] += distri_x[:, x] * distri_y[:, y]
    for d in distri_z_ref_multi:
        print(d)
        distri_z_ref *= d

    print("distri_x_ref", distri_x)
    print("distri_y_ref", distri_y)
    print("distri_z_ref", distri_z_ref)

    distri_z_ref = distri_z_ref / np.sum(distri_z_ref)
    assert np.allclose(distri_z_ref, distri_z)
    assert np.allclose(distri_z_ref, distri_z2)


def test_bad_var_norm():
    """
    When normalization of distributions can be bad...
    Root cause of bug #55.
    """
    nc = 2
    n = 1
    dy = np.array([[1.0, 1e-100]])
    dz = np.array([[1e-100, 1.0]])
    graph = f"""
        NC {nc}
        VAR MULTI x
        VAR MULTI y
        VAR MULTI z
        PUB MULTI a
        PROPERTY s1: x = y^a
        PROPERTY s2: x = !z
        """

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"a": [0]})
    bp_state.set_evidence("y", dy)
    bp_state.set_evidence("z", dz)

    bp_state.propagate_var("y")
    bp_state.propagate_var("z")
    bp_state.propagate_factor("s1")
    bp_state.propagate_factor("s2")
    bp_state.propagate_var("x")

    s1 = normalize_distr(bp_state.get_belief_from_var("x", "s1"))
    s2 = normalize_distr(bp_state.get_belief_from_var("x", "s2"))

    assert np.allclose(np.array([[1.0, 1e-40]]), s1)
    assert np.allclose(np.array([[1.0, 1e-40]]), s2)


def test_cyclic():
    graph_never_cyclic = """
    NC 2
    VAR MULTI x
    VAR MULTI x0
    VAR MULTI x1
    VAR MULTI y
    VAR MULTI y0
    VAR MULTI y1
    PUB MULTI p
    VAR SINGLE k
    VAR SINGLE k2

    PROPERTY x = p ^ k
    PROPERTY y = y0 ^ y1
    PROPERTY x = x0 ^ x1
    PROPERTY k2  = p ^ y
    """
    g = FactorGraph(graph_never_cyclic)
    assert BPState(g, 1, {"p": np.array([0], dtype=np.uint32)}).is_cyclic() == False
    assert BPState(g, 2, {"p": np.array([0, 0], dtype=np.uint32)}).is_cyclic() == False
    graph_always_cyclic = """
    NC 2
    VAR MULTI x
    VAR MULTI x0
    VAR MULTI x1
    VAR MULTI y
    VAR MULTI y0
    VAR MULTI y1
    PUB MULTI p
    VAR SINGLE k

    PROPERTY x = p ^ k
    PROPERTY y = y0 ^ y1
    PROPERTY x = x0 ^ x1
    PROPERTY y  = p ^ y1
    """
    g = FactorGraph(graph_always_cyclic)
    assert BPState(g, 1, {"p": np.array([0], dtype=np.uint32)}).is_cyclic() == True
    assert BPState(g, 2, {"p": np.array([0, 0], dtype=np.uint32)}).is_cyclic() == True
    graph_multi_cyclic = """
    NC 2
    VAR MULTI x
    VAR MULTI x0
    VAR MULTI x1
    VAR MULTI y
    VAR MULTI y0
    VAR MULTI y1
    PUB MULTI p
    VAR SINGLE k
    VAR SINGLE k2

    PROPERTY x = p ^ k
    PROPERTY y = y0 ^ y1
    PROPERTY x = x0 ^ x1
    PROPERTY k2  = x ^ y
    """
    g = FactorGraph(graph_multi_cyclic)
    assert BPState(g, 1, {"p": np.array([0], dtype=np.uint32)}).is_cyclic() == False
    assert BPState(g, 2, {"p": np.array([0, 0], dtype=np.uint32)}).is_cyclic() == True
