import pytest
from scalib.attacks import FactorGraph, BPState
import numpy as np
import os
import copy


def normalize_distr(x):
    return x / x.sum(axis=-1, keepdims=True)


def make_distri(nc, n):
    return normalize_distr(np.random.randint(1, 10000000, (n, nc)).astype(np.float64))


def test_copy_fg():
    graph = """
        NC 2
        PROPERTY s1: x = a^b
        VAR MULTI x
        VAR MULTI a
        VAR MULTI b
        """
    graph = FactorGraph(graph)
    graph2 = copy.deepcopy(graph)


def test_copy_bp():
    graph = """
        NC 2
        PROPERTY s1: x = a^b
        VAR MULTI x
        VAR MULTI a
        VAR MULTI b
        """
    graph = FactorGraph(graph)
    n = 5
    bp_state = BPState(graph, n)
    distri_a = make_distri(2, 5)
    distri_b = make_distri(2, 5)
    bp_state.set_evidence("a", distri_a)
    bp_state.set_evidence("b", distri_b)
    bp_state.bp_loopy(1, initialize_states=True)
    bp_state2 = copy.deepcopy(bp_state)


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
    nc = 241
    n = 4
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY F0: z = x+y
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


def test_ADD2():
    """
    Test ADD between distributions
    """
    nc = 16
    n = 1
    distri_x = np.zeros((n, nc))  # make_distri(nc, n)
    distri_y = np.zeros((n, nc))  # make_distri(nc, n)

    distri_x[:, 0] = 1.0
    distri_y[:, 0] = 1.0

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY F: z = x+y
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y

        """

    graph = FactorGraph(graph)
    bp_state = BPState(graph, n)
    bp_state.set_evidence("x", distri_x)
    bp_state.set_evidence("y", distri_y)
    print(bp_state.debug())
    bp_state.bp_loopy(1, True)
    print(bp_state.debug())
    bp_state.bp_loopy(10, False)
    print(bp_state.debug())
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

    distri_z_ref = distri_z_ref / np.sum(distri_z_ref)
    print("distri_z_ref", distri_z_ref)
    assert np.allclose(distri_z_ref, distri_z)
    assert np.allclose(distri_z_ref, distri_z2)


def test_xor_acyclic():
    nc = 4
    fg = FactorGraph(
        f"""NC {nc}
    VAR MULTI x0
    VAR MULTI x4
    VAR MULTI y0
    PROPERTY F1: y0 = x0 ^ x4
    """
    )
    bp = BPState(fg, 1)
    for v in fg.vars():
        bp.set_evidence(v, make_distri(nc, 1))
    for v in ["x0", "x4"]:
        bp.bp_acyclic(v)
        bp.get_distribution(v)


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


def test_and_rounding_error_simple():
    # simple reproduction of issue #86
    factor_graph = """NC 16
    VAR MULTI A
    VAR MULTI B
    VAR MULTI C
    PROPERTY P: C = A & B
    """
    priors = {
        "A": [
            1.0 + 2**-52,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "C": [
            0.0,
            2.666666666666667,
            2.666666666666667,
            0.0,
            2.666666666666667,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }
    fg = FactorGraph(factor_graph)
    bp = BPState(fg, 1)
    for k, v in priors.items():
        bp._inner.set_belief_from_var(k, "P", np.array([v]))
    bp.propagate_factor("P")
    assert (bp.get_belief_to_var("B", "P") >= 0.0).all()


def test_and_rounding_error_simple2():
    # test case of issue #86
    factor_graph = """NC 16
    VAR MULTI K0
    VAR MULTI K1
    VAR MULTI L1
    VAR MULTI N1
    VAR MULTI B
    VAR MULTI C
    VAR MULTI N0
    VAR MULTI A
    VAR MULTI L2
    VAR MULTI L3
    PUB SINGLE IV
    PROPERTY P1: L1 = K0 ^ K1
    PROPERTY P2: N1 = !K1
    PROPERTY P3: B = N1 & IV
    PROPERTY P4: C = B ^ K0
    PROPERTY P5: N0 = !K0
    PROPERTY P6: A = N0 & K1
    PROPERTY P7: L2 = A ^ C
    PROPERTY P8: L3 = K1 ^ C"""
    priors = {
        "A": [
            0.0,
            0.25,
            0.25,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "C": [
            0.0,
            0.0,
            0.0,
            0.1666666667,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.1666666667,
            0.0,
            0.0,
            0.0,
        ],
        "L1": [
            0.0,
            0.0,
            0.0,
            0.1666666667,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.1666666667,
            0.0,
            0.0,
            0.0,
        ],
        "L2": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.25,
            0.25,
            0.0,
        ],
        "L3": [
            0.0,
            0.0,
            0.0,
            0.1666666667,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.0,
            0.1666666667,
            0.1666666667,
            0.0,
            0.1666666667,
            0.0,
            0.0,
            0.0,
        ],
    }
    fg = FactorGraph(factor_graph)
    bp = BPState(fg, 1, public_values={"IV": 0xC})
    for k, v in priors.items():
        bp.set_evidence(k, distribution=np.array([v]))
    bp.bp_loopy(5, initialize_states=False)
    assert (bp.get_distribution("K0") >= 0.0).all()
    assert (bp.get_distribution("K1") >= 0.0).all()


def test_manytraces():
    """No numerical underflow with many traces."""
    nc = 8
    n = 500
    distri_x = np.ones((n, nc))
    distri_x[:, 0] = 2.0
    distri_x[:, 1] = 1.5
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T

    graph = f"""
            NC {nc}
            VAR MULTI x
            VAR SINGLE y
            PUB SINGLE c
            PROPERTY y = x ^ c
            """
    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"c": 0})

    bp_state.set_evidence("x", distri_x)

    bp_state.bp_acyclic("y")
    distri_y = bp_state.get_distribution("y")

    distri_y_ref = np.log2(distri_x).sum(axis=0, keepdims=True)
    distri_y_ref = distri_y_ref - np.max(distri_y_ref[0, :])
    distri_y_ref = 2**distri_y_ref
    distri_y_ref = distri_y_ref / distri_y_ref.sum(axis=1, keepdims=True)

    assert np.allclose(distri_y_ref, distri_y, rtol=1e-5, atol=1e-19)


def test_sparse_factor_xor():
    from scalib.attacks.factor_graph import GenFactor

    nc = 256
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    GENERIC SINGLE f
    PROPERTY F0: f(a,b,c)"""

    graph2 = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    PROPERTY F0: c = a ^ b"""
    fg = FactorGraph(graph)
    fg2 = FactorGraph(graph2)
    xor = []
    n = 4
    for a in range(nc):
        for b in range(nc):
            xor.append([a, b, (a ^ b) & 0xFF])
    bp = BPState(
        fg,
        n,
        gen_factors={"f": GenFactor.sparse_functional(np.array(xor, dtype=np.uint32))},
    )
    bp2 = BPState(fg2, n)
    distr_a = make_distri(nc, n)
    distr_b = make_distri(nc, n)
    bp.set_evidence("a", distr_a)
    bp.set_evidence("b", distr_b)
    bp2.set_evidence("a", distr_a)
    bp2.set_evidence("b", distr_b)

    bp.bp_loopy(10, False)
    bp2.bp_loopy(10, False)
    assert np.allclose(bp.get_distribution("c"), bp2.get_distribution("c"))


def test_sparse_factor_xor_multi():
    from scalib.attacks.factor_graph import GenFactor

    nc = 256
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    GENERIC MULTI f
    PROPERTY F0: f(a,b,c)"""

    graph2 = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    PROPERTY F0: c = a ^ b"""
    fg = FactorGraph(graph)
    fg2 = FactorGraph(graph2)
    xor = []
    n = 4
    for a in range(nc):
        for b in range(nc):
            xor.append([a, b, (a ^ b) & 0xFF])
    bp = BPState(
        fg,
        n,
        gen_factors={
            "f": [
                GenFactor.sparse_functional(np.array(xor, dtype=np.uint32))
                for _ in range(n)
            ]
        },
    )
    bp2 = BPState(fg2, n)
    distr_a = make_distri(nc, n)
    distr_b = make_distri(nc, n)
    bp.set_evidence("a", distr_a)
    bp.set_evidence("b", distr_b)
    bp2.set_evidence("a", distr_a)
    bp2.set_evidence("b", distr_b)

    bp.bp_loopy(10, False)
    bp2.bp_loopy(10, False)
    assert np.allclose(bp.get_distribution("c"), bp2.get_distribution("c"))


def test_dense_factor_bff():
    from scalib.attacks.factor_graph import GenFactor

    nc = 13
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    VAR MULTI d
    GENERIC SINGLE f
    PROPERTY F0: f(a,b,c,d)"""

    graph2 = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI bi
    VAR MULTI c
    VAR MULTI d
    TABLE sub
    PROPERTY F0: c = a + b
    PROPERTY F1: bi = sub[b]
    PROPERTY F2: d = a + bi"""

    fg = FactorGraph(graph)
    fg2 = FactorGraph(
        graph2, {"sub": np.array([-n % nc for n in range(nc)], dtype=np.uint32)}
    )
    bff = np.zeros((nc, nc, nc, nc))
    n = 4
    for a in range(nc):
        for b in range(nc):
            bff[a, b, (a + b) % nc, (a - b) % nc] = 1.0

    bp = BPState(fg, n, gen_factors={"f": GenFactor.dense(bff)})
    bp2 = BPState(fg2, n)

    distr_a = make_distri(nc, n)
    distr_b = make_distri(nc, n)
    bp.set_evidence("a", distr_a)
    bp.set_evidence("b", distr_b)
    bp2.set_evidence("a", distr_a)
    bp2.set_evidence("b", distr_b)

    bp.bp_loopy(10, False)
    bp2.bp_loopy(10, False)
    assert np.allclose(bp.get_distribution("c"), bp2.get_distribution("c"))


def test_dense_factor_bff():
    from scalib.attacks.factor_graph import GenFactor

    nc = 13
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    VAR MULTI d
    GENERIC MULTI f
    PROPERTY F0: f(a,b,c,d)"""

    graph2 = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI bi
    VAR MULTI c
    VAR MULTI d
    TABLE sub
    PROPERTY F0: c = a + b
    PROPERTY F1: bi = sub[b]
    PROPERTY F2: d = a + bi"""

    fg = FactorGraph(graph)
    fg2 = FactorGraph(
        graph2, {"sub": np.array([-n % nc for n in range(nc)], dtype=np.uint32)}
    )
    bff = np.zeros((nc, nc, nc, nc))
    n = 4
    for a in range(nc):
        for b in range(nc):
            bff[a, b, (a + b) % nc, (a - b) % nc] = 1.0

    bp = BPState(fg, n, gen_factors={"f": [GenFactor.dense(bff) for _ in range(n)]})
    bp2 = BPState(fg2, n)

    distr_a = make_distri(nc, n)
    distr_b = make_distri(nc, n)
    bp.set_evidence("a", distr_a)
    bp.set_evidence("b", distr_b)
    bp2.set_evidence("a", distr_a)
    bp2.set_evidence("b", distr_b)

    bp.bp_loopy(10, False)
    bp2.bp_loopy(10, False)
    assert np.allclose(bp.get_distribution("c"), bp2.get_distribution("c"))


def test_manytraces():
    """No numerical underflow with many traces."""
    nc = 8
    n = 500
    distri_x = np.ones((n, nc))
    distri_x[:, 0] = 2.0
    distri_x[:, 1] = 1.5
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T

    graph = f"""
            NC {nc}
            VAR MULTI x
            VAR SINGLE y
            PUB SINGLE c
            PROPERTY y = x ^ c
            """
    graph = FactorGraph(graph)
    bp_state = BPState(graph, n, {"c": 0})

    bp_state.set_evidence("x", distri_x)

    bp_state.bp_acyclic("y")
    distri_y = bp_state.get_distribution("y")

    distri_y_ref = np.log2(distri_x).sum(axis=0, keepdims=True)
    distri_y_ref = distri_y_ref - np.max(distri_y_ref[0, :])
    distri_y_ref = 2**distri_y_ref
    distri_y_ref = distri_y_ref / distri_y_ref.sum(axis=1, keepdims=True)

    assert np.allclose(distri_y_ref, distri_y, rtol=1e-5, atol=1e-19)


def test_ADD3():
    nc = 13
    graph = f"""NC {nc}
    TABLE SUB = [0, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    VAR MULTI a0
    VAR MULTI x0
    VAR MULTI x1



    PROPERTY F0: a0 = x0 + x1
    """
    a0_distr = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    x0_distr = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
    )
    x1_distr = np.array(
        [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    fg = FactorGraph(graph)
    bp_state = BPState(fg, 1)
    bp_state.set_evidence("x0", x0_distr)
    bp_state.set_evidence("x1", x1_distr)
    bp_state.set_evidence("a0", a0_distr)
    bp_state.bp_loopy(50, initialize_states=False)
    for x in ["x0", "x1", "a0"]:
        print(bp_state.get_distribution(x))
        assert not np.isnan(bp_state.get_distribution(x)).any()


def test_ADD4():
    nc = 256
    n = 10
    graph = f"""NC {nc}
    VAR MULTI a0
    VAR MULTI x0
    VAR MULTI x1



    PROPERTY F0: a0 = x0 + x1
    """
    fg = FactorGraph(graph)
    for _ in range(50):
        bp = BPState(fg, n)

        bp.set_evidence("x0", make_distri(nc, n))
        bp.set_evidence("x1", make_distri(nc, n))
        bp.set_evidence("a0", make_distri(nc, n))
        bp.bp_loopy(50, initialize_states=False)
        for x in ["x0", "x1", "a0"]:
            assert not np.isnan(bp.get_distribution(x)).any()


def test_mix_single_multi():
    graph_desc = """
    NC 2
    TABLE t
    VAR SINGLE s

    VAR MULTI a
    VAR MULTI b

    PROPERTY TABLE: a = t[s]
    PROPERTY XOR: b = a ^ s
    """

    graph_desc2 = """
    NC 2
    TABLE t
    VAR SINGLE s

    VAR SINGLE a0
    VAR SINGLE b0

    PROPERTY a0 = t[s]
    PROPERTY b0 = a0 ^ s

    VAR SINGLE a1
    VAR SINGLE b1

    PROPERTY a1 = t[s]
    PROPERTY b1 = a1 ^ s
    """

    graph = FactorGraph(graph_desc, {"t": np.arange(2, dtype=np.uint32)})
    graph2 = FactorGraph(graph_desc2, {"t": np.arange(2, dtype=np.uint32)})

    sasca = BPState(graph, 2)
    sasca.bp_loopy(4, initialize_states=True)  # check no crash

    sasca = BPState(graph, 2)
    sasca2 = BPState(graph2, 1)

    for _ in range(10):
        sasca.bp_loopy(1, initialize_states=False)
        sasca2.bp_loopy(1, initialize_states=False)
        for v in ("a", "b"):
            d = sasca.get_distribution(v)
            d0 = sasca2.get_distribution(f"{v}0")
            d1 = sasca2.get_distribution(f"{v}0")
            if d is not None:
                assert np.allclose(d[0, :], d0, rtol=1e-5, atol=1e-19)
                assert np.allclose(d[1, :], d1, rtol=1e-5, atol=1e-19)
        d = sasca.get_distribution("s")
        d2 = sasca2.get_distribution("s")
        if d is not None:
            assert np.allclose(d, d2, rtol=1e-5, atol=1e-19)


def test_mix_single_multi2():
    graph = f"""
        NC 16
        VAR SINGLE A
        VAR MULTI B
        VAR MULTI AnB
        VAR MULTI C
        VAR MULTI AnC

        PROPERTY A_and_B: AnB = A & B
        PROPERTY A_and_C: AnC = A & C
    """
    fg = FactorGraph(graph)
    bp = BPState(fg, 2)
    bp.bp_loopy(3, initialize_states=True)


def test_ADD5():
    for nc in [13, 256]:
        n = 10
        graph = f"""NC {nc}
        VAR MULTI x
        VAR MULTI y
        VAR MULTI z



        PROPERTY F0: z = x + y
        """

        fg = FactorGraph(graph)
        for _ in range(50):
            bp = BPState(fg, n)

            x_distri = make_distri(nc, n)
            z_distri = make_distri(nc, n)
            bp.set_evidence("x", x_distri)
            bp.set_evidence("z", z_distri)

            y_distri_ref = np.zeros(x_distri.shape)

            for x in range(nc):
                for z in range(nc):
                    y_distri_ref[:, (z - x) % nc] += x_distri[:, x] * z_distri[:, z]

            y_distri_ref = (y_distri_ref.T / np.sum(y_distri_ref, axis=1)).T

            bp.bp_loopy(3, initialize_states=False)

            assert np.allclose(y_distri_ref, bp.get_distribution("y"))
            for x in ["x", "y", "z"]:
                print(bp.get_distribution(x))
                assert not np.isnan(bp.get_distribution(x)).any()


def test_clear_beliefs():
    graph = """
    NC 2
    PROPERTY s1: x = a^b
    VAR MULTI x
    VAR MULTI a
    VAR MULTI b
    """
    fg = FactorGraph(graph)
    bp = BPState(fg, 1)
    for k in fg.vars():
        bp.set_evidence(k, make_distri(2, 1))
    bp.bp_loopy(1, False, clear_beliefs=False)

    assert bp.get_belief_from_var("x", "s1") is not None
    assert bp.get_belief_from_var("a", "s1") is not None
    assert bp.get_belief_from_var("b", "s1") is not None


def test_sub():
    g = """
    NC 256
    PROPERTY s1: z = x - y
    VAR MULTI x
    VAR MULTI y
    VAR MULTI z
    """

    nc = 256
    n = 10
    fg = FactorGraph(g)
    bp = BPState(fg, n)
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)
    bp.set_evidence("x", distri_x)
    bp.set_evidence("y", distri_y)
    z_distri_ref = np.zeros(distri_x.shape)

    for x in range(nc):
        for y in range(nc):
            z_distri_ref[:, (x - y) % nc] += distri_x[:, x] * distri_y[:, y]

    z_distri_ref = (z_distri_ref.T / np.sum(z_distri_ref, axis=1)).T
    bp.bp_loopy(1, True)
    distri_z = bp.get_distribution("z")

    assert np.allclose(z_distri_ref, distri_z)


def test_sub_multi_ops():
    g = """
    NC 13
    PROPERTY s1: z = -x + q - w + y
    VAR MULTI x
    VAR MULTI y
    VAR MULTI z
    VAR MULTI q
    VAR MULTI w
    """

    nc = 13
    n = 10
    fg = FactorGraph(g)
    bp = BPState(fg, n)
    distri_x = make_distri(nc, n)
    distri_y = make_distri(nc, n)
    distri_q = make_distri(nc, n)
    distri_w = make_distri(nc, n)

    bp.set_evidence("x", distri_x)
    bp.set_evidence("y", distri_y)
    bp.set_evidence("w", distri_w)
    bp.set_evidence("q", distri_q)
    z_distri_ref = np.zeros(distri_x.shape)

    for x in range(nc):
        for y in range(nc):
            for w in range(nc):
                for q in range(nc):
                    z_distri_ref[:, (-x + q - w + y) % nc] += (
                        distri_x[:, x]
                        * distri_y[:, y]
                        * distri_w[:, w]
                        * distri_q[:, q]
                    )

    z_distri_ref = (z_distri_ref.T / np.sum(z_distri_ref, axis=1)).T
    bp.bp_acyclic("z")
    distri_z = bp.get_distribution("z")

    assert np.allclose(z_distri_ref, distri_z)


def test_single_gf_sanity_check():
    from scalib.attacks.factor_graph import GenFactor

    nc = 13
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    VAR MULTI d
    GENERIC SINGLE f
    PROPERTY F0: f(a,b,c,d)"""

    fg = FactorGraph(graph)

    bff = np.zeros((nc, nc, nc, nc))
    n = 1
    for a in range(nc):
        for b in range(nc):
            bff[a, b, (a + b) % nc, (a - b) % nc] = 1.0

    bff_sparse = []
    for a in range(nc):
        for b in range(nc):
            bff_sparse.append([a, b, (a + b) % nc, (a - b) % nc])

    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": np.array([1, 1]),
            "c": np.array([1, 1]),
            "d": np.array([12, 12]),
        },
        {"f": GenFactor.dense(bff)},
    )
    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": np.array([1, 1]),
            "c": np.array([1, 1]),
            "d": np.array([12, 12]),
        },
        {"f": GenFactor.sparse_functional(np.array(bff_sparse, dtype=np.uint32))},
    )


def test_multi_gf_sanity_check():
    from scalib.attacks.factor_graph import GenFactor

    nc = 13
    graph = f"""NC {nc}
    VAR MULTI a
    VAR MULTI b
    VAR MULTI c
    VAR MULTI d
    GENERIC MULTI f
    PROPERTY F0: f(a,b,c,d)"""

    fg = FactorGraph(graph)

    bff = np.zeros((nc, nc, nc, nc))
    bff2 = np.zeros((nc, nc, nc, nc))
    n = 1
    for a in range(nc):
        for b in range(nc):
            bff[a, b, (a + b) % nc, (a - b) % nc] = 1.0
            bff2[a, b, (a + 2 * b) % nc, (a - 2 * b) % nc] = 1.0

    bff_sparse = []
    bff2_sparse = []
    for a in range(nc):
        for b in range(nc):
            bff_sparse.append([a, b, (a + b) % nc, (a - b) % nc])
            bff2_sparse.append([a, b, (a + 2 * b) % nc, (a - 2 * b) % nc])
    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": np.array([1, 1]),
            "c": np.array([1, 2]),
            "d": np.array([12, 11]),
        },
        {"f": [GenFactor.dense(bff), GenFactor.dense(bff2)]},
    )
    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": np.array([1, 1]),
            "c": np.array([1, 2]),
            "d": np.array([12, 11]),
        },
        {
            "f": [
                GenFactor.sparse_functional(np.array(bff_sparse, dtype=np.uint32)),
                GenFactor.sparse_functional(np.array(bff2_sparse, dtype=np.uint32)),
            ]
        },
    )


def test_mixed_gf_sanity_check():
    from scalib.attacks.factor_graph import GenFactor

    nc = 13
    graph = f"""NC {nc}
    VAR MULTI a
    VAR SINGLE b
    VAR MULTI c
    VAR MULTI d
    GENERIC MULTI f
    PROPERTY F0: f(a,b,c,d)"""

    graph2 = f"""NC {nc}
    VAR MULTI a
    VAR SINGLE b
    VAR MULTI c
    VAR MULTI d
    GENERIC SINGLE f
    PROPERTY F0: f(a,b,c,d)"""
    fg = FactorGraph(graph)
    fg2 = FactorGraph(graph2)

    bff = np.zeros((nc, nc, nc, nc))
    bff2 = np.zeros((nc, nc, nc, nc))
    n = 1
    for a in range(nc):
        for b in range(nc):
            bff[a, b, (a + b) % nc, (a - b) % nc] = 1.0
            bff2[a, b, (a + 2 * b) % nc, (a - 2 * b) % nc] = 1.0

    bff_sparse = []
    bff2_sparse = []
    for a in range(nc):
        for b in range(nc):
            bff_sparse.append([a, b, (a + b) % nc, (a - b) % nc])
            bff2_sparse.append([a, b, (a + 2 * b) % nc, (a - 2 * b) % nc])
    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": 1,
            "c": np.array([1, 2]),
            "d": np.array([12, 11]),
        },
        {
            "f": [GenFactor.dense(bff), GenFactor.dense(bff2)],
        },
    )
    fg.sanity_check(
        {},
        {
            "a": np.array([0, 0]),
            "b": 1,
            "c": np.array([1, 2]),
            "d": np.array([12, 11]),
        },
        {
            "f": [
                GenFactor.sparse_functional(np.array(bff_sparse, dtype=np.uint32)),
                GenFactor.sparse_functional(np.array(bff2_sparse, dtype=np.uint32)),
            ]
        },
    )

    fg2.sanity_check(
        {},
        {"a": np.array([0, 1]), "b": 1, "c": np.array([1, 2]), "d": np.array([12, 0])},
        {"f": GenFactor.dense(bff)},
    )
    fg2.sanity_check(
        {},
        {"a": np.array([0, 1]), "b": 1, "c": np.array([1, 2]), "d": np.array([12, 0])},
        {"f": GenFactor.sparse_functional(np.array(bff_sparse, dtype=np.uint32))},
    )


def test_sanity_or():
    graph = """
    NC 2
    PROPERTY s1: x = a | b
    VAR SINGLE x
    VAR SINGLE a
    VAR MULTI b
    """
    fg = FactorGraph(graph)
    fg.sanity_check({}, {"x": 0, "a": 0, "b": [0, 0]})
    fg.sanity_check({}, {"x": 1, "a": 1, "b": [0, 0]})
    fg.sanity_check({}, {"x": 1, "a": 0, "b": [1, 1]})
    fg.sanity_check({}, {"x": 1, "a": 1, "b": [1, 0]})
    graph = """
    NC 2
    PROPERTY s1: x = a & b
    VAR MULTI x
    VAR SINGLE a
    VAR MULTI b
    """
    fg = FactorGraph(graph)
    fg.sanity_check({}, {"x": [0, 0], "a": 0, "b": [0, 0]})
    fg.sanity_check({}, {"x": [0, 0], "a": 0, "b": [0, 1]})
    fg.sanity_check({}, {"x": [0, 0], "a": 0, "b": [1, 0]})
    fg.sanity_check({}, {"x": [1, 1], "a": 1, "b": [1, 1]})


def test_cycle_detection_single_factor():
    graph_desc = """
                NC 2
                VAR SINGLE x
                VAR SINGLE y
                PROPERTY x = !y
                """

    fg = FactorGraph(graph_desc)
    bp = BPState(fg, 2)
    assert not bp.is_cyclic()


def test_cycle_detection_single_factor2():
    graph_desc = """
                NC 2
                VAR SINGLE x
                VAR SINGLE y
                VAR SINGLE z
                PROPERTY x = !y
                PROPERTY x = y ^ z
                """

    fg = FactorGraph(graph_desc)
    bp = BPState(fg, 2)
    assert bp.is_cyclic()


def test_cycle_detection_single_factor_with_multi():
    graph_desc = """
                NC 2
                VAR SINGLE x
                VAR SINGLE y
                VAR MULTI z
                PROPERTY x = !y
                PROPERTY x = y ^ z
                """

    fg = FactorGraph(graph_desc)
    bp = BPState(fg, 2)
    assert bp.is_cyclic()
