import pytest
from scalib.attacks import SASCAGraph
import numpy as np
import os


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
    graph = SASCAGraph(graph, n)
    graph.set_table("table", table)

    x = distri_x.argmax(axis=1)
    graph.sanity_check({"x": x, "y": table[x]})

    graph.set_init_distribution("x", distri_x)

    graph.run_bp(1)
    distri_y = graph.get_distribution("y")

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
    graph = SASCAGraph(graph, n)
    graph.set_table("table", table)

    graph.sanity_check({"x": np.array([0, 1]), "y": np.array([0, 0])})

    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)

    graph.run_bp(1)
    distri_x = graph.get_distribution("x")
    distri_y = graph.get_distribution("y")

    distri_x_ref = np.array([0.5, 0.5])
    distri_y_ref = np.array([1.0, 0.0])

    assert np.allclose(distri_x_ref, distri_x)
    assert np.allclose(distri_y_ref, distri_y)


def test_not():
    """
    Test non-bijective Table lookup
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
    graph = SASCAGraph(graph, n)

    graph.sanity_check({"x": np.array([0, 1]), "y": np.array([1, 0])})

    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)

    graph.run_bp(1)
    distri_x = graph.get_distribution("x")
    distri_y = graph.get_distribution("y")

    s = 0.6 * 0.2 + 0.4 * 0.8
    b = 0.6 * 0.2 / s
    distri_x_ref = np.array([b, 1.0 - b])
    distri_y_ref = np.array([1.0 - b, b])

    assert np.allclose(distri_x_ref, distri_x)
    assert np.allclose(distri_y_ref, distri_y)


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
        VAR MULTI p#come comments
        """
    graph = SASCAGraph(graph, n)
    graph.set_public("p", public)
    graph.set_init_distribution("x", distri_x)

    graph.run_bp(1)

    distri_y = graph.get_distribution("y")
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = x & public
        distri_y_ref[np.arange(n), y] += distri_x[np.arange(n), x]

    assert np.allclose(distri_y_ref, distri_y)


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
        VAR MULTI p
        VAR MULTI p2
        NC {nc}
        """
    graph = SASCAGraph(graph, n)
    graph.set_public("p", public)
    graph.set_public("p2", public2)
    graph.set_init_distribution("x", distri_x)

    graph.run_bp(1)

    distri_y = graph.get_distribution("y")
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = x ^ public ^ public2
        distri_y_ref[np.arange(n), y] = distri_x[np.arange(n), x]

    assert np.allclose(distri_y_ref, distri_y)


def test_AND():
    """
    Test AND between distributions
    """

    def norm_d(d):
        return d / np.sum(d, axis=1, keepdims=True)

    def make_distri(nc, n):
        return norm_d(np.random.randint(1, 10000000, (n, nc)).astype(np.float64))

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
        graph = SASCAGraph(graph, n)
        graph.set_init_distribution("x", distri_x)
        graph.set_init_distribution("y", distri_y)
        graph.set_init_distribution("z", distri_z)

        distri_x_ref = np.zeros(distri_x.shape)
        distri_y_ref = np.zeros(distri_y.shape)
        distri_z_ref = np.zeros(distri_z.shape)

        for x in range(nc):
            for y in range(nc):
                distri_x_ref[:, x] += distri_z[:, x & y] * distri_y[:, y]
                distri_y_ref[:, y] += distri_z[:, x & y] * distri_x[:, x]
                distri_z_ref[:, x & y] += distri_x[:, x] * distri_y[:, y]

        # def make_dist_matrix(d):
        #    d_matrix = np.zeros((n, nc, nc))
        #    for x in range(nc):
        #        for y in range(nc):
        #            d_matrix[:,x&y,y] += d[:,x]
        #    return d_matrix
        # distri_x_matrix = make_dist_matrix(distri_x)
        # distri_y_ref = np.linalg.solve(distri_x_matrix, distri_z)
        # distri_y_matrix = make_dist_matrix(distri_y)
        # distri_x_ref = np.linalg.solve(distri_y_matrix, distri_z)

        distri_x_ref = norm_d(distri_x_ref * distri_x)
        distri_y_ref = norm_d(distri_y_ref * distri_y)
        distri_z_ref = norm_d(distri_z_ref * distri_z)

        print("#### Ref:")
        print(distri_x_ref)
        print(distri_y_ref)
        print(distri_z_ref)

        graph.run_bp(1)
        distri_x = graph.get_distribution("x")
        distri_y = graph.get_distribution("y")
        distri_z = graph.get_distribution("z")

        print("#### Got:")
        print(distri_x)
        print(distri_y)
        print(distri_z)

        assert np.allclose(distri_z_ref, distri_z)
        assert np.allclose(distri_x_ref, distri_x)
        assert np.allclose(distri_y_ref, distri_y)


def test_ADD():
    """
    Test ADD between distributions
    """
    nc = 251
    n = 4
    distri_x = np.random.randint(1, 10000000, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T
    distri_y = np.random.randint(1, 10000000, (n, nc))
    distri_y = (distri_y.T / np.sum(distri_y, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x+y
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y

        """

    graph = SASCAGraph(graph, n)
    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)

    graph.run_bp(1)
    distri_z = graph.get_distribution("z")

    distri_z_ref = np.zeros(distri_z.shape)
    msg = np.zeros(distri_z.shape)

    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:, (x + y) % nc] += distri_x[:, x] * distri_y[:, y]

    distri_z_ref = (distri_z_ref.T / np.sum(distri_z_ref, axis=1)).T
    assert np.allclose(distri_z_ref, distri_z)


def test_ADD_multiple():
    """
    Test ADD between distributions
    """
    nc = 17
    n = 4
    distri_x = np.random.randint(1, 10000000, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T
    distri_y = np.random.randint(1, 10000000, (n, nc))
    distri_y = (distri_y.T / np.sum(distri_y, axis=1)).T
    distri_w = np.random.randint(1, 10000000, (n, nc))
    distri_w = (distri_w.T / np.sum(distri_w, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x+y+w
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y
        VAR MULTI w
        """

    graph = SASCAGraph(graph, n)
    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)
    graph.set_init_distribution("w", distri_w)

    graph.run_bp(1)
    distri_z = graph.get_distribution("z")

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
    distri_x = np.random.randint(1, 10000000, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T
    distri_y = np.random.randint(1, 10000000, (n, nc))
    distri_y = (distri_y.T / np.sum(distri_y, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x*y
        VAR MULTI z
        VAR MULTI x
        VAR MULTI y

        """

    graph = SASCAGraph(graph, n)
    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)

    graph.run_bp(1)
    distri_z = graph.get_distribution("z")

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
    n = 1
    distri_x = np.random.randint(1, 10000000, (n, nc))
    distri_x = (distri_x.T / np.sum(distri_x, axis=1)).T
    distri_y = np.random.randint(1, 10000000, (n, nc))
    distri_y = (distri_y.T / np.sum(distri_y, axis=1)).T

    distri_b = np.random.randint(1, 10000000, (n, nc))
    distri_b = (distri_b.T / np.sum(distri_b, axis=1)).T
    distri_a = np.random.randint(1, 10000000, (n, nc))
    distri_a = (distri_a.T / np.sum(distri_a, axis=1)).T

    graph = f"""
        # some comments
        NC {nc}
        PROPERTY z = x^y
        PROPERTY x = a^b
        VAR MULTI x # some comments too
        VAR MULTI y
        VAR MULTI a
        VAR MULTI b
        VAR SINGLE z"""

    graph = SASCAGraph(graph, n)
    graph.set_init_distribution("x", distri_x)
    graph.set_init_distribution("y", distri_y)
    graph.set_init_distribution("a", distri_a)
    graph.set_init_distribution("b", distri_b)

    graph.run_bp(10)
    distri_z = graph.get_distribution("z")

    distri_z_ref = np.zeros(distri_z.shape)
    msg = np.zeros(distri_z.shape)

    for a in range(nc):
        for b in range(nc):
            msg[:, a ^ b] += distri_a[:, a] * distri_b[:, b]

    distri_x *= msg
    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:, x ^ y] += distri_x[:, x] * distri_y[:, y]

    distri_z_ref = (distri_z_ref.T / np.sum(distri_z_ref, axis=1)).T
    assert np.allclose(distri_z_ref, distri_z)
