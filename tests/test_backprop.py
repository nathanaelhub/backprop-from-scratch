"""
Tests for the from-scratch backprop MLP.

The load-bearing test is the gradient check: it proves the hand-derived
analytic gradients match central finite differences, i.e. the chain-rule
math in backward() is actually correct. The rest pin the matrix/activation
primitives and confirm the network trains.
"""
import math

import pytest

from backprop import (
    MLP, accuracy, col_sums, cross_entropy, gradient_check, make_clusters,
    make_xor, matmul, relu, relu_grad_mask, softmax, transpose,
)


# ----------------------------------------------------------- primitives
def test_matmul_known_product():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]


def test_matmul_shapes():
    out = matmul([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])
    assert len(out) == 2 and len(out[0]) == 2


def test_transpose_roundtrip():
    A = [[1, 2, 3], [4, 5, 6]]
    assert transpose(transpose(A)) == A
    assert transpose(A) == [[1, 4], [2, 5], [3, 6]]


def test_col_sums():
    assert col_sums([[1, 2], [3, 4], [5, 6]]) == [9, 12]


def test_relu_and_mask():
    assert relu([[-1.0, 0.0, 2.0]]) == [[0.0, 0.0, 2.0]]
    assert relu_grad_mask([[-1.0, 0.0, 2.0]]) == [[0.0, 0.0, 1.0]]


# ----------------------------------------------------------- softmax / loss
def test_softmax_rows_sum_to_one():
    for row in softmax([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]):
        assert math.isclose(sum(row), 1.0, abs_tol=1e-12)


def test_softmax_is_numerically_stable():
    # the max-subtraction trick must keep huge logits from overflowing
    out = softmax([[1000.0, 1001.0, 1002.0]])
    assert math.isclose(sum(out[0]), 1.0, abs_tol=1e-12)
    assert out[0][2] > out[0][1] > out[0][0]


def test_cross_entropy_bounds():
    # a confident correct prediction -> ~0 loss
    assert cross_entropy([[0.999, 0.001]], [0]) < 0.01
    # a uniform prediction over k classes -> log k
    k = 4
    uniform = [[1.0 / k] * k]
    assert math.isclose(cross_entropy(uniform, [0]), math.log(k), abs_tol=1e-9)


def test_accuracy():
    P = [[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]
    assert accuracy(P, [0, 1, 1]) == pytest.approx(2 / 3)


# ----------------------------------------------------- the gradient check
@pytest.mark.parametrize("dataset,hidden", [(make_xor, 8), (make_clusters, 6)])
def test_analytic_gradients_match_finite_differences(dataset, hidden):
    X, y, k = dataset()
    net = MLP(d=len(X[0]), h=hidden, k=k, seed=0)
    worst = gradient_check(net, X, y, n_checks=20)
    assert worst < 1e-4, f"gradient check failed: worst relative error {worst:.2e}"


# ----------------------------------------------------------- training
def _train(X, y, k, hidden, lr, epochs):
    net = MLP(d=len(X[0]), h=hidden, k=k, seed=0)
    for _ in range(epochs):
        P, cache = net.forward(X)
        net.step(net.backward(P, y, cache), lr)
    P, _ = net.forward(X)
    return cross_entropy(P, y), accuracy(P, y)


def test_training_converges_on_xor():
    X, y, k = make_xor()
    loss, acc = _train(X, y, k, hidden=8, lr=0.5, epochs=600)
    assert acc >= 0.95


def test_training_converges_on_clusters():
    X, y, k = make_clusters()
    loss, acc = _train(X, y, k, hidden=8, lr=0.3, epochs=300)
    assert acc >= 0.95


def test_one_step_reduces_loss():
    X, y, k = make_clusters()
    net = MLP(d=len(X[0]), h=8, k=k, seed=0)
    P, cache = net.forward(X)
    before = cross_entropy(P, y)
    net.step(net.backward(P, y, cache), lr=0.3)
    after = cross_entropy(net.forward(X)[0], y)
    assert after < before


# ----------------------------------------------------------- reproducibility
def test_seed_is_deterministic():
    a = MLP(d=2, h=8, k=2, seed=0)
    b = MLP(d=2, h=8, k=2, seed=0)
    assert a.W1 == b.W1 and a.W2 == b.W2
    c = MLP(d=2, h=8, k=2, seed=1)
    assert c.W1 != a.W1
