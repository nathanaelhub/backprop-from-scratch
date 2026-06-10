"""Backpropagation from first principles — pure Python, no numpy.

A multi-layer perceptron (input -> hidden ReLU -> softmax output) trained
with categorical cross-entropy, where every gradient is derived and coded
explicitly via the chain rule.

Matrices are represented as lists of row-lists: M[i][j] is row i, col j.

Network architecture and notation (N samples, batch forward pass):

    Z1 = X  @ W1 + b1        (N, h)   pre-activation, hidden layer
    A1 = ReLU(Z1)            (N, h)   hidden activation
    Z2 = A1 @ W2 + b2        (N, k)   pre-activation, output layer
    P  = softmax(Z2)         (N, k)   class probabilities
    L  = -(1/N) * sum_i log(P[i][y_i])   categorical cross-entropy

Backward pass (chain rule, derived in the docstrings below):

    dZ2 = (P - Y) / N                     softmax + CE combined gradient
    dW2 = A1^T @ dZ2        db2 = column-sums of dZ2
    dA1 = dZ2 @ W2^T
    dZ1 = dA1 * ReLU'(Z1)                 elementwise mask
    dW1 = X^T @ dZ1         db1 = column-sums of dZ1

Run:  python3 backprop.py
"""

import math
import random

# ----------------------------------------------------------------------------
# 1. Matrix operation layer
# ----------------------------------------------------------------------------

def matmul(A, B):
    """(n,m) @ (m,p) -> (n,p)."""
    n, m, p = len(A), len(B), len(B[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        for t in range(m):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            row = out[i]
            for j in range(p):
                row[j] += a * Bt[j]
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]


def add_rowvec(A, b):
    """Add row vector b (length p) to every row of A (n,p) — bias broadcast."""
    return [[A[i][j] + b[j] for j in range(len(b))] for i in range(len(A))]


def elementwise(A, B, op):
    """Apply op(a, b) entry-by-entry to two same-shape matrices."""
    return [[op(a, b) for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def scale(A, s):
    return [[v * s for v in row] for row in A]


def col_sums(A):
    """Sum each column of A (n,p) -> length-p vector. Used for bias gradients."""
    return [sum(col) for col in zip(*A)]


# ----------------------------------------------------------------------------
# 2. Activations and their derivatives
# ----------------------------------------------------------------------------

def relu(Z):
    return [[v if v > 0.0 else 0.0 for v in row] for row in Z]


def relu_grad_mask(Z):
    """dReLU/dz = 1 if z > 0 else 0, evaluated at the pre-activation Z."""
    return [[1.0 if v > 0.0 else 0.0 for v in row] for row in Z]


def softmax(Z):
    """Row-wise softmax with the max-subtraction trick for numerical stability:
    softmax(z)_j = exp(z_j - max(z)) / sum_k exp(z_k - max(z))."""
    out = []
    for row in Z:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        out.append([e / s for e in exps])
    return out


# ----------------------------------------------------------------------------
# 3. Loss
# ----------------------------------------------------------------------------

def cross_entropy(P, y):
    """L = -(1/N) sum_i log P[i][y_i].  Clamped to avoid log(0)."""
    eps = 1e-12
    n = len(P)
    return -sum(math.log(max(P[i][y[i]], eps)) for i in range(n)) / n


def accuracy(P, y):
    correct = sum(1 for i, row in enumerate(P)
                  if max(range(len(row)), key=row.__getitem__) == y[i])
    return correct / len(y)


# ----------------------------------------------------------------------------
# 4. The MLP with explicit forward and backward passes
# ----------------------------------------------------------------------------

class MLP:
    """input(d) -> hidden(h, ReLU) -> output(k, softmax)."""

    def __init__(self, d, h, k, seed=0):
        rng = random.Random(seed)
        # He initialization for the ReLU layer, Xavier-ish for the output:
        # keeps activation variance roughly constant so gradients don't vanish.
        s1 = math.sqrt(2.0 / d)
        s2 = math.sqrt(1.0 / h)
        self.W1 = [[rng.gauss(0, s1) for _ in range(h)] for _ in range(d)]
        self.b1 = [0.0] * h
        self.W2 = [[rng.gauss(0, s2) for _ in range(k)] for _ in range(h)]
        self.b2 = [0.0] * k
        self.k = k

    def forward(self, X):
        """Returns probabilities P plus the intermediates (cache) that the
        backward pass needs."""
        Z1 = add_rowvec(matmul(X, self.W1), self.b1)
        A1 = relu(Z1)
        Z2 = add_rowvec(matmul(A1, self.W2), self.b2)
        P = softmax(Z2)
        return P, (X, Z1, A1)

    def backward(self, P, y, cache):
        """Explicit chain rule, layer by layer.

        Output layer: for softmax p = softmax(z) composed with cross-entropy
        L = -log p_y, the Jacobians collapse to the famous simplification

            dL/dz2_j = p_j - 1{j == y}        (per sample)

        Proof sketch: dL/dz_j = sum_c (dL/dp_c)(dp_c/dz_j); with
        dL/dp_c = -1{c==y}/p_y and dp_c/dz_j = p_c(1{c==j} - p_j),
        the sum telescopes to p_j - 1{j==y}. The 1/N from the mean loss
        is folded in here so downstream gradients inherit it.
        """
        X, Z1, A1 = cache
        n = len(X)

        # One-hot encode labels: Y[i][j] = 1 if y_i == j.
        Y = [[1.0 if j == y[i] else 0.0 for j in range(self.k)]
             for i in range(n)]

        # dL/dZ2 = (P - Y) / N
        dZ2 = scale(elementwise(P, Y, lambda p, t: p - t), 1.0 / n)

        # Z2 = A1 @ W2 + b2, so:
        #   dL/dW2 = A1^T @ dZ2   (each weight W2[t][j] touches Z2[i][j] via A1[i][t])
        #   dL/db2 = sum over samples of dZ2 (bias feeds every row identically)
        dW2 = matmul(transpose(A1), dZ2)
        db2 = col_sums(dZ2)

        # Propagate into the hidden activations: dL/dA1 = dZ2 @ W2^T
        dA1 = matmul(dZ2, transpose(self.W2))

        # Through the ReLU: dL/dZ1 = dL/dA1 * 1{Z1 > 0}  (elementwise)
        dZ1 = elementwise(dA1, relu_grad_mask(Z1), lambda g, m: g * m)

        # Z1 = X @ W1 + b1, same pattern as the output layer:
        dW1 = matmul(transpose(X), dZ1)
        db1 = col_sums(dZ1)

        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        """Vanilla gradient descent: theta <- theta - lr * dL/dtheta."""
        dW1, db1, dW2, db2 = grads
        for i, row in enumerate(self.W1):
            for j in range(len(row)):
                row[j] -= lr * dW1[i][j]
        for j in range(len(self.b1)):
            self.b1[j] -= lr * db1[j]
        for i, row in enumerate(self.W2):
            for j in range(len(row)):
                row[j] -= lr * dW2[i][j]
        for j in range(len(self.b2)):
            self.b2[j] -= lr * db2[j]


# ----------------------------------------------------------------------------
# 5. Gradient check — numerically verify the analytic gradients
# ----------------------------------------------------------------------------

def gradient_check(net, X, y, eps=1e-5, n_checks=8, seed=1):
    """Compare analytic dL/dW1 against the central difference
    (L(w+eps) - L(w-eps)) / (2 eps) at a few random weight positions.
    Returns the worst relative error seen."""
    P, cache = net.forward(X)
    dW1 = net.backward(P, y, cache)[0]
    rng = random.Random(seed)
    worst = 0.0
    for _ in range(n_checks):
        i = rng.randrange(len(net.W1))
        j = rng.randrange(len(net.W1[0]))
        orig = net.W1[i][j]
        net.W1[i][j] = orig + eps
        lp = cross_entropy(net.forward(X)[0], y)
        net.W1[i][j] = orig - eps
        lm = cross_entropy(net.forward(X)[0], y)
        net.W1[i][j] = orig
        numeric = (lp - lm) / (2 * eps)
        analytic = dW1[i][j]
        rel = abs(numeric - analytic) / max(abs(numeric) + abs(analytic), 1e-12)
        worst = max(worst, rel)
    return worst


# ----------------------------------------------------------------------------
# 6. Synthetic datasets
# ----------------------------------------------------------------------------

def make_xor(n_per_quadrant=50, noise=0.15, seed=42):
    """Noisy XOR: points around (0,0),(1,1) -> class 0; (0,1),(1,0) -> class 1.
    Not linearly separable, so the hidden layer is doing real work."""
    rng = random.Random(seed)
    X, y = [], []
    for cx, cy, label in [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)]:
        for _ in range(n_per_quadrant):
            X.append([cx + rng.gauss(0, noise), cy + rng.gauss(0, noise)])
            y.append(label)
    return X, y, 2


def make_clusters(n_per_class=60, seed=7):
    """Three Gaussian blobs at the corners of a triangle — a 3-class problem
    to exercise the full softmax (k > 2)."""
    rng = random.Random(seed)
    centers = [(0.0, 0.0), (2.5, 0.0), (1.25, 2.2)]
    X, y = [], []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_class):
            X.append([cx + rng.gauss(0, 0.45), cy + rng.gauss(0, 0.45)])
            y.append(label)
    return X, y, 3


# ----------------------------------------------------------------------------
# 7. Console plotting and the training loop
# ----------------------------------------------------------------------------

def ascii_curve(values, title, width=60, height=10):
    """Render a list of values as a downsampled ASCII line chart."""
    if len(values) > width:
        stride = len(values) / width
        values = [values[int(i * stride)] for i in range(width)]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    rows = [[" "] * len(values) for _ in range(height)]
    for x, v in enumerate(values):
        r = (height - 1) - int(round((v - lo) / span * (height - 1)))
        rows[r][x] = "*"
    print(f"\n  {title}  (min={lo:.4f}, max={hi:.4f})")
    for i, row in enumerate(rows):
        label = hi if i == 0 else (lo if i == height - 1 else None)
        prefix = f"{label:8.3f} |" if label is not None else "         |"
        print(prefix + "".join(row))
    print("         +" + "-" * len(values))


def train(name, X, y, n_classes, hidden=8, lr=0.5, epochs=400, log_every=40):
    print(f"\n{'=' * 64}\nTraining on: {name}  "
          f"({len(X)} samples, {len(X[0])} features, {n_classes} classes)\n{'=' * 64}")

    net = MLP(d=len(X[0]), h=hidden, k=n_classes)

    rel_err = gradient_check(net, X, y)
    status = "OK" if rel_err < 1e-4 else "FAILED"
    print(f"Gradient check vs. central differences: "
          f"max relative error = {rel_err:.2e}  [{status}]")

    losses, accs = [], []
    for epoch in range(1, epochs + 1):
        P, cache = net.forward(X)            # full-batch gradient descent
        losses.append(cross_entropy(P, y))
        accs.append(accuracy(P, y))
        net.step(net.backward(P, y, cache), lr)
        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} | loss {losses[-1]:.4f} "
                  f"| accuracy {accs[-1] * 100:6.2f}%")

    P, _ = net.forward(X)
    final_loss, final_acc = cross_entropy(P, y), accuracy(P, y)
    print(f"  final      | loss {final_loss:.4f} | accuracy {final_acc * 100:6.2f}%")

    ascii_curve(losses, f"{name}: cross-entropy loss")
    ascii_curve(accs, f"{name}: training accuracy")
    return final_loss, final_acc


def main():
    X, y, k = make_xor()
    xor_loss, xor_acc = train("XOR (noisy, 2 classes)", X, y, k,
                              hidden=8, lr=0.5, epochs=600, log_every=60)

    X, y, k = make_clusters()
    clu_loss, clu_acc = train("Gaussian clusters (3 classes)", X, y, k,
                              hidden=8, lr=0.3, epochs=300, log_every=30)

    print(f"\n{'=' * 64}\nSummary\n{'=' * 64}")
    print(f"  XOR:      loss {xor_loss:.4f}, accuracy {xor_acc * 100:.2f}%")
    print(f"  Clusters: loss {clu_loss:.4f}, accuracy {clu_acc * 100:.2f}%")
    converged = xor_acc >= 0.95 and clu_acc >= 0.95
    print(f"  Convergence: {'YES — both tasks >= 95% accuracy' if converged else 'NO'}")


if __name__ == "__main__":
    main()
