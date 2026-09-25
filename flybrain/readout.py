"""The v2 readout: ridge regression that can answer with a label or with a number.

v1 always treated the answer as one of several unrelated labels. That cannot decode a count,
because no linear map picks out the middle values of one ordered quantity with an argmax. Here a
problem whose answers are integers may also be read as a single number that is then rounded.
Which head to use, and every other choice, is made on the validation split.
"""
import numpy as np

LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4)


def is_numeric(problem):
    return all(isinstance(c, int) and not isinstance(c, bool) for c in problem.classes)


class RidgePath:
    """Ridge fits for many penalties at once, from one eigendecomposition.

    Works on whichever Gram matrix is smaller, so it handles more features than samples.
    """

    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.Z = Z
        self.dual = Z.shape[1] > Z.shape[0]
        G = Z @ Z.T if self.dual else Z.T @ Z
        self.e, self.Q = np.linalg.eigh(G.astype(np.float64))

    def weights(self, Y, lam):
        """Weights W, with the intercept handled by centring Y, for targets Y of shape (n, m)."""
        ym = Y.mean(0)
        Yc = Y - ym
        if self.dual:
            A = self.Q @ ((self.Q.T @ Yc) / (self.e + lam)[:, None])
            W = self.Z.T @ A
        else:
            W = self.Q @ ((self.Q.T @ (self.Z.T @ Yc)) / (self.e + lam)[:, None])
        return W, ym

    def scores(self, X, W, ym):
        return ((X - self.mu) / self.sd) @ W + ym


def decode(head, s, classes):
    """Turn readout scores into class indices."""
    if head == "label":
        return s.argmax(1)
    values = np.array(classes)
    return np.abs(np.rint(s[:, :1]) - values[None, :]).argmin(1)


def targets(head, y, classes):
    return np.eye(len(classes))[y] if head == "label" else np.array(classes, float)[y][:, None]


def heads_for(problem):
    return ["label", "number"] if is_numeric(problem) else ["label"]
