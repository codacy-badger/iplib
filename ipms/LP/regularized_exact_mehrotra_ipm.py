import numpy as np
from numpy import zeros, eye, diag
from ipms.LP.merothra_ipm import MehrotraIPM


class RegularizedExactMehrotraIPM(MehrotraIPM):

    RO = 0.001
    x_prev = None

    @classmethod
    def _build_jacobian(cls, A, x, s):
        [m, n] = A.shape
        return np.r_[
            np.c_[cls.RO*eye(n), A.T, eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n))],
            np.c_[diag(s.T[0]), zeros((n, m)), diag(x.T[0])]
        ]

    @classmethod
    def _compute_residuals(cls, A, b, c, x, y, s):
        if cls.x_prev is None:
            cls.x_prev = x*10
        elif cls.x_prev.shape != x.shape:
            cls.x_prev = x*10
        rb = A @ x - b
        rc = A.T @ y + s - c - cls.RO * (x - cls.x_prev)
        cls.x_prev = x
        return rb, rc

    @classmethod
    def _clear_globals(cls):
        cls.x_prev = None
