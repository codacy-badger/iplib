from ipsolver import linprog
from ipsolver.base_ipm import Array, Vector, Matrix, List
from numpy import zeros, eye, diag
import numpy as np


class RegularizedMehrotraIPM(linprog.mehrotra.regularized_mehrotra_ipm.RegularizedMehrotraIPM):

    def __init__(self):
        super().__init__()

    def _build_jacobian(self, cost_function: List[Array], constraints: List[Array], variables: List[Vector]) -> Matrix:
        A = constraints[0]
        Q = cost_function[0]
        [m, n] = A.shape
        [x, z, _, _, _] = variables
        return np.r_[
            np.c_[Q, - A.T, - eye(n), zeros((n, m)), -self._ro * eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n)), self._delta * eye(m), zeros((m, n))],
            np.c_[zeros((m, n)), - self._delta * eye(m), zeros((m, n)), self._delta * eye(m), zeros((m, n))],
            np.c_[self._ro * eye(n), zeros((n, m)), zeros((n, n)), zeros((n, m)), self._ro * eye(n)],
            np.c_[diag(z.T[0]), zeros((n, m)), diag(x.T[0]), zeros((n, m)), zeros((n, n))]
        ]

    def _compute_residuals(self, cost_function, constraints, variables):
        Q = cost_function[0]
        c = cost_function[1]
        [A, b] = constraints
        [x, z, y, r, s] = variables
        self._set_prevs(x, y)

        rc = Q @ x + c - A.T @ y - z - self._ro * s
        rb = A @ x + self._delta * r - b
        trc = Q @ x + c - A.T @ y - z
        trb = A @ x - b
        ryk = self._delta * (r + self._y_prev) - self._delta * y
        rxk = self._ro * s + self._ro * (x - self._x_prev)
        self._x_prev = x
        self._y_prev = y
        return rc, rb, ryk, rxk, trc, trb
