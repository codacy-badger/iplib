from ipsolver.linprog.regularized_primal_dual_mehrotra_ipm import RegularizedPrimalDualMehrotraIPM as LPMehrotraIPM
from ipsolver.base_ipm import Array, Vector, Matrix, List
from numpy import zeros, eye, diag
import numpy as np


class RegularizedPrimalDualMehrotraIPM(LPMehrotraIPM):

    @classmethod
    def _build_jacobian(cls, cost_function: List[Array], constraints: List[Array], variables: List[Vector]) -> Matrix:
        A = constraints[0]
        Q = cost_function[0]
        [m, n] = A.shape
        [x, z, _, _, _] = variables
        return np.r_[
            np.c_[Q, - A.T, - eye(n), zeros((n, m)), -cls.RO * eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n)), cls.DELTA * eye(m), zeros((m, n))],
            np.c_[zeros((m, n)), - cls.DELTA * eye(m), zeros((m, n)), cls.DELTA * eye(m), zeros((m, n))],
            np.c_[cls.RO * eye(n), zeros((n, m)), zeros((n, n)), zeros((n, m)), cls.RO * eye(n)],
            np.c_[diag(z.T[0]), zeros((n, m)), diag(x.T[0]), zeros((n, m)), zeros((n, n))]
        ]

    @classmethod
    def _compute_residuals(cls, cost_function, constraints, variables):
        Q = cost_function[0]
        c = cost_function[1]
        [A, b] = constraints
        [x, z, y, r, s] = variables
        cls._set_prevs(x, y)

        rc = Q @ x + c - A.T @ y - z - cls.RO * s
        rb = A @ x + cls.DELTA * r - b
        trc = Q @ x + c - A.T @ y - z
        trb = A @ x - b
        ryk = cls.DELTA * (r + cls.y_prev) - cls.DELTA * y
        rxk = cls.RO * s + cls.RO * (x - cls.x_prev)
        cls.x_prev = x
        cls.y_prev = y
        return rc, rb, ryk, rxk, trc, trb
