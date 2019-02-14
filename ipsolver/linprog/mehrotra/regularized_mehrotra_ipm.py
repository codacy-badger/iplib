from typing import List

import numpy as np
from numpy import zeros, ones, eye, diag
from ipsolver.base_ipm import Array, Vector, Matrix
from ipsolver.linprog.mehrotra import mehrotra_ipm


class RegularizedMehrotraIPM(mehrotra_ipm.IPM):
    """ Implementation of primal-dual regularized Mehrotra IP method. """

    # Default regularization parameters.
    _RO = 0.0001
    _DELTA = 0.0001

    def __init__(self, ro=None, delta=None):
        super().__init__()
        self._x_prev = None
        self._y_prev = None

        if not ro:
            self._ro = RegularizedMehrotraIPM._RO
        else:
            self._ro = ro

        if not delta:
            self._delta = RegularizedMehrotraIPM._DELTA
        else:
            self._delta = delta

    def _variables_initialization(self, constraints: List[Array]) -> List[Vector]:
        A = constraints[0]
        [m, n] = A.shape

        # start point of primal-dual variables
        x = ones((n, 1))
        z = ones((n, 1))
        y = zeros((m, 1))
        r = zeros((m, 1))
        s = ones((n, 1))
        return [x, z, y, r, s]

    def _build_jacobian(self, cost_function: List[Array], constraints: List[Array], variables: List[Vector]) -> Matrix:
        A = constraints[0]
        [m, n] = A.shape
        [x, z, _, _, _] = variables
        return np.r_[
            np.c_[zeros((n, n)), - A.T, - eye(n), zeros((n, m)), -self._ro * eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n)), self._delta * eye(m), zeros((m, n))],
            np.c_[zeros((m, n)), - self._delta * eye(m), zeros((m, n)), self._delta * eye(m), zeros((m, n))],
            np.c_[self._delta * eye(n), zeros((n, m)), zeros((n, n)), zeros((n, m)), self._ro * eye(n)],
            np.c_[diag(z.T[0]), zeros((n, m)), diag(x.T[0]), zeros((n, m)), zeros((n, n))]
        ]

    def _log_iterations(self, *args, **kwargs):
        if len(args) == 0:
            self._logger.info(
                '  k | mu      | rc      | rb      | ryk     | rxk     | trc     | trb     | alpha_p | alpha_d')
            self._logger.info(
                ' --------------------------------------------------------------------------------------------')
        else:
            args = [self._iter_num, self.mu[0][0]] + list(np.concatenate(args))
            self._logger.info(
                "{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
                    *args))

    def _set_prevs(self, x, y):
        if self._x_prev is None:
            self._x_prev = x
        elif x.shape != self._x_prev.shape:
            self._x_prev = x

        if self._y_prev is None:
            self._y_prev = y
        elif y.shape != self._y_prev.shape:
            self._y_prev = y

    def _compute_residuals(self, cost_function, constraints, variables):
        c = cost_function[0]
        [A, b] = constraints
        [x, z, y, r, s] = variables
        self._set_prevs(x, y)

        rc = c - A.T @ y - z - self._ro * s
        rb = A @ x + self._delta * r - b
        trc = c - A.T @ y - z
        trb = A @ x - b
        ryk = self._delta * (r + self._y_prev) - self._delta * y
        rxk = self._ro * s + self._ro * (x - self._x_prev)
        self._x_prev = x
        self._y_prev = y
        return rc, rb, ryk, rxk, trc, trb

    def _predictor_step(self, constraints, variables, residuals, jacobian):
        # input parsing
        [rc, rb, ryk, rxk, _, _] = residuals
        [x, z, _, _, _] = variables

        # step performing
        rhs = - np.concatenate([rc, rb, ryk, rxk, x * z])

        d = self._newton_step(jacobian, rhs)
        d_x = d[0:self._N]
        d_z = d[self._N + self._M:2 * self._N + self._M]
        alpha_p = self._get_step_length(d_x, x)
        alpha_d = self._get_step_length(d_z, z)

        # metadata creating
        metadata = {
            "step": [alpha_p, alpha_d],
            "d": [d_x, d_z]
        }
        return metadata

    def _corrector_step(self, predictor_metadata, constraints, variables, residuals, jacobian):
        # input parsing
        [alpha_p, alpha_d] = predictor_metadata['step']
        [d_x, d_z] = predictor_metadata['d']
        [x, z, _, _, _] = variables
        [rc, rb, ryk, rxk, _, _] = residuals

        # step performing
        self.mu = x.T @ z / self._N
        mu_alpha = ((x + alpha_p * d_x).T @ (z + alpha_d * d_z)) / self._N
        sigma = np.power((mu_alpha / self.mu), 3)

        rhs = -np.concatenate([rc, rb, ryk, rxk, x * z + d_x * d_z - sigma * self.mu])

        d = self._newton_step(jacobian, rhs)
        d_x = d[0:self._N]
        d_z = d[self._N + self._M:2 * self._N + self._M]
        alpha_p = self._get_step_length(d_x, x)
        alpha_d = self._get_step_length(d_z, z)

        return d, np.array([alpha_p, alpha_d])

    def _update_variables(self, variables, direction, step_length):
        [x, z, y, r, s] = variables
        d = direction
        [alpha_p, alpha_d] = step_length

        dx = d[0:self._N]
        dy = d[self._N:self._N + self._M]
        dz = d[self._N + self._M:2 * self._N + self._M]
        dr = d[2 * self._N + self._M:2 * self._N + 2 * self._M]
        ds = d[2 * self._N + 2 * self._M:]

        # new iterate
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        z = z + alpha_d * dz
        r = r + alpha_d * dr
        s = s + alpha_d * ds
        return [x, z, y, r, s]

    def _check_exit_conditions(self, residuals_norm: Vector, tol=1e-7, max_iter=np.inf) -> bool:
        if np.max(residuals_norm[:-2]) < tol and self.mu < tol or self._iter_num > max_iter:
            return True
        return False
