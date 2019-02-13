from typing import List

import numpy as np
from numpy import zeros, ones, eye, diag
from ipsolver.base_ipm import Array, Vector, Matrix
from ipsolver.LP import mehrotra_ipm


class RegularizedPrimalDualMehrotraIPM(mehrotra_ipm.MehrotraIPM):
    RO = 0.0001
    DELTA = 0.0001
    x_prev = None
    y_prev = None

    """ Mehrotra implementation of IP method. """

    @classmethod
    def _variables_initialization(cls, constraints: List[Array]) -> List[Vector]:
        A = constraints[0]
        [m, n] = A.shape

        # start point of primal-dual variables
        x = ones((n, 1))
        z = ones((n, 1))
        y = zeros((m, 1))
        r = zeros((m, 1))
        s = ones((n, 1))
        return [x, z, y, r, s]

    @classmethod
    def _build_jacobian(cls, cost_function: List[Array], constraints: List[Array], variables: List[Vector]) -> Matrix:
        A = constraints[0]
        [m, n] = A.shape
        [x, z, _, _, _] = variables
        return np.r_[
            np.c_[zeros((n, n)), - A.T, - eye(n), zeros((n, m)), -cls.RO * eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n)), cls.DELTA * eye(m), zeros((m, n))],
            np.c_[zeros((m, n)), - cls.DELTA * eye(m), zeros((m, n)), cls.DELTA * eye(m), zeros((m, n))],
            np.c_[cls.RO * eye(n), zeros((n, m)), zeros((n, n)), zeros((n, m)), cls.RO * eye(n)],
            np.c_[diag(z.T[0]), zeros((n, m)), diag(x.T[0]), zeros((n, m)), zeros((n, n))]
        ]

    @classmethod
    def _log_iterations(cls, *args, **kwargs):
        if len(args) == 0:
            cls.logger.info(
                '  k | mu      | rc      | rb      | ryk     | rxk     | trc     | trb     | alpha_p | alpha_d')
            cls.logger.info(
                ' --------------------------------------------------------------------------------------------')
        else:
            args = [cls.iter_num, cls.mu[0][0]] + list(np.concatenate(args))
            cls.logger.info(
                "{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
                    *args))

    @classmethod
    def _set_prevs(cls, x, y):
        if cls.x_prev is None:
            cls.x_prev = x
        elif x.shape != cls.x_prev.shape:
            cls.x_prev = x

        if cls.y_prev is None:
            cls.y_prev = y
        elif y.shape != cls.y_prev.shape:
            cls.y_prev = y

    @classmethod
    def _compute_residuals(cls, cost_function, constraints, variables):
        c = cost_function[0]
        [A, b] = constraints
        [x, z, y, r, s] = variables
        cls._set_prevs(x, y)

        rc = c - A.T @ y - z - cls.RO * s
        rb = A @ x + cls.DELTA * r - b
        trc = c - A.T @ y - z
        trb = A @ x - b
        ryk = cls.DELTA * (r + cls.y_prev) - cls.DELTA * y
        rxk = cls.RO * s + cls.RO * (x - cls.x_prev)
        cls.x_prev = x
        cls.y_prev = y
        return rc, rb, ryk, rxk, trc, trb

    @classmethod
    def _predictor_step(cls, constraints, variables, residuals, jacobian):
        # input parsing
        [rc, rb, ryk, rxk, _, _] = residuals
        [x, z, _, _, _] = variables

        # step performing
        rhs = - np.concatenate([rc, rb, ryk, rxk, x * z])

        d = cls._newton_step(jacobian, rhs)
        d_x = d[0:cls.n]
        d_z = d[cls.n + cls.m:2 * cls.n + cls.m]
        alpha_p = cls._get_step_length(d_x, x)
        alpha_d = cls._get_step_length(d_z, z)

        # metadata creating
        metadata = {
            "step": [alpha_p, alpha_d],
            "d": [d_x, d_z]
        }
        return metadata

    @classmethod
    def _corrector_step(cls, predictor_metadata, constraints, variables, residuals, jacobian):
        # input parsing
        [alpha_p, alpha_d] = predictor_metadata['step']
        [d_x, d_z] = predictor_metadata['d']
        [x, z, _, _, _] = variables
        [rc, rb, ryk, rxk, _, _] = residuals

        # step performing
        cls.mu = x.T @ z / cls.n
        mu_alpha = ((x + alpha_p * d_x).T @ (z + alpha_d * d_z)) / cls.n
        sigma = np.power((mu_alpha / cls.mu), 3)

        rhs = -np.concatenate([rc, rb, ryk, rxk, x * z + d_x * d_z - sigma * cls.mu])

        d = cls._newton_step(jacobian, rhs)
        d_x = d[0:cls.n]
        d_z = d[cls.n + cls.m:2 * cls.n + cls.m]
        alpha_p = cls._get_step_length(d_x, x)
        alpha_d = cls._get_step_length(d_z, z)

        return d, np.array([alpha_p, alpha_d])

    @classmethod
    def _update_variables(cls, variables, direction, step_length):
        [x, z, y, r, s] = variables
        d = direction
        [alpha_p, alpha_d] = step_length

        dx = d[0:cls.n]
        dy = d[cls.n:cls.n + cls.m]
        dz = d[cls.n + cls.m:2 * cls.n + cls.m]
        dr = d[2 * cls.n + cls.m:2 * cls.n + 2 * cls.m]
        ds = d[2 * cls.n + 2 * cls.m:]

        # new iterate
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        z = z + alpha_d * dz
        r = r + alpha_d * dr
        s = s + alpha_d * ds
        return [x, z, y, r, s]

    @classmethod
    def _check_exit_conditions(cls, residuals_norm: Vector, tol: float, iter_num: int, max_iter: int) -> bool:
        if np.max(residuals_norm[:-2]) < tol and cls.mu < tol or iter_num > max_iter:
            return True
        return False
