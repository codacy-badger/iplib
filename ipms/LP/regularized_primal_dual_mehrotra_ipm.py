from typing import List

import numpy as np
from numpy import zeros, ones, eye, diag
from ipms.logger import get_stdout_handler
from ipms.interface_ipm import Array, Vector, Matrix
from ipms.LP import merothra_ipm


class RegularizedPrimalDualMehrotraIPM(merothra_ipm.MehrotraIPM):
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
            # k, mu, nrc, nrb, nryk, nrxk, ntrc, ntrb, alpha_p, alpha_d

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
        ryk = cls.DELTA * r - cls.DELTA * y
        rxk = cls.RO * s + cls.RO * x
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
    def solve(cls, cost_function, constraints, tol=1e-8, max_iter=np.inf, logs=False):
        """ Mehrotra IP method for solving LP problems of the form
                min  c'*x
                s.t. A*x = b
                     x >= 0

            Args:
                A: numpy.array, matrix of constraints.
                b: numpy.array, vector of constraints.
                c: numpy.array, vector of coefficients in linear cost function.
                tol: float, tolerance for termination.
                max_iter: integer, maximum number of iterations. No limit by default.
                logs: boolean, flag for logs

            Returns:
                x: numpy.array, vector of primal variables.
                y: numpy.array, vector of dual variables for equality constraints.
                s: numpy.array, vector of dual variables for inequality constrains.
            """

        if logs:
            cls.logger.addHandler(get_stdout_handler())
        [A, b] = constraints
        c = cost_function[0]

        [m, n] = A.shape

        # start point of primal-dual variables
        x = ones((n, 1))
        z = ones((n, 1))
        y = zeros((m, 1))
        r = zeros((m, 1))
        s = ones((n, 1))

        # start value of perturbation parameter 'mu'
        mu = 1

        cls.logger.info('  k | mu      | rc      | rb      | trc     | trb     | ryk     | rxk     | alpha_p | alpha_d')
        cls.logger.info(' --------------------------------------------------------------------------------------------')

        k = 0
        while True:
            if k > max_iter:
                break

            rc, rb, ryk, rxk, trc, trb = cls._compute_residuals([c], [A, b], [x, z, y, r, s])

            nrb = np.linalg.norm(rb, ord=np.inf)
            nrc = np.linalg.norm(rc, ord=np.inf)
            ntrb = np.linalg.norm(trb, ord=np.inf)
            ntrc = np.linalg.norm(trc, ord=np.inf)
            nryk = np.linalg.norm(ryk, ord=np.inf)
            nrxk = np.linalg.norm(rxk, ord=np.inf)

            # stopping condition
            if np.max([nrb, nrc, mu, nryk, nrxk]) < tol:
                break

            jac = cls._build_jacobian([c], [A, b], [x, z, y, r, s])
            # predictor part
            rhs = - np.concatenate([rc, rb, ryk, rxk, x * z])
            d = cls._newton_step(jac, rhs)

            d_x = d[0:n]
            d_z = d[n + m:2*n + m]
            alpha_p = cls._get_step_length(d_x, x)
            alpha_d = cls._get_step_length(d_z, z)

            # corrector part
            mu = x.T @ z / n
            mu_alpha = ((x + alpha_p * d_x).T @ (z + alpha_d * d_z)) / n
            sigma = np.power((mu_alpha / mu), 3)

            rhs = -np.concatenate([rc, rb, ryk, rxk, x * z + d_x * d_z - sigma * mu])
            d = cls._newton_step(jac, rhs)

            d_x = d[0:n]
            d_z = d[n + m:2*n + m]
            alpha_p = cls._get_step_length(d_x, x)
            alpha_d = cls._get_step_length(d_z, z)

            dx = d[0:n]
            dy = d[n:n + m]
            dz = d[n + m:2*n + m]
            dr = d[2*n + m:2*n + 2*m]
            ds = d[2*n + 2*m:]

            # new iterate
            x = x + alpha_p * dx
            y = y + alpha_d * dy
            z = z + alpha_d * dz
            r = r + alpha_d * dr
            s = s + alpha_d * ds


            k += 1
            mu = mu[0][0]
            cls.logger.info("{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
                k, mu, nrc, nrb, ntrc, ntrb, nryk, nrxk, alpha_p, alpha_d))

        cls.logger.info('\n')
        cls._clear_globals()
        return x.ravel(), z.ravel(), y.ravel(), r.ravel(), s.ravel()
