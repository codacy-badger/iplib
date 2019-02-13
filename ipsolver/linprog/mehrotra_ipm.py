import numpy as np
from numpy import zeros, ones, eye, diag

from ipsolver.base_ipm import Array, Vector, List
from ipsolver.LP import interface_mehrotra_imp


class MehrotraIPM(interface_mehrotra_imp.BaseMehrotraIPM):
    """ Mehrotra implementation of IP method. """

    @classmethod
    def _variables_initialization(cls, constraints: List[Array]) -> List[Vector]:
        [m, n] = constraints[0].shape

        # start point of primal-dual variables
        x = ones((n, 1))
        s = ones((n, 1))
        y = zeros((m, 1))
        return [x, s, y]

    @classmethod
    def _constants_initialization(cls, constraints):
        A = constraints[0]
        [cls.m, cls.n] = A.shape

    @classmethod
    def _predictor_step(cls, constraints, variables, residuals, jacobian):
        # input parsing
        [rc, rb] = residuals
        [x, s, _] = variables

        # step performing
        rhs = - np.concatenate([rc, rb, x * s])

        d = cls._newton_step(jacobian, rhs)
        d_x = d[0:cls.n]
        d_s = d[cls.n + cls.m:]
        alpha_p = cls._get_step_length(d_x, x)
        alpha_d = cls._get_step_length(d_s, s)

        # metadata creating
        metadata = {
            "step": [alpha_p, alpha_d],
            "d": [d_x, d_s]
        }
        return metadata

    @classmethod
    def _corrector_step(cls, predictor_metadata, constraints, variables, residuals, jacobian):
        # input parsing
        [alpha_p, alpha_d] = predictor_metadata['step']
        [d_x, d_s] = predictor_metadata['d']
        [x, s, _] = variables
        [rc, rb] = residuals

        # step performing
        cls.mu = x.T @ s / cls.n
        mu_alpha = ((x + alpha_p * d_x).T @ (s + alpha_d * d_s)) / cls.n
        sigma = np.power((mu_alpha / cls.mu), 3)

        rhs = -np.concatenate([rc, rb, x * s + d_x * d_s - sigma * cls.mu])

        d = cls._newton_step(jacobian, rhs)
        d_x = d[0:cls.n]
        d_s = d[cls.n + cls.m:]
        alpha_p = cls._get_step_length(d_x, x)
        alpha_d = cls._get_step_length(d_s, s)

        return d, [alpha_p, alpha_d]

    @classmethod
    def _update_variables(cls, variables, direction, step_length):
        [x, s, y] = variables
        d = direction
        [alpha_p, alpha_d] = step_length

        dx = d[0:cls.n]
        dy = d[cls.n:cls.n + cls.m]
        ds = d[cls.n + cls.m:]

        # new iterate
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds
        return [x, s, y]

    @classmethod
    def _log_iterations(cls, *args, **kwargs):
        if len(args) == 0:
            cls.logger.info('  k | mu      | rc      | rb      | alpha_p | alpha_d')
            cls.logger.info(' ----------------------------------------------------')
        else:
            args = [cls.iter_num, cls.mu[0][0]] + list(np.ravel(args))
            cls.logger.info("{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(*args))
            # k, mu, nrb, nrc, alpha_p, alpha_d

    @classmethod
    def _build_jacobian(cls, cost_function, constraints, variables):
        A = constraints[0]
        [x, s, _] = variables
        [m, n] = A.shape
        return np.r_[
            np.c_[zeros((n, n)), A.T, eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n))],
            np.c_[diag(s.T[0]), zeros((n, m)), diag(x.T[0])]
        ]

    @classmethod
    def _compute_residuals(cls, cost_function, constraints, variables):
        c = cost_function[0]
        [A, b] = constraints
        [x, s, y] = variables

        rc = A.T @ y + s - c
        rb = A @ x - b
        return [rc, rb]

    @classmethod
    def _get_step_length(cls, d, var):
        alpha = MehrotraIPM._UNIT_STEP_LENGTH
        ax_index = np.where(d < 0)
        if ax_index[0].size != 0:
            xi = var[ax_index]
            d_xi = d[ax_index]
            alpha = min(1, min(-xi / d_xi))
        return alpha
