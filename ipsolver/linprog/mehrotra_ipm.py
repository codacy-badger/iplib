import numpy as np
from numpy import zeros, ones, eye, diag

from ipsolver.base_ipm import Array, Vector, List
from ipsolver.linprog import base_mehrotra_imp


class MehrotraIPM(base_mehrotra_imp.BaseMehrotraIPM):
    """ Mehrotra implementation of IP method. """

    def __init__(self):
        super().__init__()

    def _compute_function_value(self, cost_function, point):
        return (point @ cost_function[0])[0]

    def _variables_initialization(self, constraints: List[Array]) -> List[Vector]:
        [m, n] = constraints[0].shape

        # start point of primal-dual variables
        x = ones((n, 1))
        s = ones((n, 1))
        y = zeros((m, 1))
        return [x, s, y]

    def _constants_initialization(self, constraints):
        [self._M, self._N] = constraints[0].shape

    def _predictor_step(self, constraints, variables, residuals, jacobian):
        # input parsing
        [rc, rb] = residuals
        [x, s, _] = variables

        # step performing
        rhs = - np.concatenate([rc, rb, x * s])

        d = self._newton_step(jacobian, rhs)
        d_x = d[0:self._N]
        d_s = d[self._N + self._M:]
        alpha_p = self._get_step_length(d_x, x)
        alpha_d = self._get_step_length(d_s, s)

        # metadata creating
        metadata = {
            "step": [alpha_p, alpha_d],
            "d": [d_x, d_s]
        }
        return metadata

    def _corrector_step(self, predictor_metadata, constraints, variables, residuals, jacobian):
        # input parsing
        [alpha_p, alpha_d] = predictor_metadata['step']
        [d_x, d_s] = predictor_metadata['d']
        [x, s, _] = variables
        [rc, rb] = residuals

        # step performing
        self.mu = x.T @ s / self._N
        mu_alpha = ((x + alpha_p * d_x).T @ (s + alpha_d * d_s)) / self._N
        sigma = np.power((mu_alpha / self.mu), 3)

        rhs = -np.concatenate([rc, rb, x * s + d_x * d_s - sigma * self.mu])

        d = self._newton_step(jacobian, rhs)
        d_x = d[0:self._N]
        d_s = d[self._N + self._M:]
        alpha_p = self._get_step_length(d_x, x)
        alpha_d = self._get_step_length(d_s, s)

        return d, [alpha_p, alpha_d]

    def _update_variables(self, variables, direction, step_length):
        [x, s, y] = variables
        d = direction
        [alpha_p, alpha_d] = step_length

        dx = d[0:self._N]
        dy = d[self._N:self._N + self._M]
        ds = d[self._N + self._M:]

        # new iterate
        x = x + alpha_p * dx
        y = y + alpha_d * dy
        s = s + alpha_d * ds
        return [x, s, y]

    def _log_iterations(self, *args, **kwargs):
        if len(args) == 0:
            self._logger.info('  k | mu      | rc      | rb      | alpha_p | alpha_d')
            self._logger.info(' ----------------------------------------------------')
        else:
            args = [self._iter_num, self.mu[0][0]] + list(np.ravel(args))
            self._logger.info("{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(*args))
            # k, mu, nrb, nrc, alpha_p, alpha_d

    def _build_jacobian(self, cost_function, constraints, variables):
        A = constraints[0]
        [x, s, _] = variables
        [m, n] = A.shape
        return np.r_[
            np.c_[zeros((n, n)), A.T, eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n))],
            np.c_[diag(s.T[0]), zeros((n, m)), diag(x.T[0])]
        ]

    def _compute_residuals(self, cost_function, constraints, variables):
        c = cost_function[0]
        [A, b] = constraints
        [x, s, y] = variables

        rc = A.T @ y + s - c
        rb = A @ x - b

        return [rc, rb]

    @staticmethod
    def _get_step_length(d, var):
        alpha = MehrotraIPM._UNIT_STEP_LENGTH
        ax_index = np.where(d < 0)
        if ax_index[0].size != 0:
            xi = var[ax_index]
            d_xi = d[ax_index]
            alpha = min(1, min(-xi / d_xi))
        return alpha
