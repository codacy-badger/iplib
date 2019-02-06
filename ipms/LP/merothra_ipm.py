import numpy as np
from numpy import zeros, ones, eye, diag
from ipms.logger import get_stdout_handler
from ipms.abstract_ipm import AbstractIPM


class MehrotraIPM(AbstractIPM):
    """ Mehrotra implementation of IP method. """

    @staticmethod
    def _build_jacobian(A, x, s):
        [m, n] = A.shape
        return np.r_[
            np.c_[zeros((n, n)), A.T, eye(n)],
            np.c_[A, zeros((m, m)), zeros((m, n))],
            np.c_[diag(s.T[0]), zeros((n, m)), diag(x.T[0])]
        ]

    @staticmethod
    def _get_step_length(d, var):
        alpha = MehrotraIPM.UNIT_STEP_LENGTH
        ax_index = np.where(d < 0)
        if ax_index[0].size != 0:
            xi = var[ax_index]
            d_xi = d[ax_index]
            alpha = min(1, min(-xi / d_xi))
        return alpha

    @staticmethod
    def solve(A, b, c, tol=1e-8,  max_iter=np.inf, logs=False):
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
            MehrotraIPM.logger.addHandler(get_stdout_handler())

        [m, n] = A.shape

        # start point of primal-dual variables
        x = ones((n, 1))
        s = ones((n, 1))
        y = zeros((m, 1))

        # start value of perturbation parameter 'mu'
        mu = 1

        MehrotraIPM.logger.info('  k | mu      | rb      | rc      | alpha_p | alpha_d')
        MehrotraIPM.logger.info(' ----------------------------------------------------')

        k = 0
        while True:
            if k > max_iter:
                break

            rb = A @ x - b
            rc = A.T @ y + s - c

            nrb = np.linalg.norm(rb, ord=np.inf)
            nrc = np.linalg.norm(rc, ord=np.inf)

            # stopping condition
            if np.max([nrb, nrc, mu]) < tol:
                break

            # predictor part
            jac = MehrotraIPM._build_jacobian(A, x, s)
            rhs = - np.concatenate([rc, rb, x * s])
            d = MehrotraIPM._newton_step(jac, rhs)

            d_x = d[0:n]
            d_s = d[n + m:]
            alpha_p = MehrotraIPM._get_step_length(d_x, x)
            alpha_d = MehrotraIPM._get_step_length(d_s, s)

            # corrector part
            mu = x.T @ s / n
            mu_alpha = ((x + alpha_p * d_x).T @ (s + alpha_d * d_s)) / n
            sigma = np.power((mu_alpha / mu), 3)

            rhs = -np.concatenate([rc, rb, x * s + d_x * d_s - sigma * mu])
            d = MehrotraIPM._newton_step(jac, rhs)

            d_x = d[0:n]
            d_s = d[n + m:]
            alpha_p = MehrotraIPM._get_step_length(d_x, x)
            alpha_d = MehrotraIPM._get_step_length(d_s, s)

            dx = d[0:n]
            dy = d[n:n + m]
            ds = d[n + m:]

            # new iterate
            x = x + alpha_p * dx
            y = y + alpha_d * dy
            s = s + alpha_d * ds

            k += 1
            mu = mu[0][0]
            MehrotraIPM.logger.info("{:3d} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
                k, mu, nrb, nrc, alpha_p, alpha_d))

        MehrotraIPM.logger.info('\n')
        return x.ravel(), y.ravel(), s.ravel()
