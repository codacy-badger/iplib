import numpy as np
from numpy import zeros, ones, eye, diag
from logger import logger


def _build_jacobian(A, x, s):
    [m, n] = A.shape
    return np.r_[
        np.c_[zeros((n, n)), A.T, eye(n)],
        np.c_[A, zeros((m, m)), zeros((m, n))],
        np.c_[diag(s.T[0]), zeros((n, m)), diag(x.T[0])]
    ]


def _get_alpha(d, var):
    alpha = 1
    ax_index = np.where(d < 0)
    if ax_index[0].size != 0:
        xi = var[ax_index]
        d_xi = d[ax_index]
        alpha = min(1, min(-xi / d_xi))
    return alpha


def mehrotra(A: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float, max_iter=np.inf):
    """ Mehrotra IP method for solving LP problems of the form
        min  c'*x
        s.t. A*x = b
             x >= 0

    Args:
        A: numpy.ndarray, matrix of constraints.
        b: numpy.ndarray, vector of constraints.
        c: numpy.ndarray, vector of coefficients in linear cost function.
        tol: float, tolerance for termination.
        max_iter: integer, maximum number of iterations. No limit by default.

    Returns:
        x: numpy.ndarray, vector of primal variables.
        y: numpy.ndarray, vector of dual variables for equality constraints.
        s: numpy.ndarray, vector of dual variables for inequality constrains.
    """
    [m, n] = A.shape

    # start point of primal-dual variables
    x = ones((n, 1))
    s = ones((n, 1))
    y = zeros((m, 1))

    # constants
    tau = 0.999

    # start value of pertrubation parameter 'mu'
    mu = 1

    logger.info('k    mu        rb        rc        alpha_p    alpha_d')

    k = 0
    while True:
        if k > max_iter:
            break

        rb = A@x - b
        rc = A.T@y + s - c

        nrb = np.linalg.norm(rb, ord=np.inf)
        nrc = np.linalg.norm(rc, ord=np.inf)

        # stopping condition
        if np.max([nrb, nrc, mu]) < tol:
            break

        M = _build_jacobian(A, x, s)
        rhs = - np.concatenate([rc, rb, x*s])

        da = np.linalg.solve(M, rhs)
        da_x = da[0:n]
        da_s = da[n + m:]

        alpha_p = _get_alpha(da_x, x)
        alpha_d = _get_alpha(da_s, s)

        mu = x.T@s/n
        mu_alfa = ((x + alpha_p * da_x).T@(s + alpha_d * da_s))/n

        # 3. Set the centering parameter
        sigma = np.power((mu_alfa / mu), 3)

        # 4. Solve the second linear system

        # right-hand side of the second linear system
        rhs = -np.concatenate([rc, rb, x*s + da_x*da_s - sigma*mu])

        d = np.linalg.solve(M, rhs)
        d_x = d[0:n]
        d_s = d[n + m:]

        alpha_p = _get_alpha(d_x, x)
        alpha_d = _get_alpha(d_s, s)

        # components of d
        dx = d[0:n]
        dy = d[n:n+m]
        ds = d[n+m:]

        # new iterate
        x = x + alpha_p*dx
        y = y + alpha_d*dy
        s = s + alpha_d*ds

        k += 1
        mu = mu[0][0]
        logger.info("{:d}    {:1.4f}     {:1.4f}   {:1.4f}   {:1.4f}     {:1.4f}".format(k, mu, nrb, nrc, alpha_p, alpha_d))

    return x, y, s


if __name__ == "__main__":
    A = np.array([[1, 1, 0],
                  [1, 3, 2],
                  [0, 2, 3]], dtype=np.float64)
    I = np.eye(A.shape[0])
    A = np.c_[A, I]
    b = np.array([[7], [15], [9]], dtype=np.float64)
    c = - np.array([[3], [13], [13], [0], [0], [0]], dtype=np.float64)
    tol = 1e-4

    [x, y, s] = mehrotra(A, b, c, tol)

    print("x = {}".format(x))
    print("y = {}".format(y))
    print("s = {}".format(s))
    # [m, n] = A.shape
    # x = np.ones(shape=(n, 1))
    # s = np.ones(shape=(n, 1))
    # print(_build_jacobian(A, x, s))



