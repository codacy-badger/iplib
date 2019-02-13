import unittest
import numpy as np
from scipy.optimize import linprog
from ipsolver.QP.regularized_primal_dual_mehrotra_ipm import RegularizedPrimalDualMehrotraIPM
from scipy import optimize, sparse


def solveqp(Q, c, A, b):
    c = c.ravel()
    b = b.ravel()

    def loss(x, sign=1.):
        return sign * (0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(c, x))

    def jac(x, sign=1.):
        return sign * (np.dot(Q, x) + c)

    cons = {'type': 'eq',
             'fun': lambda x: b - np.dot(A, x),
             'jac': lambda x: -A}

    x0 = np.random.randn(*c.shape)
    opt = {'disp': False}
    res_cons = optimize.minimize(loss, x0, jac=jac, constraints=cons,
                                 method='SLSQP', bounds=((0, np.inf),)*c.size,
                                 options=opt)
    return res_cons


class TestRegularizedPrimalDualMehrotra(unittest.TestCase):
    def test_initial(self):
        A = np.array([[1, 1, 0],
                      [1, 3, 2],
                      [0, 2, 3]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [0], [0], [0]], dtype=np.float64)
        Q = np.array([[1, 0, 0, 0, 0, 10],
                      [0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [10, 0, 0, 0, 0, 1]], dtype=np.float64)
        tol = 1e-7

        res = solveqp(Q, c, A, b)
        [x, _, _, _, _] = RegularizedPrimalDualMehrotraIPM.solve([Q, c], [A, b], tol, logs=False)
        self.assertTrue(np.allclose(x, res.x, rtol=1.e-3, atol=1.e-3))

    def test_rank_deficient(self):
        A = np.array([[1, -1, 0, 5, 3],
                      [-4, 3, 2, 0, 4],
                      [0, 2, -3, 1, 2],
                      [0, 2, -3, 1, 2]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9], [8]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [-5], [9], [0], [0], [0]], dtype=np.float64)
        Q = np.array([[1, 0, 0, 0, 0, 10],
                      [0, 1, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [10, 0, 0, 0, 0, 1]], dtype=np.float64)
        tol = 1e-7

        self.assertRaises(ValueError, RegularizedPrimalDualMehrotraIPM.solve, [Q, c], [A, b], tol)

    def test_random_problems_with_slack_variables(self):
        cnt_experiments = 20
        for _ in range(cnt_experiments):
            A = np.random.rand(np.random.randint(3, 25), np.random.randint(3, 25))*10
            oldA = A
            I = np.eye(A.shape[0])
            A = np.c_[A, I]
            b = np.random.rand(A.shape[0], 1)*10
            c = - np.random.rand(A.shape[1], 1)
            # Q = np.random.random((A.shape[1], A.shape[1]))
            Q = sparse.random(oldA.shape[1], oldA.shape[1], 0.9).A
            Q = Q @ Q.T
            pd = A.shape[1] - oldA.shape[1]
            Q = np.pad(Q, ((0, pd), (0, pd)), mode='constant', constant_values=(0, 0))
            tol = 1e-7

            res = solveqp(Q, c, A, b)
            [x, _, _, _, _] = RegularizedPrimalDualMehrotraIPM.solve([Q, c], [A, b], tol, logs=False)
            self.assertTrue(np.allclose(x, res.x, rtol=1.e-3, atol=1.e-3))
