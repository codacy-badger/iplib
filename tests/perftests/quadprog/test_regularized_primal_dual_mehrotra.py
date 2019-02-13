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
    def test_random_problems_without_slack_variables_and_down_rectangular_matrix(self):
        cntr_ok = 0
        cntr_fail = 0
        cnt_experiments = 20
        for _ in range(cnt_experiments):
            cnt_rows = np.random.randint(5, 25)
            A = np.random.rand(cnt_rows, np.random.randint(2, cnt_rows - 1))*10
            b = np.random.rand(A.shape[0], 1)*10
            c = - np.random.rand(A.shape[1], 1)
            Q = sparse.random(A.shape[1], A.shape[1], 0.9).A
            Q = Q @ Q.T
            tol = 1e-5

            try:
                # res = solveqp(Q, c, A, b)
                [x, _, _, _, _] = RegularizedPrimalDualMehrotraIPM.solve([Q, c], [A, b], tol, logs=False)

                # if (x.T @ Q @ x + x @ c)[0] <= (res.x.T @ Q @ res.x + res.x @ c)[0] and np.linalg.norm(A@x-b, np.inf) < np.linalg.norm(A@res.x-b, np.inf):
                cntr_ok += 1
            except ValueError:
                cntr_fail += 1
        print("{} - Percentage when Jacobian is not rank-deficient: {:.2f}%".format(
            self.test_random_problems_without_slack_variables_and_down_rectangular_matrix.__name__,
            cntr_ok / cnt_experiments * 100))

    def test_random_problems_without_slack_variables_and_right_rectangular_matrix(self):
        cntr_ok = 0
        cntr_fail = 0
        cnt_experiments = 20
        primal_resid = 0
        for _ in range(cnt_experiments):
            cnt_rows = np.random.randint(5, 25)
            A = np.random.rand(cnt_rows, np.random.randint(cnt_rows, 26))*10
            b = np.random.rand(A.shape[0], 1)*10
            c = - np.random.rand(A.shape[1], 1)
            tol = 1e-4
            Q = sparse.random(A.shape[1], A.shape[1], 0.9).A
            Q = Q @ Q.T

            try:
                [x, _, _, _, _] = RegularizedPrimalDualMehrotraIPM.solve([Q, c], [A, b], tol, logs=False)
                res = solveqp(Q, c, A, b)
                primal_resid += np.linalg.norm(A@x-b) / x.shape[0]
                if (x.T @ Q @ x + x @ c)[0] <= (res.x.T @ Q @ res.x + res.x @ c)[0] and np.linalg.norm(A@x-b) < np.linalg.norm(A@res.x-b):
                    cntr_ok += 1
            except ValueError:
                cntr_fail += 1
        print("{} - Percentage when IP is better then SLSQPx: {:.2f}, Percentage of fails: {:.2f}%, Mean primal residual: {:.4f}".format(
            self.test_random_problems_without_slack_variables_and_right_rectangular_matrix.__name__,
            cntr_ok / cnt_experiments * 100, cntr_fail / cnt_experiments * 100, primal_resid / cnt_experiments))

    def test_random_rank_deficient_problems(self):
        cntr_ok = 0
        cntr_fail = 0
        cnt_experiments = 20
        for _ in range(cnt_experiments):
            A = np.random.rand(np.random.randint(3, 25), np.random.randint(3, 25))*10
            cnt_rows = A.shape[0]
            b = np.random.rand(cnt_rows, 1) * 10

            A = np.r_[A, A[0:int(cnt_rows * 0.4), :]]
            b = np.r_[b, b[0:int(cnt_rows * 0.4), :]]

            c = - np.random.rand(A.shape[1], 1)

            Q = sparse.random(A.shape[1], A.shape[1], 0.9).A
            Q = Q @ Q.T
            tol = 1e-7

            try:
                [x, _, _, _, _] = RegularizedPrimalDualMehrotraIPM.solve([Q, c], [A, b], tol, logs=False)
                #res = solveqp(Q, c, A, b)
                #if (x.T @ Q @ x + x @ c)[0] <= (res.x.T @ Q @ res.x + res.x @ c)[0] and np.linalg.norm(
                #        A @ x - b) < np.linalg.norm(A @ res.x - b):
                cntr_ok += 1
            except ValueError:
                cntr_fail += 1
        print("{} - Percentage when Jacobian is not rank-deficient: {:.2f}%".format(
            self.test_random_rank_deficient_problems.__name__,
            cntr_ok / cnt_experiments * 100))
