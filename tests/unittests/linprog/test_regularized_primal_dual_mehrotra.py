import unittest
import numpy as np
from scipy.optimize import linprog
import ipsolver


class TestRegularizedPrimalDualMehrotra(unittest.TestCase):
    def test_initial(self):
        A = np.array([[1, 1, 0],
                      [1, 3, 2],
                      [0, 2, 3]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [0], [0], [0]], dtype=np.float64)
        tol = 1e-7

        ip_res = ipsolver.optimize([c], [A, b], method=ipsolver.REGULARIZED_MEHROTRA_METHOD_LP, tol=tol)
        lp_res = linprog(c, A_eq=A, b_eq=b, bounds=((0, None),) * c.shape[0])
        self.assertTrue(np.allclose(ip_res.x, lp_res.x, rtol=1.e-3, atol=1.e-3))

    def test_rank_deficient(self):
        A = np.array([[1, -1, 0, 5, 3],
                      [-4, 3, 2, 0, 4],
                      [0, 2, -3, 1, 2],
                      [0, 2, -3, 1, 2]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9], [9]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [-5], [9], [0], [0], [0], [0]], dtype=np.float64)
        tol = 1e-7

        ip_res = ipsolver.optimize([c], [A, b], method=ipsolver.REGULARIZED_MEHROTRA_METHOD_LP, tol=tol)
        self.assertTrue(ip_res.success)

    def test_random_problems_with_slack_variables(self):
        cnt_experiments = 20
        for _ in range(cnt_experiments):
            A = np.random.rand(np.random.randint(3, 25), np.random.randint(3, 25))*10
            I = np.eye(A.shape[0])
            A = np.c_[A, I]
            b = np.random.rand(A.shape[0], 1)*10
            c = - np.random.rand(A.shape[1], 1)
            tol = 1e-7

            ip_res = ipsolver.optimize([c], [A, b], method=ipsolver.REGULARIZED_MEHROTRA_METHOD_LP, tol=tol)
            lp_res = linprog(c, A_eq=A, b_eq=b, bounds=((0, None),) * c.shape[0])
            self.assertTrue(np.allclose(ip_res.x, lp_res.x))
