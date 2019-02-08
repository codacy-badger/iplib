import unittest
import numpy as np
from scipy.optimize import linprog
from ipms.LP.merothra_ipm import MehrotraIPM


class TestMehrotra(unittest.TestCase):
    def test_initial(self):
        A = np.array([[1, 1, 0],
                      [1, 3, 2],
                      [0, 2, 3]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [0], [0], [0]], dtype=np.float64)
        tol = 1e-7

        [x, s, y] = MehrotraIPM.solve([c], [A, b], tol)
        res = linprog(c, A_eq=A, b_eq=b, bounds=((0, None),) * c.shape[0])
        self.assertTrue(np.allclose(x, res.x))

    def test_rank_deficient(self):
        A = np.array([[1, -1, 0, 5, 3],
                      [-4, 3, 2, 0, 4],
                      [0, 2, -3, 1, 2],
                      [0, 2, -3, 1, 2]], dtype=np.float64)
        I = np.eye(A.shape[0])
        A = np.c_[A, I]
        b = np.array([[7], [15], [9], [8]], dtype=np.float64)
        c = - np.array([[3], [13], [13], [-5], [9], [0], [0], [0]], dtype=np.float64)
        tol = 1e-7

        self.assertRaises(ValueError, MehrotraIPM.solve, [c], [A, b], tol)

    def test_10_random_problems_with_slack_variables(self):
        for _ in range(10):
            A = np.random.rand(np.random.randint(3, 25), np.random.randint(3, 25))*10
            I = np.eye(A.shape[0])
            A = np.c_[A, I]
            b = np.random.rand(A.shape[0], 1)*10
            c = - np.random.rand(A.shape[1], 1)
            tol = 1e-7

            [x, y, s] = MehrotraIPM.solve([c], [A, b], tol)
            res = linprog(c, A_eq=A, b_eq=b, bounds=((0, None),) * c.shape[0])
            self.assertTrue(np.allclose(x, res.x))

    def test_20_random_rank_deficient_problems(self):
        for _ in range(20):
            A = np.random.rand(np.random.randint(3, 25), np.random.randint(3, 25))*10
            cnt_rows = A.shape[0]
            b = np.random.rand(cnt_rows, 1) * 10

            A = np.r_[A, A[0:int(cnt_rows * 0.4), :]]
            b = np.r_[b, b[0:int(cnt_rows * 0.4), :]]

            c = - np.random.rand(A.shape[1], 1)
            tol = 1e-7

            self.assertRaises(ValueError, MehrotraIPM.solve, [c], [A, b], tol)
