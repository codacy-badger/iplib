""" Collection of Interior point methods.

This collection contains IP methods for solving linear and
quadratic optimization problems.

LP:
  - Mehrotra method
  - Regularized primal dual Mehrotra method
QP:
  - Regularized primal dual Mehrotra method
"""

import numpy as np
from . import linprog
from . import quadprog
from . import sdp
from . import socp

AUTO_METHOD = "auto"
MEHROTRA_METHOD_LP = "mehrotra_lp"
MEHROTRA_METHOD_QP = "mehrotra_qp"
REGULARIZED_MEHROTRA_METHOD_LP = "regularized_mehrotra_lp"
REGULARIZED_MEHROTRA_METHOD_QP = "regularized_mehrotra_qp"


def optimize(cost_function, constraints, method=AUTO_METHOD, tol=1e-8,  max_iter=np.inf, logs=False):
    if method == AUTO_METHOD:
        method = _detect_solver(cost_function)
    solver = _get_solver(method)
    return solver.solve(cost_function, constraints, tol, max_iter, logs)


def _get_solver(method_name):
    if method_name == MEHROTRA_METHOD_LP:
        return linprog.mehrotra.mehrotra_ipm.IPM()
    elif method_name == REGULARIZED_MEHROTRA_METHOD_LP:
        return linprog.mehrotra.regularized_mehrotra_ipm.RegularizedMehrotraIPM()
    elif method_name == MEHROTRA_METHOD_QP:
        raise NotImplementedError
    elif method_name == REGULARIZED_MEHROTRA_METHOD_QP:
        return quadprog.regularized_mehrotra_ipm.RegularizedMehrotraIPM()
    else:
        raise ValueError(f"Method with name '{method_name}' could not be found.")


def _detect_solver(cost_function):
    if len(cost_function) == 1:
        return REGULARIZED_MEHROTRA_METHOD_LP
    elif len(cost_function) == 2:
        return REGULARIZED_MEHROTRA_METHOD_QP
    else:
        raise ValueError("Can not detect the solver automatically. Please specify the one using 'method' parameter.")
