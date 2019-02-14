from ipsolver import base_ipm
from abc import abstractmethod
from utils.logger import get_stdout_handler
import numpy as np
from collections import namedtuple


_Result = namedtuple('Result', ['success', 'x', 'dual', 'f'])


class BaseIPM(base_ipm.BaseIPM):
    """ Base class for Mehrotra interior point method in linear programming. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _predictor_step(self, constraints, variables, residuals, jacobian):
        """ Predictor part of Mehrotra IP method. """

    @abstractmethod
    def _corrector_step(self, predictor_metadata, constraints, variables, residuals, jacobian):
        """ Corrector part of Mehrotra IP method. """

    @abstractmethod
    def _update_variables(self, variables, direction, step_length):
        """ Updates point position with respect to steepest direction and the optimal step length. """

    @abstractmethod
    def _compute_function_value(self, cost_function, point):
        """ Computes the function value in point. """

    def solve(self, cost_function, constraints, tol=1e-8,  max_iter=np.inf, logs=False):
        """ Solves a linear programming optimization problem. """
        if logs:
            self._logger.addHandler(get_stdout_handler())

        self._constants_initialization(constraints)
        self._parameters_initialization()
        variables = self._variables_initialization(constraints)

        success = True
        step_length = 1.0
        self._log_iterations()
        while True:
            residuals = self._compute_residuals(cost_function, constraints, variables)
            residuals_norm = self._compute_norm_of_residuals(residuals)

            if self._check_exit_conditions(residuals_norm, tol, max_iter):
                self._iter_num += 1
                self._log_iterations(residuals_norm, step_length)
                break

            jacobian = self._build_jacobian(cost_function, constraints, variables)
            try:
                metadata = self._predictor_step(constraints, variables, residuals, jacobian)
                direction, step_length = self._corrector_step(metadata, constraints, variables, residuals, jacobian)
            except base_ipm.JacobianIsRankDeficient:
                success = False
                break

            variables = self._update_variables(variables, direction, step_length)

            self._iter_num += 1
            self._log_iterations(residuals_norm, step_length)

        variables = [np.ravel(var) for var in variables]
        f = self._compute_function_value(cost_function, variables[0])
        return _Result(success=success, x=variables[0], dual=variables[1:], f=f)
