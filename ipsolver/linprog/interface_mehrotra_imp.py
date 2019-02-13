from ipsolver import base_ipm
from abc import abstractmethod
from utils.logger import get_stdout_handler
import numpy as np


class BaseMehrotraIPM(base_ipm.BaseIPM):
    """ Interface for Mehrotra interior point method for linear programming. """

    @classmethod
    @abstractmethod
    def _predictor_step(cls, constraints, variables, residuals, jacobian):
        """ Predictor part of Mehrotra IP method. """

    @classmethod
    @abstractmethod
    def _corrector_step(cls, predictor_metadata, constraints, variables, residuals, jacobian):
        """ Corrector part of Mehrotra IP method. """

    @classmethod
    @abstractmethod
    def _update_variables(cls, variables, direction, step_length):
        """ Updates point position with respect to steepest direction and the step length. """

    @classmethod
    def _clear_globals(cls):
        """ Clean all variables which are class members. """
        pass

    @classmethod
    def solve(cls, cost_function, constraints, tol=1e-8,  max_iter=np.inf, logs=False):
        """ Solves a linear programming optimization problem. """
        if logs:
            cls.logger.addHandler(get_stdout_handler())

        cls._constants_initialization(constraints)
        cls._parameters_initialization()
        variables = cls._variables_initialization(constraints)

        cls._log_iterations()

        cls.iter_num = 0
        while True:
            residuals = cls._compute_residuals(cost_function, constraints, variables)
            residuals_norm = cls._compute_norm_of_residuals(residuals)

            if cls._check_exit_conditions(residuals_norm, tol, cls.iter_num, max_iter):
                cls.iter_num += 1
                cls._log_iterations(residuals_norm, step_length)
                break

            jacobian = cls._build_jacobian(cost_function, constraints, variables)
            metadata = cls._predictor_step(constraints, variables, residuals, jacobian)
            direction, step_length = cls._corrector_step(metadata, constraints, variables, residuals, jacobian)
            variables = cls._update_variables(variables, direction, step_length)

            cls.iter_num += 1
            cls._log_iterations(residuals_norm, step_length)
        cls._clear_globals()
        return [np.ravel(var) for var in variables]









