import numpy as np

from abc import ABCMeta, abstractmethod
from typing import List
from utils.typing import Matrix, Vector, Array
from utils.logger import get_logger


class BaseIPM(metaclass=ABCMeta):
    """ Base class for interior point methods. """

    _UNIT_STEP_LENGTH = 1.0

    def __init__(self):
        self._logger = get_logger()

    def _check_exit_conditions(self,
                               residuals_norm: Vector,
                               tol: float,
                               iter_num: int,
                               max_iter: int) -> bool:
        """ Checks condition for exit from the loop.


        Args:
            residuals_norm: norms of the residuals from the right hand side of Newton system.
            tol: tolerance of the algorithm.
            iter_num: current iteration number.
            max_iter: maximum number of iterations.


        Returns:
            True if condition satisfied, else False.
        """
        # TODO: fix issue with cls.mu
        if np.max(residuals_norm) < tol and self.mu < tol or iter_num > max_iter:
            return True
        return False

    @abstractmethod
    def _compute_residuals(self,
                           cost_function: List[Array],
                           constraints: List[Array],
                           variables: List[Vector]) -> List[Vector]:
        """ Computes residuals in the right hand side of Newton system.


        Args:
            cost_function: list of Vectors and Matrices from definition of the cost function.
            constraints: list of Vectors Matrices which relates to the optimization problem constraints.
            variables: list of variables (primal and dual) which take a part in calculation.


        Returns:
            list of Vectors which are residuals evaluated and the point from 'variables'.
        """

    @staticmethod
    def _compute_norm_of_residuals(residuals: List[Vector],
                                   order=np.inf) -> Vector:
        """ Computes norms of residuals.


        Args:
            residuals: list of Vectors which are residuals.
            order: the type of norm (look at the numpy.linalg.norm documentation).


        Returns:
            Vector of norm of residuals.
        """
        return np.array([np.linalg.norm(residual, ord=order) for residual in residuals])

    @abstractmethod
    def _variables_initialization(self, constraints: List[Array]) -> List[Vector]:
        """ Initializes variables which are the point in N-dim space.

        Args:
            constraints: list of Matrices and Vectors which are the constraints for optimization problem.
        """

    def _parameters_initialization(self):
        """ Initializes set of parameters which is required for optimization. """
        self.mu = 1.0

    @abstractmethod
    def _constants_initialization(self, *args):
        """ Initializes constants which are required for optimization loop. """

    @abstractmethod
    def _log_iterations(self, *args, **kwargs):
        """" Logs iterations in optimization process. """

    @abstractmethod
    def _build_jacobian(self,
                        cost_function: List[Array],
                        constraints: List[Array],
                        variables: List[Vector]) -> Matrix:
        """ Builds Jacobian for Newton step.


        Args:
            cost_function: list of Vectors and Matrices from definition of the cost function.
            constraints: list of Vectors Matrices which relates to the optimization problem constraints.
            variables: list of variables (primal and dual) which take a part in calculation.


        Returns:
            Jacobian matrix.
        """

    @abstractmethod
    def _get_step_length(self, *args, **kwargs):
        """ Returns optimal step length for Newton step.

        Optimality with respect to the fact that the next step should be in the neighbourhood of the
        central path
        """

    @staticmethod
    def _newton_step(jac: Matrix, rhs: Vector) -> Vector:
        """ Solve a Newton system and returns the steepest decent direction.

        Args:
            jac: Jacobian matrix.
            rhs: Vector of right hand side of Newton system.

        Returns:
            Vector which is the steepest decent direction.
        """
        if np.linalg.matrix_rank(jac) == jac.shape[0]:
            return np.linalg.solve(jac, rhs)
        else:
            raise ValueError("Jacobian of Newton system is rank-deficient.")

    @abstractmethod
    def solve(self, *args, **kwargs):
        """ Solves a optimization problem. """
