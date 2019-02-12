from abc import ABCMeta, abstractmethod
from utils.logger import get_logger
import numpy as np
from typing import List

Matrix = np.array
Vector = np.array
Array = np.array


class InterfaceIPM(metaclass=ABCMeta):
    """ Interface for Interior point methods. """

    UNIT_STEP_LENGTH = 1.0
    logger = get_logger()

    @classmethod
    def _check_exit_conditions(cls,
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
        if np.max(residuals_norm) < tol and cls.mu < tol or iter_num > max_iter:
            return True
        return False

    @classmethod
    @abstractmethod
    def _compute_residuals(cls,
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

    @classmethod
    def _compute_norm_of_residuals(cls,
                                   residuals: List[Vector],
                                   order=np.inf) -> Vector:
        """ Computes norms of residuals.

        Args:
            residuals: list of Vectors which are residuals.
            order: the type of norm (look at the numpy.linalg.norm documentation).

        Returns:
            Vector of norm of residuals.
        """
        return np.array([np.linalg.norm(residual, ord=order) for residual in residuals])

    @classmethod
    @abstractmethod
    def _variables_initialization(cls, constraints: List[Array]) -> List[Vector]:
        """ Initializes variables which are the point in N-dim space.

        Args:
            constraints: list of Matrices and Vectors which are the constraints in optimization problem.
        """

    @classmethod
    def _parameters_initialization(cls):
        """ Initializes set of parameters which is required for optimization. """
        cls.mu = 1

    @classmethod
    @abstractmethod
    def _constants_initialization(cls, *args):
        """ Initialize constants which are required for optimization loop. """

    @classmethod
    @abstractmethod
    def _log_iterations(cls, *args, **kwargs):
        """" Logging of optimization process. """

    @classmethod
    @abstractmethod
    def _build_jacobian(cls,
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

    @classmethod
    @abstractmethod
    def _get_step_length(cls, *args, **kwargs):
        """ Returns optimal step length.

        Optimality with respect to the fact that the next step should be in the neighbourhood of the
        central path"""

    @classmethod
    def _newton_step(cls, jac: Matrix, rhs: Vector) -> Vector:
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

    @classmethod
    @abstractmethod
    def solve(cls, *args, **kwargs):
        """ Solves a optimization problem. """
        pass
