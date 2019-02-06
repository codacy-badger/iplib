import numpy as np
from abc import ABCMeta, abstractmethod
from ipms.logger import get_logger


class AbstractIPM(metaclass=ABCMeta):
    """ Interface for Interior point ipms. """

    UNIT_STEP_LENGTH = 1.0
    logger = get_logger()

    @staticmethod
    @abstractmethod
    def _build_jacobian(*args, **kwargs):
        """ Builds Jacobian for Newton step"""
        pass

    @staticmethod
    @abstractmethod
    def _get_step_length(*args, **kwargs):
        """ Computes optimal step length

        Optimality with respect to the fact that the
        next step should be in the neighbourhood of the
        central path"""
        pass

    @staticmethod
    def _newton_step(jac, rhs):
        if np.linalg.matrix_rank(jac) == jac.shape[0]:
            return np.linalg.solve(jac, rhs)
        else:
            raise ValueError("Jacobian of Newton system is rank-deficient.")

    @staticmethod
    @abstractmethod
    def solve(*args, **kwargs):
        pass
