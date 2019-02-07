from abc import ABCMeta, abstractmethod
from ipms.logger import get_logger


class AbstractIPM(metaclass=ABCMeta):
    """ Interface for Interior point ipms. """

    UNIT_STEP_LENGTH = 1.0
    logger = get_logger()

    @classmethod
    @abstractmethod
    def _build_jacobian(cls, *args, **kwargs):
        """ Builds Jacobian for Newton step"""
        pass

    @classmethod
    @abstractmethod
    def _get_step_length(cls, *args, **kwargs):
        """ Computes optimal step length

        Optimality with respect to the fact that the
        next step should be in the neighbourhood of the
        central path"""
        pass

    @classmethod
    @abstractmethod
    def _newton_step(cls, jac, rhs):
        pass

    @classmethod
    @abstractmethod
    def solve(cls, *args, **kwargs):
        pass
