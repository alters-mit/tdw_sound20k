from abc import ABC, abstractmethod
from typing import Tuple
from numpy.random import RandomState


class TDWSceneSize(ABC):
    """
    The size of a TDW scene room.
    """

    @abstractmethod
    def get_size(self) -> Tuple[int, int]:
        """
        :return: The dimensions of the room.
        """

        raise Exception()


class StandardSize(TDWSceneSize):
    """
    A "normal-sized" room.
    """

    def get_size(self) -> Tuple[int, int]:
        return 12, 12


class SmallSize(TDWSceneSize):
    """
    A small room.
    """

    def get_size(self) -> Tuple[int, int]:
        return 4, 4


class RandomSize(TDWSceneSize):
    """
    A randomly-sized room.
    """

    _RNG = RandomState(0)

    def get_size(self) -> Tuple[int, int]:
        return RandomSize._RNG.randint(4, 16), RandomSize._RNG.randint(4, 16)
