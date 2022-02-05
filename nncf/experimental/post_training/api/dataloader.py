from typing import Tuple
from typing import TypeVar

from abc import ABC
from abc import abstractmethod

Input = TypeVar('Input')
Target = TypeVar('Target')


class DataLoader(ABC):
    """
    Base class provides interface to get elements of the dataset.
    """

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    @abstractmethod
    def __getitem__(self, i) -> Tuple[Input, Target]:
        """
        Returns the i-th element of the dataset.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
