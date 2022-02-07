from typing import Tuple
from typing import TypeVar

from abc import ABC
from abc import abstractmethod

ModelInput = TypeVar('ModelInput')
Target = TypeVar('Target')


class DataLoader(ABC):
    """
    Base class provides interface to get elements of the dataset.
    """

    def __init__(self, batch_size=1, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def __getitem__(self, i: int) -> Tuple[ModelInput, Target]:
        """
        Returns the i-th element of the dataset with the target value.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
