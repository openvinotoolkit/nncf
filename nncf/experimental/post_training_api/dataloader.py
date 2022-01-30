from abc import ABC
from abc import abstractmethod


class DataLoader(ABC):
    """
    Base class provides interface to get elements of the dataset.
    """

    @abstractmethod
    def __getitem__(self, i):
        """
        Returns the i-th element of the dataset.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
