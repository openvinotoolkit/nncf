"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

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
        # TODO (kshpv): add support batch_size
        self.batch_size = 1
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
