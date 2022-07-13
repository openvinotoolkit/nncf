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

from typing import Dict

from abc import ABC
from abc import abstractmethod

from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.tensor import NNCFTensor

NNCFData = Dict[str, NNCFTensor]


class Dataset(ABC):
    """
    Base class provides interface to get elements of the dataset.
    """

    def __init__(self, batch_size: int = 1, shuffle: bool = True, has_batch_dim: bool = True):
        """
        FasterRCNN-12 and MaskRCNN-12 models have no batch dimension.
        """
        # TODO (kshpv): add support batch_size
        if batch_size != 1:
            nncf_logger.warning(
                f"We don't support batch_size={batch_size} > 1 yet. Set batch_size=1")
        self.batch_size = 1
        self.shuffle = shuffle
        self.has_batch_dim = has_batch_dim

    @abstractmethod
    def __getitem__(self, i: int) -> NNCFData:
        """
        Returns the i-th element of the dataset with the target value.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
