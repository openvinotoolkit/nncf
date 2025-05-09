# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator

from nncf.common.graph.graph import NNCFNode
from nncf.tensor import Tensor


class LayerwiseIterator(Iterator, ABC):
    """
    An abstract base class for iterating through the layers of a model.
    """

    @abstractmethod
    def __next__(self) -> tuple[NNCFNode, dict[int, list[Tensor]]]:
        """
        Returns the next node and its associated tensors in the iteration.

        :return: The next node and its associated tensors in the iteration.
        :raises StopIteration: When there are no more elements to iterate over.
        """
