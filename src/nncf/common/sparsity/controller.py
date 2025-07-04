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

from abc import abstractmethod

from nncf.api.compression import CompressionAlgorithmController


class SparsityController(CompressionAlgorithmController):
    """
    This is the class from which all sparsity controllers inherit.
    """

    def set_sparsity_level(self, sparsity_level: float) -> None:
        """
        Sets the sparsity level that should be applied to the model's weights.

        :param sparsity_level: Sparsity level that should be applied to the model's weights.
        """

    def freeze(self) -> None:
        """
        Freezes all sparsity masks. Sparsity masks will not be trained after calling this method.
        """

    @property
    @abstractmethod
    def current_sparsity_level(self) -> float:
        """
        Returns the current sparsity level of the underlying model.
        """
