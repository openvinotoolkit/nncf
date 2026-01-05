# Copyright (c) 2026 Intel Corporation
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
from typing import Any, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters

TModel = TypeVar("TModel")


class Quantizer(ABC):
    """
    Quantizer is an interface for the RangeEstimator algorithm
    which specifies all the required methods to retrieve quantization setup from the given model.
    """

    @abstractmethod
    def transform_prior_quantization(self, model: TModel) -> TModel:
        """
        Transforms the given model in-place with the necessary modifications required prior to quantization.

        :param model: Backend-specific model to be transformed.
        :return: Transformed backend-specific model.
        """

    @abstractmethod
    def get_quantization_setup(self, model: TModel, nncf_graph: NNCFGraph) -> SingleConfigQuantizerSetup:
        """
        Builds SingleConfigQuantizerSetup for the given model.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: SingleConfigQuantizerSetup for the given model.
        """

    @abstractmethod
    def get_weight_compression_parameters(
        self,
        model: TModel,
        nncf_graph: NNCFGraph,
    ) -> tuple[
        list[WeightCompressionParameters],
        list[WeightCompressionParameters],
        list[WeightCompressionParameters],
    ]:
        """
        Obtains the weight compression parameters from the quantizer which can be used to determine
        weights to compress, weights to skip, weights to consider for mixed precision assignment.

        :param model: Backend-specific model.
        :param nncf_graph: NNCFGraph instance.
        :return: Tuple of (all_weight_params, ratio_defining_params, skipped_weight_params) where:
            1. all_weight_params: all compressible weight parameters in the model
            2. ratio_defining_params: subset of weights used for mixed precision assignment
            3. skipped_weight_params: weights that should be excluded from compression
        """

    @abstractmethod
    def get_weight_compression_config(self) -> dict[str, Any]:
        """
        Returns the weight compression configuration as a dictionary.

        :return: Dictionary containing compression configuration parameters obtained
            from the quantizer.
        """
