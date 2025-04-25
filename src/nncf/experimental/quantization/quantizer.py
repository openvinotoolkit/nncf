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
from enum import Enum
from typing import Any, Optional, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup

TModel = TypeVar("TModel")


class IntDtype(Enum):
    INT8 = "INT8"
    UINT8 = "UINT8"


class ExtendedQuantizerSetup(ABC, SingleConfigQuantizerSetup):
    """
    Quantizer setup with additional info required to insert
    quantizers to torch.fx models.
    """

    @abstractmethod
    def get_extra_params(self) -> dict[QuantizationPointId, dict[str, Any]]:
        """
        Returns extra params
        """


class ExtendedFXQuantizerSetup(ExtendedQuantizerSetup):
    """
    Quantizer setup with additional info required to insert
    quantizers to torch.fx models.
    """

    QUANTIZER_DTYPE_NAME = "quantizer_dtype"

    def __init__(self) -> None:
        super().__init__()
        self._quantization_dtypes: dict[QuantizationPointId, Optional[IntDtype]] = {}

    def add_independent_quantization_point(
        self, qp: QuantizationPointBase, intermediate_dtype: Optional[IntDtype]
    ) -> QuantizationPointId:
        id = super().add_independent_quantization_point(qp)
        self._quantization_dtypes[id] = intermediate_dtype
        return id

    def get_extra_params(self) -> dict[int, dict[str, Any]]:
        return {k: {self.QUANTIZER_DTYPE_NAME: v} for k, v in self._quantization_dtypes.items()}

    def get_state(self) -> dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        base_state = super().get_state()
        base_state[self.QUANTIZER_DTYPE_NAME] = {
            qp_id: dtype.value for qp_id, dtype in self.quantization_points.items()
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ExtendedFXQuantizerSetup":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        state_ = state.copy()
        dtype_names = state_.pop(cls.QUANTIZER_DTYPE_NAME)
        super_setup = super().from_state(state_)
        setup = ExtendedFXQuantizerSetup()

        setup.quantization_points = super_setup.quantization_points
        setup.unified_scale_groups = super_setup.unified_scale_groups
        setup.shared_input_operation_set_groups = super_setup.shared_input_operation_set_groups
        setup._quantization_dtypes = {
            qp_id: None if name is None else IntDtype[name] for qp_id, name in dtype_names.items()
        }

        return setup


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
    def get_quantization_setup(self, model: TModel, nncf_graph: NNCFGraph) -> ExtendedFXQuantizerSetup:
        """
        Builds SingleConfigQuantizerSetup for the given model.

        :param model: Backend-specific model, for which Quantization Target Points are being seek.
        :param nncf_graph: NNCFGraph instance.
        :return: SingleConfigQuantizerSetup for the given model.
        """
