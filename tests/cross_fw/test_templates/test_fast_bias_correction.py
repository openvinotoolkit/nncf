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
from typing import List, TypeVar

import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from tests.cross_fw.test_templates.helpers import ConvBNTestModel
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import get_static_dataset

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestFBCAlgorithm:
    @staticmethod
    @abstractmethod
    def list_to_backend_type(data: List) -> TTensor:
        """
        Convert list to backend specific type

        :param data: List of data.

        :return: Converted data.
        """

    @staticmethod
    @abstractmethod
    def get_backend() -> FastBiasCorrectionAlgoBackend:
        """
        Get backend specific FastBiasCorrectionAlgoBackend

        :return FastBiasCorrectionAlgoBackend: Backend specific FastBiasCorrectionAlgoBackend
        """

    @pytest.mark.parametrize(
        "bias_value, bias_shift, channel_axis, ref_shape",
        (
            ([1, 1], [0.1, 0.1], 1, [2]),
            ([[1, 1]], [0.1, 0.1], -1, [1, 2]),
            ([[1, 1]], [0.1, 0.1], 1, [1, 2]),
        ),
    )
    def test_reshape_bias_shift(self, bias_value: list, bias_shift: list, channel_axis: int, ref_shape: list):
        """
        Checks the result of the FastBiasCorrection.reshape_bias_shift method for backend specific datatype.
        """
        bias_value = self.list_to_backend_type(data=bias_value)
        bias_shift = self.list_to_backend_type(data=bias_shift)

        algo = FastBiasCorrection(subset_size=1, inplace_statistics=False)

        algo._backend_entity = self.get_backend()
        new_bias_shift = algo._reshape_bias_shift(bias_shift, bias_value, channel_axis)
        assert list(new_bias_shift.shape) == ref_shape

    @staticmethod
    def fn_to_type(tensor):
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn():
        """
        Get transformation function for dataset.
        """

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str):
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_bias(model: TModel, ref_bias: list):
        """
        Return backend specific model.
        """

    @staticmethod
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            fast_bias_correction=True,
            advanced_parameters=AdvancedQuantizationParameters(overflow_fix=OverflowFix.DISABLE),
        )

    @pytest.mark.parametrize(
        "model_cls, ref_bias",
        (
            (ConvTestModel, [0.0288348, 1.0838453]),
            (ConvBNTestModel, [0.08396978, 1.1676897]),
        ),
    )
    def test_update_bias(self, model_cls, ref_bias, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = get_static_dataset(model_cls.INPUT_SIZE, self.get_transform_fn(), self.fn_to_type)

        quantization_algorithm = self.get_quantization_algorithm()
        graph = NNCFGraphFactory.create(model)
        quantized_model = quantization_algorithm.apply(model, graph, dataset=dataset)

        self.check_bias(quantized_model, ref_bias)
