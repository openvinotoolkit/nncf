# Copyright (c) 2023 Intel Corporation
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
from typing import Callable, Dict, TypeVar

import pytest

from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from tests.post_training.test_templates.helpers import LinearModel
from tests.post_training.test_templates.helpers import get_static_dataset

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestSQAlgorithm:
    @staticmethod
    def fn_to_type(tensor) -> TTensor:
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn() -> Callable:
        """
        Get transformation function for dataset.
        """

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str) -> TModel:
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_scales(model: TModel, reference_values: Dict[str, TTensor]) -> None:
        """
        Checking scales from model with references.
        """

    @staticmethod
    @abstractmethod
    def get_backend() -> SmoothQuantAlgoBackend:
        """
        Returns backend-specific SmoothQuantAlgoBackend.
        """

    @staticmethod
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            model_type=ModelType.TRANSFORMER,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, smooth_quant_alpha=0.95
            ),
        )

    @pytest.mark.parametrize(
        "model_cls, reference_values",
        (
            (
                LinearModel,
                {"/Reshape/smooth_quant_multiply": [[[1.0708091, 1.0854627, 1.2070981, 1.1213733]]]},
            ),
        ),
    )
    def test_smooth_quant_algo(self, model_cls, reference_values, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = get_static_dataset(model_cls.INPUT_SIZE, self.get_transform_fn(), self.fn_to_type)

        quantization_algorithm = self.get_quantization_algorithm()
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)

        self.check_scales(quantized_model, reference_values)

    # pylint:disable=protected-access
    def test_smooth_quant(self):
        backend = self.get_backend()
        reduction_shape = (3, 2, 1)
        samples = 1

        for inplace_type in [False, True]:
            backend_tensor_collector = backend.get_abs_max_channel_collector(
                num_samples=samples,
                stats_reduction_shape=reduction_shape,
                inplace=inplace_type,
                branch_key="test_branch",
            )

            for aggregator in backend_tensor_collector.aggregators.values():
                assert isinstance(aggregator, MaxAggregator)

            for reducer in backend_tensor_collector.reducers:
                assert isinstance(reducer, AbsMaxReducer)
                assert reducer.inplace == inplace_type
                assert reducer._reduction_shape == reduction_shape
