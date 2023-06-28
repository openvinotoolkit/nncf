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
from typing import Dict, List, Tuple, TypeVar

import pytest

from nncf.data import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from tests.post_training.test_templates.helpers import ConvTestModel
from tests.post_training.test_templates.helpers import MultipleConvTestModel
from tests.post_training.test_templates.helpers import StaticDatasetMock

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestBCAlgorithm:
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
    def get_backend() -> BiasCorrectionAlgoBackend:
        """
        Get backend specific BiasCorrectionAlgoBackend

        :return BiasCorrectionAlgoBackend: Backend specific BiasCorrectionAlgoBackend
        """

    @staticmethod
    def fn_to_type(tensor):
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn():
        """
        Get transformation function for dataset.
        """

    def get_dataset(self, input_size: Tuple):
        """
        Return backend specific random dataset.

        :param model: The model for which the dataset is being created.
        """
        return StaticDatasetMock(input_size, self.fn_to_type)

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str):
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_bias(model: TModel, ref_biases: Dict):
        """
        Checks biases values.
        """

    @staticmethod
    def map_references(ref_biases: Dict) -> Dict[str, List]:
        """
        Returns backend-specific reference.
        """
        return ref_biases

    @staticmethod
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            fast_bias_correction=False,
            advanced_parameters=AdvancedQuantizationParameters(overflow_fix=OverflowFix.DISABLE),
        )

    @pytest.mark.parametrize(
        "model_cls, ref_biases",
        (
            (
                MultipleConvTestModel,
                {
                    "/conv_1/Conv": [0.6658976, -0.70563036],
                    "/conv_2/Conv": [-0.307696, -0.42806846, 0.44965455],
                    "/conv_3/Conv": [-0.0033792169, 1.0661412],
                    "/conv_4/Conv": [-0.6941606, 0.9958957, 0.6081058],
                    # Disabled latest layer due to backends differences
                    # "/conv_5/Conv": [0.07476559, -0.75797373],
                },
            ),
            (ConvTestModel, {"/conv/Conv": [0.11085186, 1.0017344]}),
        ),
    )
    def test_update_bias(self, model_cls, ref_biases, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm()
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)

        mapped_ref_biases = self.map_references(ref_biases)
        self.check_bias(quantized_model, mapped_ref_biases)
