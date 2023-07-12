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
from typing import List, Tuple, TypeVar

import pytest

import nncf
from nncf.data import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from tests.post_training.test_templates.helpers import ConvTestModel
from tests.post_training.test_templates.helpers import StaticDatasetMock

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestMinMaxAlgorithm:
    @staticmethod
    @abstractmethod
    def list_to_backend_type(data: List) -> TTensor:
        """
        Convert list to backend specific type

        :param data: List of data.
        :return: Converted data.
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
    def get_quantization_algorithm():
        return PostTrainingQuantization(
            subset_size=1,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE,
                disable_bias_correction=True,
            ),
        )

    @pytest.mark.parametrize("model_cls", (ConvTestModel,))
    def test_quantization(self, model_cls, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm()
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)
