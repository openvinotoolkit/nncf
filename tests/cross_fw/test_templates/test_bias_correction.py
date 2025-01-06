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
from typing import Any, Dict, List, Tuple, TypeVar

import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.data import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import DepthwiseConvTestModel
from tests.cross_fw.test_templates.helpers import MultipleConvTestModel
from tests.cross_fw.test_templates.helpers import SplittedModel
from tests.cross_fw.test_templates.helpers import StaticDatasetMock
from tests.cross_fw.test_templates.helpers import TransposeConvTestModel

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
    def fn_to_type(tensor) -> TTensor:
        return tensor

    @staticmethod
    @abstractmethod
    def get_transform_fn() -> callable:
        """
        Get transformation function for dataset.
        """

    def get_dataset(self, input_size: Tuple) -> StaticDatasetMock:
        """
        Return backend specific random dataset.

        :param model: The model for which the dataset is being created.
        """
        return StaticDatasetMock(input_size, self.fn_to_type)

    @staticmethod
    @abstractmethod
    def backend_specific_model(model: TModel, tmp_dir: str) -> TModel:
        """
        Return backend specific model.
        """

    @staticmethod
    @abstractmethod
    def check_bias(model: TModel, ref_biases: Dict) -> None:
        """
        Checks biases values.
        """

    @staticmethod
    def map_references(ref_biases: Dict, model_cls: Any) -> Dict[str, List]:
        """
        Returns backend-specific reference.
        """
        return ref_biases

    @staticmethod
    def get_quantization_algorithm(disable_bias_correction=False) -> PostTrainingQuantization:
        return PostTrainingQuantization(
            subset_size=1,
            fast_bias_correction=False,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, disable_bias_correction=disable_bias_correction
            ),
        )

    @staticmethod
    def get_bias_correction_algorithm() -> BiasCorrection:
        return BiasCorrection(subset_size=1)

    @staticmethod
    @abstractmethod
    def remove_fq_from_inputs(model: TModel) -> TModel:
        """
        Removes quantizer nodes from inputs.
        """

    @pytest.fixture()
    def quantized_test_model(self, tmpdir) -> TModel:
        model_cls = SplittedModel
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm(disable_bias_correction=True)
        graph = NNCFGraphFactory.create(model)
        quantized_model = quantization_algorithm.apply(model, graph, dataset=dataset)
        modified_model = self.remove_fq_from_inputs(quantized_model)
        return modified_model

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
            (DepthwiseConvTestModel, {"/conv/Conv": [-1.1229, -0.1863]}),
            (TransposeConvTestModel, {"/conv/ConvTranspose": [0.66797173, -0.7070703]}),
        ),
    )
    def test_update_bias(self, model_cls, ref_biases, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm()
        graph = NNCFGraphFactory.create(model)
        quantized_model = quantization_algorithm.apply(model, graph, dataset=dataset)

        mapped_ref_biases = self.map_references(ref_biases, model_cls)
        self.check_bias(quantized_model, mapped_ref_biases)

    def test__get_subgraph_data_for_node(self, quantized_test_model, layer_name, ref_data):
        nncf_graph = NNCFGraphFactory.create(quantized_test_model)

        bc_algo = self.get_bias_correction_algorithm()
        bc_algo._set_backend_entity(quantized_test_model)

        node = nncf_graph.get_node_by_name(layer_name)
        bc_algo._collected_stat_inputs_map.update(ref_data["collected_inputs"])
        subgraph_data = bc_algo._get_subgraph_data_for_node(node, nncf_graph)
        ref_subgraph_data = ref_data["subgraph_data"]

        assert subgraph_data == ref_subgraph_data

    def test_verify_collected_stat_inputs_map(self, model_cls, ref_stat_inputs_map, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        graph = NNCFGraphFactory.create(model)

        bc_algo = self.get_bias_correction_algorithm()
        bc_algo.get_statistic_points(model, graph)

        collected_stat_inputs_map = getattr(bc_algo, "_collected_stat_inputs_map")
        assert collected_stat_inputs_map == ref_stat_inputs_map
