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

from nncf.common.factory import NNCFGraphFactory
from nncf.data import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from tests.post_training.test_templates.helpers import ConvTestModel
from tests.post_training.test_templates.helpers import MultipleConvTestModel
from tests.post_training.test_templates.helpers import SplittedModel
from tests.post_training.test_templates.helpers import StaticDatasetMock

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


# pylint: disable=protected-access
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
    def map_references(ref_biases: Dict) -> Dict[str, List]:
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

    @staticmethod
    @abstractmethod
    def get_ref_path(suffix: str) -> str:
        """
        Returns backend-specific reference graph paths.
        """

    @staticmethod
    @abstractmethod
    def compare_nncf_graphs(model: TModel, ref_path: str) -> None:
        """
        Compares backend-specific model with reference graph.
        """

    @pytest.fixture()
    def quantized_test_model(self, tmpdir) -> TModel:
        model_cls = SplittedModel
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm(disable_bias_correction=True)
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)
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
        ),
    )
    def test_update_bias(self, model_cls, ref_biases, tmpdir):
        model = self.backend_specific_model(model_cls(), tmpdir)
        dataset = Dataset(self.get_dataset(model_cls.INPUT_SIZE), self.get_transform_fn())

        quantization_algorithm = self.get_quantization_algorithm()
        quantized_model = quantization_algorithm.apply(model, dataset=dataset)

        mapped_ref_biases = self.map_references(ref_biases)
        self.check_bias(quantized_model, mapped_ref_biases)

    @pytest.mark.parametrize(
        "layer_name, ref_data",
        (
            (
                "/conv_1/Conv/WithoutBiases",
                {
                    "collected_inputs": {"/conv_1/Conv/WithoutBiases": ("input.1", 0)},
                    "subgraph_data": {
                        "subgraph_input_names": {"/conv_1/Conv/WithoutBiases"},
                        "subgraph_output_names": {"/maxpool_1/MaxPool", "/Split"},
                        "subgraph_output_ids": {("/Split", 0), ("/maxpool_1/MaxPool", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_2/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        "/conv_1/Conv/WithoutBiases": ("input.1", 0),
                        "/conv_2/Conv/WithoutBiases": ("/maxpool_1/MaxPool", 0),
                        "/conv_4/Conv/WithoutBiases": ("/Split", 0),
                        "/conv_6/Conv/WithoutBiases": ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {"/conv_2/Conv/WithoutBiases"},
                        "subgraph_output_names": {"/Relu_1"},
                        "subgraph_output_ids": {("/Relu_1", 0)},
                    },
                },
            ),
            (
                "/conv_3/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        "/conv_1/Conv/WithoutBiases": ("input.1", 0),
                        "/conv_2/Conv/WithoutBiases": ("/maxpool_1/MaxPool", 0),
                        "/conv_3/Conv/WithoutBiases": ("/Relu_1", 0),
                        "/conv_4/Conv/WithoutBiases": ("/Split", 0),
                        "/conv_6/Conv/WithoutBiases": ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {"/conv_1/Conv/WithoutBiases", "/conv_3/Conv/WithoutBiases"},
                        "subgraph_output_names": {"/Split"},
                        "subgraph_output_ids": {("/Split", 0), ("/Split", 1)},
                    },
                },
            ),
            (
                "/conv_4/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        "/conv_4/Conv/WithoutBiases": ("/Split", 0),
                        "/conv_6/Conv/WithoutBiases": ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {"/conv_4/Conv/WithoutBiases"},
                        "subgraph_output_names": {"/Relu_2"},
                        "subgraph_output_ids": {("/Relu_2", 0)},
                    },
                },
            ),
            (
                "/conv_6/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        "/conv_5/Conv/WithoutBiases": ("/Relu_2", 0),
                        "/conv_6/Conv/WithoutBiases": ("/Split", 1),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {"/conv_5/Conv/WithoutBiases", "/conv_6/Conv/WithoutBiases"},
                        "subgraph_output_names": {"/Add_3", "/Concat"},
                        "subgraph_output_ids": {("/Add_3", 0), ("/Concat", 0)},
                    },
                },
            ),
            (
                "/conv_10/Conv/WithoutBiases",
                {
                    "collected_inputs": {
                        "/conv_8/Conv/WithoutBiases": ("/conv_7/Conv", 0),
                        "/conv_9/Conv/WithoutBiases": ("/Add_3", 0),
                        "/conv_10/Conv/WithoutBiases": ("/Concat", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {
                            "/conv_8/Conv/WithoutBiases",
                            "/conv_9/Conv/WithoutBiases",
                            "/conv_10/Conv/WithoutBiases",
                        },
                        "subgraph_output_names": {"/Concat_1"},
                        "subgraph_output_ids": {("/Concat_1", 0)},
                    },
                },
            ),
            (
                "/MatMul",
                {
                    "collected_inputs": {
                        "/MatMul": ("/Reshape", 0),
                    },
                    "subgraph_data": {
                        "subgraph_input_names": {"/MatMul"},
                        "subgraph_output_names": {"/Reshape_1", "/Add_4"},
                        "subgraph_output_ids": {("/Reshape_1", 0), ("/Add_4", 0)},
                    },
                },
            ),
        ),
    )
    def test__get_subgraph_data_for_node(self, quantized_test_model, layer_name, ref_data):
        nncf_graph = NNCFGraphFactory.create(quantized_test_model)

        bc_algo = self.get_bias_correction_algorithm()
        bc_algo._set_backend_entity(quantized_test_model)

        node = nncf_graph.get_node_by_name(layer_name)
        bc_algo._collected_stat_inputs_map.update(ref_data["collected_inputs"])
        subgraph_data = bc_algo._get_subgraph_data_for_node(node, nncf_graph)
        ref_subgraph_data = ref_data["subgraph_data"]

        assert subgraph_data == ref_subgraph_data
