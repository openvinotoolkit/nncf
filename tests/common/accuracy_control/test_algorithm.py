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

import logging
from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

from nncf.common.graph.graph import NNCFNode
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorerReport
from nncf.quantization.algorithms.accuracy_control.algorithm import _create_message
from tests.common.accuracy_control.backend import AABackendForTests
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_mergeable_pattern


def test_create_message():
    nodes = [
        NNCFNode(
            {
                NNCFNode.NODE_NAME_ATTR: "node_name_1",
            }
        ),
        NNCFNode(
            {
                NNCFNode.NODE_NAME_ATTR: "node_name_2",
            }
        ),
    ]
    result = _create_message(nodes)
    assert result == "\tnode_name_1\n\tnode_name_2"


@pytest.mark.parametrize(
    "removed_groups, expected_removed_quantizers, expected_reverted_operations",
    [
        # all empty
        [
            [
                type(
                    "MockGroup",
                    (),
                    {
                        "quantizers": [],
                        "operations": [],
                    },
                )
            ],
            [],
            [],
        ],
        # one of each
        [
            [
                type(
                    "MockGroup",
                    (),
                    {
                        "quantizers": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_1",
                                }
                            )
                        ],
                        "operations": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_2",
                                }
                            )
                        ],
                    },
                )
            ],
            ["node_name_1"],
            ["node_name_2"],
        ],
        # multiple reverted ops
        [
            [
                type(
                    "MockGroup",
                    (),
                    {
                        "quantizers": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_1",
                                }
                            )
                        ],
                        "operations": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_2",
                                }
                            ),
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_3",
                                }
                            ),
                        ],
                    },
                )
            ],
            ["node_name_1"],
            ["node_name_2", "node_name_3"],
        ],
        # multiple quantizer groups
        [
            [
                type(
                    "MockGroup",
                    (),
                    {
                        "quantizers": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_1",
                                }
                            )
                        ],
                        "operations": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_2",
                                }
                            ),
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_3",
                                }
                            ),
                        ],
                    },
                ),
                type(
                    "MockGroup",
                    (),
                    {
                        "quantizers": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_4",
                                }
                            )
                        ],
                        "operations": [
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_5",
                                }
                            ),
                            NNCFNode(
                                {
                                    NNCFNode.NODE_NAME_ATTR: "node_name_6",
                                }
                            ),
                        ],
                    },
                ),
            ],
            ["node_name_1", "node_name_4"],
            ["node_name_2", "node_name_3", "node_name_5", "node_name_6"],
        ],
    ],
    ids=["all empty", "one of each", "amultiple reverted ops", "multiple quantizer groups"],
)
def test_quantization_accuracy_restorer_report_properties(
    removed_groups, expected_removed_quantizers, expected_reverted_operations
):
    report = QuantizationAccuracyRestorerReport()
    report.removed_groups = removed_groups
    assert [n.node_name for n in report.removed_quantizers] == expected_removed_quantizers
    assert [n.node_name for n in report.reverted_operations] == expected_reverted_operations


@dataclass
class StructForPrintTest:
    max_num_iterations: int
    removed_all: bool
    reached_required_drop: bool
    num_iterations: int
    msg: str
    case: str
    num_quantized_operations: int


@pytest.mark.parametrize(
    "ts",
    [
        StructForPrintTest(
            max_num_iterations=5,
            removed_all=True,
            reached_required_drop=True,
            num_iterations=3,
            msg="The algorithm could not achieve the required accuracy drop.",
            case="removed all",
            num_quantized_operations=1,
        ),
        StructForPrintTest(
            max_num_iterations=5,
            removed_all=False,
            reached_required_drop=False,
            num_iterations=3,
            msg="The algorithm could not achieve the required accuracy drop.",
            case="not reached required drop",
            num_quantized_operations=1,
        ),
        StructForPrintTest(
            max_num_iterations=5,
            removed_all=False,
            reached_required_drop=True,
            num_iterations=4,
            msg="Maximum number of iteration was reached",
            case="max number of iterations reached",
            num_quantized_operations=1,
        ),
        StructForPrintTest(
            max_num_iterations=5,
            removed_all=False,
            reached_required_drop=True,
            num_iterations=3,
            msg="1 out of 1",
            case="regular case",
            num_quantized_operations=1,
        ),
    ],
)
def test_print_report_parameterized(ts: StructForPrintTest, nncf_caplog):
    setup_quantization_accuracy_restorer_report = QuantizationAccuracyRestorerReport()
    setup_quantization_accuracy_restorer_report.removed_groups = [
        type(
            "MockGroup",
            (),
            {
                "quantizers": [
                    NNCFNode(
                        {
                            NNCFNode.NODE_NAME_ATTR: "node_name_1",
                        }
                    )
                ],
                "operations": [
                    NNCFNode(
                        {
                            NNCFNode.NODE_NAME_ATTR: "node_name_2",
                        }
                    )
                ],
            },
        )
    ]
    setup_quantization_accuracy_restorer_report.removed_all = ts.removed_all
    setup_quantization_accuracy_restorer_report.reached_required_drop = ts.reached_required_drop
    setup_quantization_accuracy_restorer_report.num_iterations = ts.num_iterations
    setup_quantization_accuracy_restorer_report.num_quantized_operations = ts.num_quantized_operations
    QuantizationAccuracyRestorer._print_report(setup_quantization_accuracy_restorer_report, ts.max_num_iterations)

    with nncf_caplog.at_level(logging.INFO):
        assert ts.msg in nncf_caplog.text


@pytest.fixture
def model_and_quantized_model():
    model = Mock()
    initial_model_graph = get_mock_model_graph_with_mergeable_pattern()
    quantized_model_graph = get_mock_model_graph_with_mergeable_pattern()
    return model, initial_model_graph, quantized_model_graph


def test_collect_original_biases_and_weights_openvino(model_and_quantized_model, mocker):
    model, initial_model_graph, quantized_model_graph = model_and_quantized_model
    quantization_acc_restorer = QuantizationAccuracyRestorer()

    def isConv2dBias(node: NNCFNode, initial_model_graph):
        return node.node_id == 1

    def isConv2dWeight(node: NNCFNode):
        return node.node_id == 1

    # patch properties of AABackendForTests
    mocker.patch("tests.common.accuracy_control.backend.AABackendForTests.is_node_with_bias", side_effect=isConv2dBias)
    mocker.patch(
        "tests.common.accuracy_control.backend.AABackendForTests.is_node_with_weight", side_effect=isConv2dWeight
    )
    mocker.patch(
        "tests.common.accuracy_control.backend.AABackendForTests.get_bias_value",
        return_value=np.array([[1, 2, 3], [4, 5, 6]]),
    )
    mocker.patch(
        "tests.common.accuracy_control.backend.AABackendForTests.get_weight_value",
        return_value=np.array([[8, 9, 10], [11, 12, 13]]),
    )
    mocker.patch(
        "tests.common.accuracy_control.backend.AABackendForTests.get_weight_tensor_port_ids", return_value=[1, 2]
    )

    quantization_acc_restorer._collect_original_biases_and_weights(
        initial_model_graph,
        quantized_model_graph,
        model,
        AABackendForTests,
    )

    # original attributes stored for node with bias and weights
    conv_node = quantized_model_graph.get_node_by_id(1)
    assert (conv_node.attributes["original_bias"] == np.array([[1, 2, 3], [4, 5, 6]])).all()
    assert (conv_node.attributes["original_weight.1"] == np.array([[8, 9, 10], [11, 12, 13]])).all()
    assert (conv_node.attributes["original_weight.2"] == np.array([[8, 9, 10], [11, 12, 13]])).all()
    with pytest.raises(KeyError):
        assert conv_node.attributes["original_weight.3"] is None

    # original attributes not stored for node with no bias and weights
    non_conv_node = quantized_model_graph.get_node_by_id(2)
    with pytest.raises(KeyError):
        non_conv_node.attributes["original_bias"]
    with pytest.raises(KeyError):
        non_conv_node.attributes["original_weight"]


@dataclass
class StructForWorkerCalcTest:
    model_size: int
    preparation_time: int
    validation_time: int
    validation_dataset_size: int
    result: int
    id: str


@pytest.mark.parametrize(
    "ts",
    [
        StructForWorkerCalcTest(
            model_size=100,
            preparation_time=0.1,
            validation_time=1.0,
            validation_dataset_size=1000,
            id="preparation time < threshold",
            result=1,
        ),
        StructForWorkerCalcTest(
            model_size=100,
            preparation_time=1.0,
            validation_time=1.0,
            validation_dataset_size=10,
            id="preparation time == threshold",
            result=2,
        ),
        StructForWorkerCalcTest(
            model_size=100,
            preparation_time=3.0,
            validation_time=1.0,
            validation_dataset_size=10,
            id="preparation time > threshold",
            result=2,
        ),
        StructForWorkerCalcTest(
            model_size=10,
            preparation_time=500.0,
            validation_time=1.0,
            validation_dataset_size=2,
            id="limited by cpu count",
            result=5,
        ),
        StructForWorkerCalcTest(
            model_size=10000000,
            preparation_time=500.0,
            validation_time=1.0,
            validation_dataset_size=2,
            id="limited by memory",
            result=1,
        ),
    ],
)
def test_calculate_number_ranker_workers(ts: StructForWorkerCalcTest, mocker):
    mocker.patch("nncf.quantization.algorithms.accuracy_control.algorithm.get_available_cpu_count", return_value=10)
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.algorithm.get_available_memory_amount", return_value=10000
    )
    quantization_acc_restorer = QuantizationAccuracyRestorer()
    assert ts.result == quantization_acc_restorer._calculate_number_ranker_workers(
        ts.model_size,
        ts.preparation_time,
        ts.validation_time,
        ts.validation_dataset_size,
    )
