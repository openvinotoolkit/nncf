# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import logging
import pytest
from nncf.common.graph.graph import NNCFNode
from nncf.common.utils.backend import BackendType
from nncf.parameters import DropType
from nncf.quantization.algorithms.accuracy_control.algorithm import (
    QuantizationAccuracyRestorer,
    QuantizationAccuracyRestorerReport,
    _create_message,
    get_algo_backend,
)
from nncf.quantization.algorithms.accuracy_control.openvino_backend import (
    OVAccuracyControlAlgoBackend,
)
from nncf.errors import UnsupportedBackendError


def test_get_algo_backend():
    result = get_algo_backend(BackendType.OPENVINO)
    assert isinstance(result, OVAccuracyControlAlgoBackend)


def test_get_algo_backend_error():
    with pytest.raises(UnsupportedBackendError):
        get_algo_backend(BackendType.TORCH)


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
    assert isinstance(result, str)
    assert result == "\n".join(["\t" + node.node_name for node in nodes])


@pytest.fixture
def setup_quantization_accuracy_restorer_report():
    report = QuantizationAccuracyRestorerReport()
    node1 = NNCFNode(
        {
            NNCFNode.NODE_NAME_ATTR: "node_name_1",
        }
    )
    node2 = NNCFNode(
        {
            NNCFNode.NODE_NAME_ATTR: "node_name_2",
        }
    )
    report.removed_groups = [
        type("MockGroup", (), {"quantizers": [node1], "operations": [node2]})
    ]
    return report


def test_removed_quantizers(setup_quantization_accuracy_restorer_report):
    assert (
        setup_quantization_accuracy_restorer_report.removed_quantizers[0].node_name
        == "node_name_1"
    )


def test_reverted_operations(setup_quantization_accuracy_restorer_report):
    assert (
        setup_quantization_accuracy_restorer_report.reverted_operations[0].node_name
        == "node_name_2"
    )


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
def test_print_report_parameterized(
    ts: StructForPrintTest, setup_quantization_accuracy_restorer_report, nncf_caplog
):
    setup_quantization_accuracy_restorer_report.removed_all = ts.removed_all
    setup_quantization_accuracy_restorer_report.reached_required_drop = (
        ts.reached_required_drop
    )
    setup_quantization_accuracy_restorer_report.num_iterations = ts.num_iterations
    setup_quantization_accuracy_restorer_report.num_quantized_operations = (
        ts.num_quantized_operations
    )
    QuantizationAccuracyRestorer._print_report(
        setup_quantization_accuracy_restorer_report, ts.max_num_iterations
    )

    with nncf_caplog.at_level(logging.INFO):
        assert ts.msg in nncf_caplog.text
