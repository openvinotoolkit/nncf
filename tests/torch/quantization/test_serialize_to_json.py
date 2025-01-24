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

import json

import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizerConfig
from nncf.torch.dynamic_graph.context import Scope
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.cross_fw.shared.serialization import check_serialization

DUMMY_STR = "dummy"


def single_config_quantizer_setup_cmp(self, other):
    return (
        all(map(lambda x: x[0] == x[1], zip(self.quantization_points.values(), other.quantization_points.values())))
        and self.unified_scale_groups == other.unified_scale_groups
        and self.shared_input_operation_set_groups == other.shared_input_operation_set_groups
    )


GROUND_TRUTH_STATE = {
    "quantization_points": {
        0: {
            "directly_quantized_operator_node_names": ["MyConv/1[2]/3[4]/5"],
            "qconfig": {"mode": "symmetric", "num_bits": 8, "per_channel": False, "signedness_to_force": None},
            "qip": {"target_node_name": "dummy"},
            "qip_class": "WeightQuantizationInsertionPoint",
        },
        1: {
            "directly_quantized_operator_node_names": ["MyConv/1[2]/3[4]/5"],
            "qconfig": {"mode": "symmetric", "num_bits": 8, "per_channel": False, "signedness_to_force": None},
            "qip": {"input_port_id": 0, "target_node_name": "dummy"},
            "qip_class": "ActivationQuantizationInsertionPoint",
        },
    },
    "shared_input_operation_set_groups": {2: [0, 1]},
    "unified_scale_groups": {2: [0, 1]},
}


def test_quantizer_setup_serialization():
    target_type_1 = TargetType.OPERATOR_POST_HOOK
    check_serialization(target_type_1)

    target_type_2 = TargetType.POST_LAYER_OPERATION
    check_serialization(target_type_2)

    scope = Scope.from_str("MyConv/1[2]/3[4]/5")
    assert scope == Scope.from_str(str(scope))

    pttp_1 = PTTargetPoint(target_type_1, target_node_name=str(scope), input_port_id=7)
    check_serialization(pttp_1)

    wqip = WeightQuantizationInsertionPoint(target_node_name=DUMMY_STR)
    check_serialization(wqip)

    aqip = ActivationQuantizationInsertionPoint(target_node_name=DUMMY_STR, input_port_id=0)
    check_serialization(aqip)

    qc = QuantizerConfig()
    check_serialization(qc)

    scqp_1 = SingleConfigQuantizationPoint(wqip, qc, directly_quantized_operator_node_names=[str(scope)])
    check_serialization(scqp_1)

    scqp_2 = SingleConfigQuantizationPoint(aqip, qc, directly_quantized_operator_node_names=[str(scope)])
    check_serialization(scqp_2)

    scqs = SingleConfigQuantizerSetup()
    scqs.quantization_points = {0: scqp_1, 1: scqp_2}
    scqs.unified_scale_groups = {2: {0, 1}}
    scqs.shared_input_operation_set_groups = {2: {0, 1}}

    check_serialization(scqs, comparator=single_config_quantizer_setup_cmp)
    assert scqs.get_state() == GROUND_TRUTH_STATE


def test_precision_float():
    f1 = [1e-3, 2e-5, 1e-35, 2e40, 1.12341e-32, 0.2341123412345, 0.542e-63]
    f1_str = json.dumps(f1)
    f1_bytes = f1_str.encode("utf-8")
    f1_t = torch.ByteTensor(list(f1_bytes))
    f2_bytes = bytes(f1_t)
    f2_str = f2_bytes.decode("utf-8")
    f2 = json.loads(f2_str)
    assert f1 == f2
