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

import numpy as np
import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.common.utils.backend import BackendType
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConcatMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSoftmaxMetatype
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from nncf.scopes import IgnoredScope
from tests.common.quantization.metatypes import CatTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.cross_fw.test_templates.models import NNCFGraphToTestMatMul
from tests.cross_fw.test_templates.test_ptq_params import TemplateTestPTQParams
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.openvino.native.models import LinearModel


def get_hw_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_hw_pattern_graph(backend=BackendType.OPENVINO, device=device)


def get_ignored_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_ignored_pattern_graph(backend=BackendType.OPENVINO, device=device)


@pytest.mark.parametrize("target_device", [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU])
def test_target_device(target_device):
    min_max_algo = MinMaxQuantization(target_device=target_device)
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()
    assert min_max_algo._target_device.value == HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value]


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return OVMinMaxAlgoBackend()

    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 3
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def check_unified_scale_layout(self, layout, unified_scale_group):
        assert len(layout.transformations) == len(unified_scale_group)
        for t, ref_tp in zip(layout.transformations, unified_scale_group):
            assert isinstance(t, OVQuantizerInsertionCommand)
            assert t.target_point == ref_tp
            assert t.type == TransformationType.INSERT
            assert np.isclose(t.quantizer_parameters.input_low.data, -4.031496)
            assert np.isclose(t.quantizer_parameters.input_high.data, 4)

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    def get_backend_tensor(self, value):
        return np.array(value)

    @property
    def metatypes_mapping(self):
        return {
            Conv2dTestMetatype: OVConvolutionMetatype,
            LinearTestMetatype: OVMatMulMetatype,
            SoftmaxTestMetatype: OVSoftmaxMetatype,
            CatTestMetatype: OVConcatMetatype,
        }

    @property
    def nncf_graph_cls(self):
        return NNCFGraph

    @pytest.fixture(scope="session")
    def test_params(self):
        linear_model = LinearModel().ov_model
        linear_model_graph = GraphConverter.create_nncf_graph(linear_model)
        depthwise_model = DepthwiseConv4DModel().ov_model
        depthwise_model_graph = GraphConverter.create_nncf_graph(depthwise_model)

        return {
            "test_range_estimator_per_tensor": {
                "model": linear_model,
                "nncf_graph": linear_model_graph,
                "stat_points_num": 2,
            },
            "test_range_estimator_per_channel": {
                "model": depthwise_model,
                "nncf_graph": depthwise_model_graph,
                "stat_points_num": 2,
            },
            "test_quantize_outputs": {
                "nncf_graph": linear_model_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_ignored_scopes": {
                "nncf_graph": linear_model_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_model_type_pass": {
                "nncf_graph": NNCFGraphToTestMatMul(OVMatMulMetatype).nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_validate_scope": {
                "nncf_graph": linear_model_graph,
                "ignored_patterns": get_ignored_patterns(),
            },
        }

    @pytest.fixture(
        params=[
            (IgnoredScope(), 1, 1),
            (IgnoredScope(["MatMul"]), 0, 0),
            (IgnoredScope(["Add"]), 1, 1),
            (IgnoredScope(["MatMul", "Add"]), 0, 0),
        ]
    )
    def ignored_scopes_data(self, request):
        return request.param
