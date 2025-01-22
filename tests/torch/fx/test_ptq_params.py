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

import pytest
import torch

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.utils.backend import BackendType
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.scopes import IgnoredScope
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import PTCatMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from tests.common.quantization.metatypes import CatTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.cross_fw.test_templates.test_ptq_params import TemplateTestPTQParams
from tests.torch.fx.helpers import get_single_conv_nncf_graph
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed
from tests.torch.ptq.helpers import get_single_no_weight_matmul_nncf_graph
from tests.torch.test_models.synthetic import LinearPTQParamsTestModel


def get_hw_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH_FX, device=device)


def get_ignored_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_ignored_pattern_graph(backend=BackendType.TORCH_FX, device=device)


@pytest.mark.parametrize("target_device", TargetDevice)
def test_target_device(target_device):
    min_max_algo = MinMaxQuantization(target_device=target_device)
    min_max_algo._backend_entity = FXMinMaxAlgoBackend()
    assert min_max_algo._target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return FXMinMaxAlgoBackend()

    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def check_unified_scale_layout(self, layout, unified_scale_group):
        assert len(layout.transformations) == len(unified_scale_group)
        for t, ref_tp in zip(layout.transformations, unified_scale_group):
            assert isinstance(t, FXApplyTransformationCommand)
            assert t.transformation_fn.__closure__[1].cell_contents[0] == ref_tp
            assert t.type == TransformationType.INSERT
            assert t.transformation_fn.__closure__[0].cell_contents.zero_point == 0
            assert torch.allclose(
                t.transformation_fn.__closure__[0].cell_contents.scale, torch.tensor(0.031496062874794006)
            )

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    def get_backend_tensor(self, value):
        return torch.tensor(value)

    @property
    def metatypes_mapping(self):
        return {
            Conv2dTestMetatype: PTConv2dMetatype,
            LinearTestMetatype: PTLinearMetatype,
            SoftmaxTestMetatype: PTSoftmaxMetatype,
            CatTestMetatype: PTCatMetatype,
        }

    @property
    def nncf_graph_cls(self):
        return PTNNCFGraph

    @pytest.fixture(scope="session")
    def test_params(self):
        linear_model = LinearPTQParamsTestModel()
        linear_model = get_torch_fx_model_q_transformed(linear_model, torch.ones((1, 3, 32, 32)))

        return {
            "test_range_estimator_per_tensor": {
                "model": linear_model,
                "nncf_graph": GraphConverter.create_nncf_graph(linear_model),
                "stat_points_num": 5,
            },
            "test_quantize_outputs": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_ignored_scopes": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_model_type_pass": {
                "nncf_graph": get_single_no_weight_matmul_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_validate_scope": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "ignored_patterns": get_ignored_patterns(),
            },
        }

    @pytest.fixture(params=[(IgnoredScope([]), 1, 1), (IgnoredScope(["/Conv_1_0"]), 0, 0)])
    def ignored_scopes_data(self, request):
        return request.param
