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

import pytest
from torch import nn

from nncf import NNCFConfig
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationMode
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.scopes import IgnoredScope
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from nncf.torch.quantization.quantize_model import _create_nncf_config
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.post_training.test_templates.test_ptq_params import TemplateTestPTQParams
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_depthwise_conv
from tests.torch.ptq.helpers import get_nncf_network
from tests.torch.ptq.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_no_weight_matmul_nncf_graph

# pylint: disable=protected-access


def get_hw_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=device)


def get_ignored_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_ignored_pattern_graph(backend=BackendType.TORCH, device=device)


class ToNNCFNetworkInterface:
    def get_nncf_network(self):
        return get_nncf_network(self)


class LinearTestModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(3, 3, 1)
        self.bn1 = create_bn(3)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = create_conv(3, 1, 1)
        self.bn2 = create_bn(1)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        return x


class OneDepthwiseConvModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self) -> None:
        super().__init__()
        self.depthwise_conv = create_depthwise_conv(3, 1, 1, 1)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        return self.depthwise_conv(x)


@pytest.mark.parametrize("target_device", TargetDevice)
def test_target_device(target_device):
    min_max_algo = MinMaxQuantization(target_device=target_device)
    min_max_algo._backend_entity = PTMinMaxAlgoBackend()
    assert min_max_algo._target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    def check_is_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MinAggregator in aggrs
        assert MaxAggregator in aggrs

    def check_is_mean_min_max_statistic_collector(self, tensor_collector: TensorCollector):
        aggrs = [aggr.__class__ for aggr in tensor_collector.aggregators.values()]
        assert len(aggrs) == 2
        assert MeanAggregator in aggrs
        assert aggrs[0].__class__ == aggrs[1].__class__

    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @property
    def metatypes_mapping(self):
        return {
            Conv2dTestMetatype: PTModuleConv2dMetatype,
            LinearTestMetatype: PTModuleLinearMetatype,
            SoftmaxTestMetatype: PTSoftmaxMetatype,
        }

    @pytest.fixture(scope="session")
    def test_params(self):
        linear_model = LinearTestModel().get_nncf_network()
        depthwise_model = OneDepthwiseConvModel().get_nncf_network()

        return {
            "test_range_estimator_per_tensor": {
                "model": linear_model,
                "nncf_graph": linear_model.nncf.get_graph(),
                "stat_points_num": 5,
            },
            "test_range_estimator_per_channel": {
                "model": depthwise_model,
                "nncf_graph": depthwise_model.nncf.get_graph(),
                "stat_points_num": 2,
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


@pytest.mark.parametrize(
    "params",
    (
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 1,
            "model_type": ModelType.TRANSFORMER,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, quantize_outputs=True, disable_bias_correction=True
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 2,
            "model_type": None,
            "ignored_scope": None,
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.ENABLE, quantize_outputs=False, disable_bias_correction=False
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 3,
            "model_type": None,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.FIRST_LAYER, quantize_outputs=True, disable_bias_correction=False
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 4,
            "model_type": None,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.FIRST_LAYER,
                quantize_outputs=True,
                disable_bias_correction=False,
                activations_quantization_params=QuantizationParameters(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                activations_range_estimator_params=RangeEstimatorParametersSet.MEAN_MINMAX,
                weights_quantization_params=QuantizationParameters(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                weights_range_estimator_params=RangeEstimatorParametersSet.MEAN_MINMAX,
            ),
        },
    ),
)
def test_create_nncf_config(params):
    config = _create_nncf_config(**params)

    assert config["compression"]["overflow_fix"] == params["advanced_parameters"].overflow_fix.value
    assert config["compression"]["quantize_outputs"] == params["advanced_parameters"].quantize_outputs

    assert config["compression"]["preset"] == params["preset"].value

    range_config = config["compression"]["initializer"]["range"]
    if isinstance(range_config, dict):
        assert range_config["num_init_samples"] == params["subset_size"]
        assert range_config["type"] == "mean_min_max"
    else:
        for rc in range_config:
            assert rc["num_init_samples"] == params["subset_size"]
            assert rc["type"] == "mean_min_max"

    num_bn_samples = config["compression"]["initializer"]["batchnorm_adaptation"]["num_bn_adaptation_samples"]
    if params["advanced_parameters"].disable_bias_correction is True or params["model_type"] == ModelType.TRANSFORMER:
        assert num_bn_samples == 0
    else:
        assert num_bn_samples == params["subset_size"]

    ref_scope = params["ignored_scope"].names if params["ignored_scope"] is not None else []
    if params["model_type"] == ModelType.TRANSFORMER:
        ref_scope = [
            "{re}.*Embeddings.*",
            "{re}.*__add___[0-1]",
            "{re}.*layer_norm_0",
            "{re}.*matmul_1",
            "{re}.*__truediv__*",
        ] + ref_scope
    assert config["compression"].get("ignored_scopes", []) == ref_scope

    # To validate NNCFConfig requared input_info
    config["input_info"] = {"sample_size": [1, 2, 224, 224]}
    NNCFConfig.validate(config)
