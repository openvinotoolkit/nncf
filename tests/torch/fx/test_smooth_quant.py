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

from typing import Any, Callable, Dict, Type

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from nncf import IgnoredScope
from nncf.parameters import ModelType
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.smooth_quant.torch_fx_backend import FXSmoothQuantAlgoBackend
from nncf.quantization.algorithms.smooth_quant.torch_fx_backend import FXSQMultiply
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import LinearMultiShapeModel
from tests.cross_fw.test_templates.helpers import ShareWeghtsConvAndShareLinearModel
from tests.cross_fw.test_templates.test_smooth_quant import TemplateTestSQAlgorithm
from tests.torch.fx.helpers import get_torch_fx_model_q_transformed

PT_LINEAR_MODEL_MM_MAP = {"Linear1": "linear_3", "Linear2": "linear_2", "Linear3": "linear", "Linear4": "linear_1"}

PT_CONV_MODEL_MM_MAP = {"Conv1": "conv2d"}


class TestTorchSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def backend_supports_shared_layers() -> bool:
        return False

    @staticmethod
    def fn_to_type(tensor) -> torch.Tensor:
        return torch.tensor(tensor)

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def get_node_name_map(self, model_cls) -> Dict[str, str]:
        if model_cls is LinearMultiShapeModel:
            return PT_LINEAR_MODEL_MM_MAP
        if model_cls is ConvTestModel:
            return PT_CONV_MODEL_MM_MAP
        if model_cls is ShareWeghtsConvAndShareLinearModel:
            return {}
        raise NotImplementedError

    @staticmethod
    def get_ignored_scope(model_cls: Any) -> IgnoredScope:
        if model_cls is LinearMultiShapeModel:
            # Ignore matmul nodes before the min/max nodes as
            # min/max operatons could  not be quantized
            # due to nncf propagation algo  restrictions.
            return IgnoredScope(names=["matmul_5", "matmul_6"])
        return IgnoredScope()

    @staticmethod
    def get_quantization_algorithm(ignored_scope: IgnoredScope):
        return PostTrainingQuantization(
            subset_size=1,
            model_type=ModelType.TRANSFORMER,
            ignored_scope=ignored_scope,
            advanced_parameters=AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE,
                smooth_quant_alphas=AdvancedSmoothQuantParameters(matmul=0.95, convolution=0.95),
                inplace_statistics=False,
                disable_bias_correction=True,
            ),
        )

    @staticmethod
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            return data_item[0]

        return transform_fn

    @staticmethod
    def get_backend() -> Type[FXSmoothQuantAlgoBackend]:
        return FXSmoothQuantAlgoBackend()

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> ov.Model:
        return get_torch_fx_model_q_transformed(model, torch.ones(model.INPUT_SIZE))

    @staticmethod
    def check_scales(model: torch.nn.Module, reference_values: Dict[str, np.ndarray], model_cls) -> None:
        names_map = PT_LINEAR_MODEL_MM_MAP if model_cls is LinearMultiShapeModel else PT_CONV_MODEL_MM_MAP
        ops_list = {node.name: node for node in model.graph.nodes}
        for ref_names, ref_value in reference_values.items():
            if not all(name.startswith("Linear") or name.startswith("Conv") for name in ref_names):
                # Pytorch SQ algorithm supports only linear and conv modules by far,
                # so other multiplies are skipped
                continue
            sq_modules = []
            for ref_name in ref_names:
                node = ops_list[names_map[ref_name]]
                while node.op != "call_module":
                    node = node.all_input_nodes[0]

                sq_modules.append(getattr(model, node.target))
            # Check unified group acutally shares one constant
            assert all(node is sq_modules[0] for node in sq_modules[1:])
            sq_node = sq_modules[0]
            assert isinstance(sq_node, FXSQMultiply)

            value = sq_node._scale_value
            ref_value = torch.tensor(ref_value)
            assert value.shape == ref_value.shape
            assert torch.all(torch.isclose(value, ref_value, rtol=1e-4))

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, port_id, reference_value",
        (
            (PTLinearMetatype, None, 0, -1),
            (PTConv2dMetatype, None, 0, 1),
        ),
    )
    def test_get_activation_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        return super().test_get_activation_channel_axis(node_metatype, layer_attributes, port_id, reference_value)

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, reference_value",
        (
            (PTLinearMetatype, None, 1),
            (PTConv2dMetatype, None, 1),
        ),
    )
    def test_get_weight_channel_axis(self, node_metatype, layer_attributes, reference_value):
        return super().test_get_weight_channel_axis(node_metatype, layer_attributes, reference_value)

    @staticmethod
    def get_matmul_metatype():
        return PTLinearMetatype
