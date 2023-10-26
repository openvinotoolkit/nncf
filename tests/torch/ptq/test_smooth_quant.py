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

from typing import Callable, Dict, Type

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.quantization.algorithms.smooth_quant.torch_backend import PTSmoothQuantAlgoBackend
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.transformations.command_creation import SQMultiply
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.model_creation import wrap_model
from nncf.torch.nncf_network import ExtraCompressionModuleType
from tests.post_training.test_templates.test_smooth_quant import TemplateTestSQAlgorithm

PT_LINEAR_MODEL_SQ_MAP = {
    ("Linear1",): "LinearMultiShapeModel/split_0_1_0/nncf_smooth_quant",
    ("Linear2",): "LinearMultiShapeModel/split_0_0_0/nncf_smooth_quant",
    ("Linear3", "Linear4"): "LinearMultiShapeModel/add_0_0_0/nncf_smooth_quant",
}

PT_LINEAR_MODEL_MM_MAP = {
    "Linear1": "LinearMultiShapeModel/NNCFLinear[linear_2]/linear_0",
    "Linear2": "LinearMultiShapeModel/NNCFLinear[linear_1]/linear_0",
    "Linear3": "LinearMultiShapeModel/NNCFLinear[linear_3]/linear_0",
    "Linear4": "LinearMultiShapeModel/NNCFLinear[linear_4]/linear_0",
}


class TestTorchSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def fn_to_type(tensor) -> torch.Tensor:
        return torch.tensor(tensor)

    @pytest.fixture(params=[False], ids=["out_of_palce"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def get_node_name_map(self) -> Dict[str, str]:
        return PT_LINEAR_MODEL_MM_MAP

    @staticmethod
    def get_target_node_name(command: TransformationCommand):
        if isinstance(command, PTSharedFnInsertionCommand):
            return command.target_points[0].target_node_name
        return command.target_point.target_node_name

    @staticmethod
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            return data_item[0]

        return transform_fn

    @staticmethod
    def get_backend() -> Type[PTSmoothQuantAlgoBackend]:
        return PTSmoothQuantAlgoBackend()

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> ov.Model:
        return wrap_model(model.eval(), torch.rand(model.INPUT_SIZE))

    @staticmethod
    def check_scales(model: torch.nn.Module, reference_values: Dict[str, np.ndarray]) -> None:
        modules = model.nncf.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_OP)
        for ref_names, ref_value in reference_values.items():
            if not all(name.startswith("Linear") for name in ref_names):
                # Pytorch SQ algorithm supports only linear modules by far,
                # so other multiplies are skipped
                continue
            sq_node = modules[PT_LINEAR_MODEL_SQ_MAP[ref_names]]

            assert isinstance(sq_node, SQMultiply)

            value = sq_node._scale_value
            ref_value = torch.tensor(ref_value)
            assert value.shape == ref_value.shape
            assert torch.all(torch.isclose(value, ref_value, rtol=1e-4))

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, port_id, reference_value",
        (
            (PTModuleLinearMetatype, None, 0, -1),
            (PTModuleConv2dMetatype, None, 0, 1),
        ),
    )
    def test_get_activation_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        return super().test_get_activation_channel_axis(node_metatype, layer_attributes, port_id, reference_value)

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, reference_value",
        (
            (PTModuleLinearMetatype, None, 1),
            (PTModuleConv2dMetatype, None, 1),
        ),
    )
    def test_get_weight_channel_axis(self, node_metatype, layer_attributes, reference_value):
        return super().test_get_weight_channel_axis(node_metatype, layer_attributes, reference_value)

    @staticmethod
    def get_matmul_metatype():
        return PTModuleLinearMetatype
