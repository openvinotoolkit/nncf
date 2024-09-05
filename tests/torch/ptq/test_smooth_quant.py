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

from nncf.quantization.algorithms.smooth_quant.torch_backend import PTSmoothQuantAlgoBackend
from nncf.quantization.algorithms.smooth_quant.torch_backend import SQMultiply
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.model_creation import wrap_model
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import LinearMultiShapeModel
from tests.cross_fw.test_templates.helpers import ShareWeghtsConvAndShareLinearModel
from tests.cross_fw.test_templates.test_smooth_quant import TemplateTestSQAlgorithm

PT_LINEAR_MODEL_SQ_MAP = {
    ("Linear1",): "LinearMultiShapeModel/split_0_1_0/nncf_smooth_quant",
    ("Linear2",): "LinearMultiShapeModel/split_0_0_0/nncf_smooth_quant",
    ("Linear3", "Linear4"): "LinearMultiShapeModel/add_0_0_0/nncf_smooth_quant",
}

PT_LINEAR_MODEL_MM_MAP = {
    "Linear1": "LinearMultiShapeModel/Linear[linear_2]/linear_0",
    "Linear2": "LinearMultiShapeModel/Linear[linear_1]/linear_0",
    "Linear3": "LinearMultiShapeModel/Linear[linear_3]/linear_0",
    "Linear4": "LinearMultiShapeModel/Linear[linear_4]/linear_0",
}

PT_CONV_MODEL_SQ_MAP = {("Conv1",): "/nncf_model_input_0_0_0/nncf_smooth_quant"}

PT_CONV_MODEL_MM_MAP = {"Conv1": "ConvTestModel/Conv2d[conv]/conv2d_0"}


class TestTorchSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def backend_supports_shared_layers() -> bool:
        return True

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
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            return data_item[0]

        return transform_fn

    @staticmethod
    def get_backend() -> Type[PTSmoothQuantAlgoBackend]:
        return PTSmoothQuantAlgoBackend()

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> ov.Model:
        return wrap_model(model.eval(), torch.rand(model.INPUT_SIZE), trace_parameters=True)

    @staticmethod
    def check_scales(model: torch.nn.Module, reference_values: Dict[str, np.ndarray], model_cls) -> None:
        names_map = PT_LINEAR_MODEL_SQ_MAP if model_cls is LinearMultiShapeModel else PT_CONV_MODEL_SQ_MAP
        modules = model.nncf.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_OP)
        for ref_names, ref_value in reference_values.items():
            if not all(name.startswith("Linear") or name.startswith("Conv") for name in ref_names):
                # Pytorch SQ algorithm supports only linear and conv modules by far,
                # so other multiplies are skipped
                continue
            sq_node = modules[names_map[ref_names]]

            assert isinstance(sq_node, SQMultiply)

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
