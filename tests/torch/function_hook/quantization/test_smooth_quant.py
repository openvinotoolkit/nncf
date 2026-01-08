# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import numpy as np
import openvino as ov
import pytest
import torch

from nncf.quantization.algorithms.smooth_quant.torch_backend import PTSmoothQuantAlgoBackend
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import LinearMultiShapeModel
from tests.cross_fw.test_templates.helpers import ShareWeghtsConvAndShareLinearModel
from tests.cross_fw.test_templates.test_smooth_quant import TemplateTestSQAlgorithm

PT_LINEAR_MODEL_SQ_MAP = {
    ("Linear1",): "__nncf_hooks.pre_hooks.linear_2/linear/0__0.0._scale_value",
    ("Linear2",): "__nncf_hooks.pre_hooks.linear_1/linear/0__0.0._scale_value",
    ("Linear3", "Linear4"): "__nncf_hooks.pre_hooks.linear_3/linear/0__0.0._scale_value",
}
PT_LINEAR_MODEL_MM_MAP = {
    "Linear1": "linear_2/linear/0",
    "Linear2": "linear_1/linear/0",
    "Linear3": "linear_3/linear/0",
    "Linear4": "linear_4/linear/0",
}

PT_CONV_MODEL_SQ_MAP = {("Conv1",): "__nncf_hooks.pre_hooks.conv/conv2d/0__0.0._scale_value"}

PT_CONV_MODEL_MM_MAP = {"Conv1": "conv/conv2d/0"}


class TestTorchSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def backend_supports_shared_layers() -> bool:
        return True

    @staticmethod
    def fn_to_type(tensor) -> torch.Tensor:
        return torch.tensor(tensor)

    @pytest.fixture(params=[False], ids=["out_of_place"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def get_node_name_map(self, model_cls) -> dict[str, str]:
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
    def get_backend() -> type[PTSmoothQuantAlgoBackend]:
        return PTSmoothQuantAlgoBackend()

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> ov.Model:
        return GraphModelWrapper(wrap_model(model.eval()), torch.rand(model.INPUT_SIZE))

    @staticmethod
    def check_scales(model: GraphModelWrapper, reference_values: dict[str, np.ndarray], model_cls) -> None:
        names_map = PT_LINEAR_MODEL_SQ_MAP if model_cls is LinearMultiShapeModel else PT_CONV_MODEL_SQ_MAP
        data_map = {n: x for n, x in model.model.named_parameters()}

        for ref_names, ref_value in reference_values.items():
            if not all(name.startswith("Linear") or name.startswith("Conv") for name in ref_names):
                # Pytorch SQ algorithm supports only linear and conv modules by far,
                # so other multiplies are skipped
                continue

            ref_value = torch.tensor(ref_value)
            actual_data = data_map[names_map[ref_names]].data
            assert actual_data.shape == ref_value.shape
            assert torch.all(torch.isclose(actual_data, ref_value, rtol=1e-4)), (
                f"{ref_names}: {actual_data=} != {ref_value=}"
            )

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
        return [PTLinearMetatype]
