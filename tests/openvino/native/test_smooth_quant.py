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

from pathlib import Path
from typing import Callable, Dict

import numpy as np
import openvino.runtime as ov
import pytest
import torch
from openvino.tools.mo import convert_model

from nncf.common.graph.layer_attributes import ConvLayoutElem
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend
from tests.post_training.test_templates.test_smooth_quant import TemplateTestSQAlgorithm


class TestOVSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @staticmethod
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input.1": tensor}

        return transform_fn

    @staticmethod
    def get_backend() -> OVSmoothQuantAlgoBackend:
        return OVSmoothQuantAlgoBackend()

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> ov.Model:
        # TODO(AlexanderDokuchaev): remove onnx export after fix 119625
        onnx_path = Path(f"{tmp_dir}/model.onnx")
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        ov_model = convert_model(onnx_path, input_shape=model.INPUT_SIZE, compress_to_fp16=False)
        return ov_model

    @staticmethod
    def check_scales(model: ov.Model, reference_values: Dict[str, np.ndarray]) -> None:
        ops_list = {op.get_friendly_name(): op for op in model.get_ops()}
        for ref_name, ref_value in reference_values.items():
            node = ops_list[ref_name]
            const_node = node.input(1).get_source_output().get_node()

            assert const_node.get_type_name() == "Constant"

            value = const_node.data
            ref_value = np.array(ref_value)
            assert value.shape == ref_value.shape
            assert np.all(np.isclose(value, ref_value, atol=0.0001)), f"{value} != {ref_value}"

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, port_id, reference_value",
        (
            (OVMatMulMetatype, OVLayerAttributes({}, inputs_attributes={"transpose": False}), 0, -1),
            (OVMatMulMetatype, OVLayerAttributes({}, inputs_attributes={"transpose": True}), 0, -2),
            (OVMatMulMetatype, OVLayerAttributes({}, inputs_attributes={"transpose": False}), 1, -2),
            (OVMatMulMetatype, OVLayerAttributes({}, inputs_attributes={"transpose": True}), 1, -1),
            (OVMatMulMetatype, OVLayerAttributes({}, inputs_attributes={"transpose": False}), 2, RuntimeError),
            (OVConvolutionMetatype, OVLayerAttributes({}, inputs_attributes={}), 0, 1),
        ),
    )
    def test_get_activation_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        return super().test_get_activation_channel_axis(node_metatype, layer_attributes, port_id, reference_value)

    @pytest.mark.parametrize(
        "node_metatype,layer_attributes,reference_value",
        (
            (
                OVMatMulMetatype,
                OVLayerAttributes(
                    {},
                    LinearLayerAttributes(
                        weight_requires_grad=False,
                        in_features=5,
                        out_features=10,
                        with_bias=False,
                        weights_layout=[ConvLayoutElem.C_OUT, ConvLayoutElem.C_IN],
                    ),
                ),
                1,
            ),
            (
                OVMatMulMetatype,
                OVLayerAttributes(
                    {},
                    LinearLayerAttributes(
                        weight_requires_grad=False,
                        in_features=5,
                        out_features=None,
                        with_bias=False,
                        weights_layout=[ConvLayoutElem.C_IN],
                    ),
                ),
                0,
            ),
            (
                OVConvolutionMetatype,
                OVLayerAttributes(
                    {},
                    ConvolutionLayerAttributes(
                        weight_requires_grad=False,
                        in_channels=5,
                        out_channels=10,
                        kernel_size=(5, 5),
                        stride=(1, 1),
                        dilations=(1, 1),
                        groups=1,
                        transpose=False,
                        padding_values=[1, 1, 1, 1],
                        with_bias=False,
                        weights_layout=[
                            ConvLayoutElem.SPATIAL,
                            ConvLayoutElem.SPATIAL,
                            ConvLayoutElem.C_IN,
                            ConvLayoutElem.C_OUT,
                        ],
                    ),
                ),
                2,
            ),
            (
                OVMatMulMetatype,
                OVLayerAttributes(
                    {},
                    None,
                ),
                1,
            ),
        ),
    )
    def test_get_weight_channel_axis(self, node_metatype, layer_attributes, reference_value):
        return super().test_get_weight_channel_axis(node_metatype, layer_attributes, reference_value)

    @staticmethod
    def get_matmul_metatype():
        return OVMatMulMetatype
