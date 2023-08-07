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

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend
from tests.post_training.test_templates.test_smooth_quant import TemplateTestSQAlgorithm
from tests.shared.command import Command


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
        onnx_path = Path(f"{tmp_dir}/model.onnx")
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        ov_path = Path(f"{tmp_dir}/model.xml")
        runner = Command(f"mo -m {onnx_path} -o {tmp_dir} -n model")
        runner.run()
        core = ov.Core()
        ov_model = core.read_model(ov_path)
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
        "node_metatype, inputs_attributes, port_id, reference_value",
        (
            (OVMatMulMetatype, {"transpose": False}, 0, -1),
            (OVMatMulMetatype, {"transpose": True}, 0, -2),
            (OVMatMulMetatype, {"transpose": False}, 1, -2),
            (OVMatMulMetatype, {"transpose": True}, 1, -1),
            (OVMatMulMetatype, {"transpose": False}, 2, RuntimeError),
            (OVConvolutionMetatype, {}, 0, 1),
        ),
    )
    def test_get_activation_channel_axis(self, node_metatype, inputs_attributes, port_id, reference_value):
        backend = self.get_backend()

        attributes = {
            NNCFGraph.METATYPE_ATTR: node_metatype,
            NNCFGraph.LAYER_ATTRIBUTES: OVLayerAttributes(constant_attributes={}, inputs_attributes=inputs_attributes),
        }
        node = NNCFNode(0, "test_node", attributes)

        try:
            # pylint: disable=protected-access
            activation_channel_axis = backend.get_activation_channel_axis(node, port_id)
        except RuntimeError as e:
            if isinstance(e, reference_value):
                pytest.xfail("Expected exception")

        assert activation_channel_axis == reference_value

    @pytest.mark.parametrize(
        "node_metatype, constant_attributes, port_id, reference_value",
        (
            (OVMatMulMetatype, {1: {"transpose": False}}, 1, -1),
            (OVMatMulMetatype, {1: {"transpose": True}}, 1, -2),
            (OVMatMulMetatype, {0: {"transpose": False}}, 0, -2),
            (OVMatMulMetatype, {0: {"transpose": True}}, 0, -1),
            (OVMatMulMetatype, {1: {"transpose": False}}, 2, RuntimeError),
            (OVConvolutionMetatype, {1: {}}, 1, 0),
        ),
    )
    def test_get_weight_reduction_axis(self, node_metatype, constant_attributes, port_id, reference_value):
        backend = self.get_backend()

        attributes = {
            NNCFGraph.METATYPE_ATTR: node_metatype,
            NNCFGraph.LAYER_ATTRIBUTES: OVLayerAttributes(constant_attributes=constant_attributes),
        }
        node = NNCFNode(0, "test_node", attributes)

        try:
            # pylint: disable=protected-access
            activation_channel_axis = backend.get_weight_reduction_axis(node, port_id)
        except RuntimeError as e:
            if isinstance(e, reference_value):
                pytest.xfail("Expected exception")

        assert activation_channel_axis == reference_value
