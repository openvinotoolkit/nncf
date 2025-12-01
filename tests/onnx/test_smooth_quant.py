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

from typing import Callable

import numpy as np
import onnx
import pytest
import torch

from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXMatMulMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.quantization.algorithms.smooth_quant.onnx_backend import ONNXSmoothQuantAlgoBackend
from tests.cross_fw.test_templates.helpers import ConvTestModel
from tests.cross_fw.test_templates.helpers import LinearMultiShapeModel
from tests.cross_fw.test_templates.helpers import ShareWeghtsConvAndShareLinearModel
from tests.cross_fw.test_templates.test_smooth_quant import TemplateTestSQAlgorithm

ONNX_LINEAR_MODEL_MM_OP_MAP = {
    "MatMul1": "/MatMul",
    "MatMul2": "/MatMul_1",
    "MatMul3": "/MatMul_2",
    "MatMul4": "/MatMul_4",
    "MatMul5": "/MatMul_3",
    "MatMul6": "/MatMul_5",
    "MatMul7": "/MatMul_6",
    "MatMul8": "/MatMul_7",
    "Linear1": "/linear_2/MatMul",
    "Linear2": "/linear_1/MatMul",
    "Linear3": "/linear_3/MatMul",
    "Linear4": "/linear_4/MatMul",
}


ONNX_LINEAR_MODEL_SQ_OP_MAP = {
    "MatMul1": "/Reshape_0_0/nncf_smooth_quant",
    "MatMul2": "/Reshape_0_0/nncf_smooth_quant",
    "MatMul3": "/Reshape_1_0_0/nncf_smooth_quant",
    "MatMul4": "/Reshape_1_0_1/nncf_smooth_quant",
    "MatMul5": "/Reshape_2_0_0/nncf_smooth_quant",
    "MatMul6": "/ReduceMax_0_0/nncf_smooth_quant",
    "MatMul7": "/Reshape_3_0_0/nncf_smooth_quant",
    "MatMul8": "/Reshape_4_0_0/nncf_smooth_quant",
    "Linear1": "/Split_1_0/nncf_smooth_quant",
    "Linear2": "/Split_0_0/nncf_smooth_quant",
    "Linear3": "/Add_0_0/nncf_smooth_quant",
    "Linear4": "/Add_0_0/nncf_smooth_quant",
}


ONNX_CONV_MODEL_MM_OP_MAP = {
    "Conv1": "/conv/Conv",
}
ONNX_CONV_MODEL_SQ_OP_MAP = {
    "Conv1": "nncf_model_input_0_0_0/nncf_smooth_quant",
}


class TestONNXSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def backend_supports_shared_layers() -> bool:
        return False

    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @pytest.fixture(params=[False], ids=["out_of_place"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def get_node_name_map(self, model_cls) -> dict[str, str]:
        if model_cls is LinearMultiShapeModel:
            return ONNX_LINEAR_MODEL_MM_OP_MAP
        if model_cls is ConvTestModel:
            return ONNX_CONV_MODEL_MM_OP_MAP
        if model_cls is ShareWeghtsConvAndShareLinearModel:
            return {}
        raise NotImplementedError

    @staticmethod
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input": tensor}

        return transform_fn

    @staticmethod
    def backend_specific_model(model: torch.nn.Module, tmp_dir: str) -> onnx.ModelProto:
        dummy_input = torch.rand(model.INPUT_SIZE)
        model_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model.cpu(), dummy_input, model_path, input_names=["input"], opset_version=13, dynamo=False)
        onnx_model = onnx.load(model_path)
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        return onnx_model

    @staticmethod
    def check_scales(model: onnx.ModelProto, reference_values: dict[str, np.ndarray], model_cls) -> None:
        names_map = ONNX_LINEAR_MODEL_SQ_OP_MAP if model_cls is LinearMultiShapeModel else ONNX_CONV_MODEL_SQ_OP_MAP

        for ref_names, ref_value in reference_values.items():
            for name in ref_names:
                initializer_name = f"{names_map[name]}_scale"
                value = get_tensor_value(model, initializer_name)
                ref_value = np.array(ref_value)

                assert value.shape == ref_value.shape
                assert np.all(np.isclose(value, ref_value, atol=0.0001)), f"{value} != {ref_value}"

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, reference_value",
        (
            (ONNXConvolutionMetatype, ONNXLayerAttributes({1: {"shape": [2, 1, 2, 2]}}), 1),
            (ONNXMatMulMetatype, ONNXLayerAttributes({1: {"shape": [5, 6]}}), -2),
            (ONNXMatMulMetatype, ONNXLayerAttributes({0: {"shape": [5, 6]}}), -1),
            (ONNXMatMulMetatype, ONNXLayerAttributes({1: {"shape": [1, 8, 3]}}), -2),
            (ONNXMatMulMetatype, ONNXLayerAttributes({0: {"shape": [1, 8, 3]}}), -1),
        ),
    )
    def test_get_weight_channel_axis(self, node_metatype, layer_attributes, reference_value):
        return super().test_get_weight_channel_axis(node_metatype, layer_attributes, reference_value)

    @pytest.mark.parametrize(
        "node_metatype, layer_attributes, port_id, reference_value",
        (
            (ONNXConvolutionMetatype, ONNXLayerAttributes({1: {"shape": [2, 1, 2, 2]}}), 0, 1),
            (ONNXMatMulMetatype, ONNXLayerAttributes({1: {"shape": [5, 6]}}), 0, -1),
            (ONNXMatMulMetatype, ONNXLayerAttributes({0: {"shape": [5, 6]}}), 1, -2),
            (ONNXMatMulMetatype, ONNXLayerAttributes({1: {"shape": [1, 8, 3]}}), 0, -1),
            (ONNXMatMulMetatype, ONNXLayerAttributes({0: {"shape": [1, 8, 3]}}), 1, -2),
        ),
    )
    def test_get_activation_channel_axis(self, node_metatype, layer_attributes, port_id, reference_value):
        return super().test_get_activation_channel_axis(node_metatype, layer_attributes, port_id, reference_value)

    @staticmethod
    def get_backend() -> ONNXSmoothQuantAlgoBackend:
        return ONNXSmoothQuantAlgoBackend()

    @staticmethod
    def get_matmul_metatype():
        return MATMUL_METATYPES
