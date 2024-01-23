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
import openvino as ov
import pytest
import torch

from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend
from tests.post_training.test_templates.helpers import ConvTestModel
from tests.post_training.test_templates.helpers import LinearMultiShapeModel
from tests.post_training.test_templates.helpers import ShareWeghtsConvAndShareLinearModel
from tests.post_training.test_templates.test_smooth_quant import TemplateTestSQAlgorithm

OV_LINEAR_MODEL_MM_OP_MAP = {
    "MatMul1": "/MatMul",
    "MatMul2": "/MatMul_1",
    "MatMul3": "/MatMul_2",
    "MatMul4": "/MatMul_4",
    "MatMul5": "32",
    "MatMul6": "37",
    "MatMul7": "54",
    "MatMul8": "68",
    "Linear1": "/linear_2/MatMul",
    "Linear2": "/linear_1/MatMul",
    "Linear3": "/linear_3/MatMul",
    "Linear4": "/linear_4/MatMul",
}

OV_LINEAR_MODEL_SQ_OP_MAP = {
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

OV_CONV_MODEL_MM_OP_MAP = {
    "Conv1": "/conv/Conv/WithoutBiases",
}

OV_CONV_MODEL_SQ_OP_MAP = {
    "Conv1": "input.1_0_0/nncf_smooth_quant",
}


class TestOVSQAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @pytest.fixture(params=[False, True], ids=["out_of_palce", "inplace"])
    def inplace_statistics(self, request) -> bool:
        return request.param

    def get_node_name_map(self, model_cls) -> Dict[str, str]:
        if model_cls is LinearMultiShapeModel:
            return OV_LINEAR_MODEL_MM_OP_MAP
        if model_cls is ConvTestModel:
            return OV_CONV_MODEL_MM_OP_MAP
        if model_cls is ShareWeghtsConvAndShareLinearModel:
            return {}
        raise NotImplementedError

    @staticmethod
    def get_target_node_name(command: TransformationCommand):
        return command.target_point.target_node_name

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
        ov_model = ov.convert_model(onnx_path, input=model.INPUT_SIZE)
        return ov_model

    @staticmethod
    def check_scales(model: ov.Model, reference_values: Dict[str, np.ndarray], model_cls) -> None:
        names_map = OV_LINEAR_MODEL_SQ_OP_MAP if model_cls is LinearMultiShapeModel else OV_CONV_MODEL_SQ_OP_MAP
        ops_list = {op.get_friendly_name(): op for op in model.get_ops()}
        for ref_names, ref_value in reference_values.items():
            const_nodes = []
            for ref_name in ref_names:
                node = ops_list[names_map[ref_name]]
                const_nodes.append(node.input(1).get_source_output().get_node())
            # Check unified group acutally shares one constant
            assert all(node is const_nodes[0] for node in const_nodes[1:])
            const_node = const_nodes[0]
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
        "node_metatype,weights_layout,reference_value",
        (
            (
                OVMatMulMetatype,
                (OVLayoutElem.C_OUT, OVLayoutElem.C_IN),
                1,
            ),
            (
                OVMatMulMetatype,
                (OVLayoutElem.C_IN,),
                0,
            ),
            (
                OVMatMulMetatype,
                (
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.C_IN,
                    OVLayoutElem.C_OUT,
                ),
                2,
            ),
            (
                OVConvolutionMetatype,
                (
                    OVLayoutElem.C_IN,
                    OVLayoutElem.C_OUT,
                    OVLayoutElem.SPATIAL,
                    OVLayoutElem.SPATIAL,
                ),
                1,
            ),
        ),
    )
    def test_get_weight_channel_axis(self, node_metatype, weights_layout, reference_value, mocker):
        mocker.patch(
            "nncf.quantization.algorithms.smooth_quant.openvino_backend.get_linear_weights_layout_from_node",
            return_value=weights_layout,
        )
        return super().test_get_weight_channel_axis(node_metatype, None, reference_value)

    @staticmethod
    def get_matmul_metatype():
        return OVMatMulMetatype
