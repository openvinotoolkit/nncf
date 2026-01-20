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


import numpy as np
import onnx
import torch

from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import build_graph
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.transformations.command_creation import ONNXCommandCreator
from nncf.quantization.algorithms.fast_bias_correction.onnx_backend import ONNXFastBiasCorrectionAlgoBackend
from tests.cross_fw.test_templates.test_fast_bias_correction import TemplateTestFBCAlgorithm
from tests.onnx.common import ModelBuilder


def get_data_from_node(model: onnx.ModelProto, node_name: str):
    data = [t for t in model.graph.initializer if t.name == node_name]
    if data:
        return onnx.numpy_helper.to_array(data[0])
    return None


class TestONNXFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: list) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def get_backend() -> ONNXFastBiasCorrectionAlgoBackend:
        return ONNXFastBiasCorrectionAlgoBackend

    @staticmethod
    def backend_specific_model(model, tmp_dir: str):
        onnx_path = f"{tmp_dir}/model.onnx"
        model.eval()
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=18, input_names=["input.1"])
        onnx_model = onnx.load(onnx_path)
        return onnx_model

    @staticmethod
    def fn_to_type(tensor):
        return np.array(tensor)

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input.1": tensor}

        return transform_fn

    @staticmethod
    def check_bias(model: onnx.ModelProto, ref_bias: list):
        ref_bias = np.array(ref_bias)
        nncf_graph = build_graph(model)
        for node in nncf_graph.get_all_nodes():
            if not is_node_with_bias(node):
                continue
            bias_value = get_bias_value(node, model)
            # TODO(AlexanderDokuchaev): return atol=0.0001 after fix 109189
            assert np.all(np.isclose(bias_value, ref_bias, atol=0.01)), f"{bias_value} != {ref_bias}"
            return
        msg = "Not found node with bias"
        raise ValueError(msg)


def _build_matmul_add_model() -> onnx.ModelProto:
    mb = ModelBuilder()

    x = mb.add_input("X", (2, 3))

    x = mb.add_matmul(x, (3, 3))
    x = mb.add_add(x, mb.add_initializer(np.array([1, 1, 1], dtype=np.float32)))

    x = mb.add_matmul(x, (3, 3))
    x = mb.add_add(x, mb.add_initializer(np.array([2, 2, 2], dtype=np.float32)))

    mb.add_output(x, (2, 3))

    return mb.build(opset_version=19, ir_version=9)


def test_update_bias_in_matmul_add():
    """
    Tests the ability to retrieve and update the value of the bias constant in a MatMul->Add subgraph,
    where the second input to the Add operation is a constant.
    """
    model = _build_matmul_add_model()
    graph = build_graph(model)

    nodes = [node for node in graph.get_all_nodes() if is_node_with_bias(node)]
    assert [x.node_name for x in nodes] == ["MatMul_0", "MatMul_2"]

    for matmul, data in zip(nodes, [[1, 1, 1], [2, 2, 2]]):
        bias = get_bias_value(matmul, model)
        bias_ref = np.array(data, dtype=np.float32)
        assert np.all(np.isclose(bias, bias_ref, atol=0.0001)), f"{bias} != {bias_ref}"

    layout = TransformationLayout()
    for matmul, data in zip(nodes, [[2, 2, 2], [1, 1, 1]]):
        new_bias = np.array(data, dtype=np.float32)
        layout.register(ONNXCommandCreator.create_command_to_update_bias(matmul, new_bias, graph))
    model = ModelTransformerFactory.create(model).transform(layout)

    for matmul, data in zip(nodes, [[2, 2, 2], [1, 1, 1]]):
        bias = get_bias_value(matmul, model)
        bias_ref = np.array(data, dtype=np.float32)
        assert np.all(np.isclose(bias, bias_ref, atol=0.0001)), f"{bias} != {bias_ref}"
