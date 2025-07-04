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

import tempfile

import numpy as np
import onnx
import onnxruntime
import pytest
from packaging import version

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.engine import ONNXEngine
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import set_metadata
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from tests.onnx.models import NonShapeModel
from tests.onnx.models import build_matmul_model


def check_engine_creation_and_inference(model: onnx.ModelProto, input_data: np.ndarray, reference_outputs: list[str]):
    engine = ONNXEngine(model)
    outputs = engine.infer(input_data)
    for result_name in outputs:
        assert result_name in reference_outputs


TARGET_LAYERS = [["Conv"], ["Conv", "Relu1"], ["Conv", "Relu1", "Reshape"]]
TARGET_LAYERS_OUTPUT = [["Y", "conv"], ["Y", "conv", "relu_1"], ["Y", "conv", "relu_1", "reshape"]]


@pytest.mark.parametrize("target_layers, target_layers_output", zip(TARGET_LAYERS, TARGET_LAYERS_OUTPUT))
def test_output_insertion(target_layers, target_layers_output):
    model = NonShapeModel().onnx_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nncf_input_node_next_onnx_nodes = {}
    for input_node in nncf_graph.get_input_nodes():
        next_nodes = nncf_graph.get_next_nodes(input_node)
        nncf_input_node_next_onnx_nodes[input_node.node_name] = [node.node_name for node in next_nodes]

    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, target_layer, 0)
        command = ONNXOutputInsertionCommand(target_point, nncf_input_node_next_onnx_nodes)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)

    input_data = {"X": np.ones([1, 3, 32, 32]).astype(np.float32)}
    check_engine_creation_and_inference(transformed_model, input_data, target_layers_output)


@pytest.mark.skipif(version.parse(onnxruntime.__version__) < version.parse("1.21.1"), reason="Requires onnx >= 1.21.1")
@pytest.mark.xfail(
    version.parse(onnx.__version__) >= version.parse("1.18.0"),
    reason="onnxruntime not support default IR for onnx==1.18.0",
)
def test_infer_for_model_with_external_data():
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        model = build_matmul_model()
        model_path = f"{temp_dir}/model.onnx"
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
            size_threshold=0,
            convert_attribute=False,
        )
        model = onnx.load_model(model_path, load_external_data=False)
        set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, temp_dir)
        check_engine_creation_and_inference(model, {"X": np.ones([2, 3], dtype=np.float32)}, ["A"])
