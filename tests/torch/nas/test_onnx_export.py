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
import numpy as np
import onnx
import onnxruntime as rt
import torch

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.torch.utils import get_model_device
from tests.torch.helpers import get_all_inputs_for_graph_node
from tests.torch.helpers import get_nodes_by_type
from tests.torch.nas.descriptors import THREE_CONV_TEST_DESC
from tests.torch.nas.helpers import ref_kernel_transform
from tests.torch.nas.test_all_elasticity import ThreeConvModel
from tests.torch.nas.test_all_elasticity import create_bnas_model_and_ctrl_by_test_desc


def check_onnx_weights(ctrl, path_to_onnx, ref_orig_weights, expected_num_nodes):
    ctrl.export_model(path_to_onnx)
    onnx_model = onnx.load(path_to_onnx)
    conv_nodes = get_nodes_by_type(onnx_model, "Conv")
    inputs = [get_all_inputs_for_graph_node(conv_node, onnx_model.graph) for conv_node in conv_nodes]
    actual_orig_weights = [torch.from_numpy(weight) for input_dict in inputs for weight in input_dict.values()]
    assert all(torch.equal(ref, act) for ref, act in zip(ref_orig_weights, actual_orig_weights))
    assert len(onnx_model.graph.node) == expected_num_nodes


def test_multi_elasticity_weights_in_onnx(tmp_path):
    _, ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC)
    multi_elasticity_handler = ctrl.multi_elasticity_handler
    orig_model = ThreeConvModel()
    conv1 = orig_model.conv1
    conv2 = orig_model.conv_to_skip
    last_conv = orig_model.last_conv

    path_to_onnx = tmp_path / "supernet.onnx"
    ref_orig_weights = [conv1.weight, conv2.weight, last_conv.weight, last_conv.bias]
    check_onnx_weights(ctrl, path_to_onnx, ref_orig_weights, 5)

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "kernel_stage.onnx"
    ref_weight1 = ref_kernel_transform(conv1.weight)
    ref_orig_weights = [ref_weight1, conv2.weight, last_conv.weight, last_conv.bias]
    check_onnx_weights(ctrl, path_to_onnx, ref_orig_weights, 5)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "depth_stage.onnx"
    ref_orig_weights = [ref_weight1, last_conv.weight, last_conv.bias]
    check_onnx_weights(ctrl, path_to_onnx, ref_orig_weights, 4)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "width_stage.onnx"
    ref_orig_weights = [ref_weight1[:1], last_conv.weight[:, :1, :, :], last_conv.bias[:1]]
    check_onnx_weights(ctrl, path_to_onnx, ref_orig_weights, 4)


def check_onnx_outputs(ctrl, model, path_to_onnx):
    model.eval()
    device = get_model_device(model)
    ctrl.export_model(path_to_onnx)
    input_sizes = model.INPUT_SIZE
    torch_input = torch.ones(input_sizes).to(device)
    np_input = np.ones(input_sizes)
    with torch.no_grad():
        torch_model_output = model(torch_input)
    sess = rt.InferenceSession(str(path_to_onnx))
    input_name = sess.get_inputs()[0].name
    onnx_model_output = sess.run(None, {input_name: np_input.astype(np.float32)})[0]
    assert np.allclose(torch_model_output.cpu().numpy(), onnx_model_output, rtol=1e-4)


def test_multi_elasticity_outputs_in_onnx(tmp_path):
    model, ctrl = create_bnas_model_and_ctrl_by_test_desc(THREE_CONV_TEST_DESC)
    multi_elasticity_handler = ctrl.multi_elasticity_handler

    path_to_onnx = tmp_path / "supernet.onnx"
    check_onnx_outputs(ctrl, model, path_to_onnx)

    multi_elasticity_handler.disable_all()
    multi_elasticity_handler.enable_elasticity(ElasticityDim.KERNEL)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "kernel_stage.onnx"
    check_onnx_outputs(ctrl, model, path_to_onnx)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.DEPTH)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "depth_stage.onnx"
    check_onnx_outputs(ctrl, model, path_to_onnx)

    multi_elasticity_handler.enable_elasticity(ElasticityDim.WIDTH)
    multi_elasticity_handler.activate_minimum_subnet()
    path_to_onnx = tmp_path / "width_stage.onnx"
    check_onnx_outputs(ctrl, model, path_to_onnx)
