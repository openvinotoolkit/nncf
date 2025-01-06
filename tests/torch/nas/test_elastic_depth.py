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
from typing import List, Optional

import numpy as np
import onnx
import onnxruntime as rt
import pytest
import torch
from packaging import version
from torch import nn

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock
from nncf.experimental.torch.search_building_blocks.search_blocks import get_building_blocks
from nncf.torch import register_operator
from nncf.torch.layers import NNCFConv2d
from nncf.torch.utils import get_model_device
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import create_bootstrap_training_model_and_ctrl
from tests.torch.nas.helpers import do_training_step
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.test_models.resnet import ResNet18
from tests.torch.test_models.resnet import ResNet50
from tests.torch.test_models.resnet import ResNet50__elastic

###########################
# Helpers
###########################

RESNET50_INPUT_SIZE = [1, 3, 32, 32]
INCEPTION_INPUT_SIZE = [2, 3, 299, 299]


class BasicTestSuperNet(nn.Module):
    INPUT_SIZE = [1, 1, 5, 5]

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 5, weight_init=1, bias_init=1)
        self.conv_to_skip = create_conv(3, 3, 1, weight_init=2, bias_init=2)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv_to_skip(o1)
        return o1 + o2


class DepthBasicConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 15, 15]

    def __init__(self, depth=3):
        super().__init__()
        self._depth = depth
        self._skipped_layers = []
        self.conv1 = create_conv(1, 3, 3, weight_init=1, bias_init=1, padding=1)
        self.branch_with_blocks = nn.Sequential()
        for idx in range(depth):
            conv = create_conv(3, 3, 5, weight_init=idx + 1, bias_init=idx + 1, padding=2)
            self.branch_with_blocks.add_module("conv{}".format(idx), conv)
        self.last_conv = create_conv(3, 1, 1)

    def forward(self, x):
        output = self.conv1(x)

        for name, module in self.branch_with_blocks._modules.items():
            if name not in self._skipped_layers:
                output = module(output)
        output = self.last_conv(output)
        return output

    def set_skipped_layers(self, skipped_layers: Optional[List] = None):
        if skipped_layers is None:
            skipped_layers = []
        self._skipped_layers = skipped_layers

    def get_skipped_layers(self):
        return self._skipped_layers


BASIC_ELASTIC_DEPTH_PARAMS = {
    "skipped_blocks": [
        [
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
        ]
    ],
    "skipped_blocks_dependencies": {0: [0]},
    "skipped_blocks_state": [
        {
            "start_node_name": "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
            "end_node_name": "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
        }
    ],
}


@pytest.fixture()
def basic_model_and_device():
    model = DepthBasicConvTestModel(depth=3)
    device = move_model_to_cuda_if_available(model)
    return model, device


def init_weight(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
        m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.fill_(0.01)


def reset_bn_stats(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()


###########################
# Behavior
###########################


def test_not_matched_add_node():
    model = BasicTestSuperNet()
    device = move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=model.INPUT_SIZE)
    model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)
    skipped_blocks = [BuildingBlock("BasicTestSuperNet/NNCFConv2d[conv1]/conv2d_0", "/nncf_model_output_0")]
    ctx = model.nncf.get_tracing_context()
    ctx.set_elastic_blocks(skipped_blocks)
    ctx.elastic_depth = True
    ctx.set_active_skipped_block(list(range(0, len(ctx.skipped_blocks))))

    input_ = torch.ones(model.INPUT_SIZE).to(device)
    model(input_)


@pytest.mark.parametrize("model_creator", (DepthBasicConvTestModel, BasicTestSuperNet))
def test_reproduce_error_with_parsing_node_id(model_creator):
    model = model_creator()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=model.INPUT_SIZE)
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)
    get_building_blocks(compressed_model)


###########################
# Check outputs
###########################


def test_skip_one_block_resnet18(mocker):
    model = ResNet18()
    device = move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    nncf_config["bootstrapNAS"] = {
        "training": {
            "elasticity": {
                "available_elasticity_dims": [ElasticityDim.DEPTH.value],
                "depth": {
                    "skipped_blocks": [
                        [
                            "ResNet/Sequential[layer1]/BasicBlock[0]/relu_0",  # 1
                            "ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0",
                        ]
                    ],
                },
            }
        }
    }
    compressed_model, _ = create_bootstrap_training_model_and_ctrl(model, nncf_config)

    ctx = compressed_model.nncf.get_tracing_context()
    spy_agent_conv2d = mocker.spy(NNCFConv2d, "__call__")
    spy_agent_bn = mocker.spy(torch.nn.BatchNorm2d, "__call__")

    ctx.elastic_depth = True  # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    compressed_model(torch.ones(RESNET50_INPUT_SIZE).to(device))

    assert id(spy_agent_conv2d.call_args_list[2][0][1]) == id(spy_agent_bn.call_args_list[2][0][1])  # TracedTensor

    # check torch.tensor.data by element
    assert (spy_agent_conv2d.call_args_list[2][0][1] == spy_agent_bn.call_args_list[2][0][1]).sum() == np.prod(
        spy_agent_bn.call_args_list[2][0][1].shape
    )

    spy_agent_conv2d.reset_mock()
    spy_agent_bn.reset_mock()

    ctx.elastic_depth = False
    compressed_model(torch.ones(RESNET50_INPUT_SIZE).to(device))
    assert id(spy_agent_conv2d.call_args_list[2][0][1]) != id(spy_agent_bn.call_args_list[2][0][1])  # TracedTensor


def test_can_export_model_with_one_skipped_block_resnet18(tmp_path):
    model = ResNet18()
    move_model_to_cuda_if_available(model)

    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    skipped_blocks = [
        BuildingBlock(
            "ResNet/Sequential[layer1]/BasicBlock[0]/relu_0",
            "ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0",
        )
    ]
    orig_onnx_model_path = tmp_path / "resnet18.onnx"
    onnx_model_without_block_path = tmp_path / "resnet18_with_one_skipped_block.onnx"

    compressed_model, ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)
    compressed_model.nncf.get_tracing_context().set_elastic_blocks(skipped_blocks)
    # export model to onnx
    ctx = compressed_model.nncf.get_tracing_context()
    ctrl.export_model(str(orig_onnx_model_path))

    ctx.elastic_depth = True  # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    ctrl.export_model(str(onnx_model_without_block_path))

    # load onnx graphs

    load_model_fn = onnx.load_model
    onnx_resnet18_without_one_block = load_model_fn(onnx_model_without_block_path)
    onnx_resnet18_orig = load_model_fn(orig_onnx_model_path)

    # Temporary variable for number of nodes is required to avoid hang. When assertion happens on calculation of number
    # of nodes it tries to output the whole expression (onnx node), and sometimes it causes pytest to freeze.
    num_all_nodes = len(onnx_resnet18_orig.graph.node)
    num_not_skipped_nodes = len(onnx_resnet18_without_one_block.graph.node)
    ref_num_nodes = 65
    ref_not_skipped_nodes = 63
    if version.parse(torch.__version__) < version.parse("1.12"):
        # different ONNX format for older pytorch version - no Identity nodes
        ref_num_nodes = 49
        ref_not_skipped_nodes = 48
    assert num_all_nodes == ref_num_nodes
    assert num_not_skipped_nodes == ref_not_skipped_nodes

    input_tensor = np.ones(nncf_config["input_info"][0]["sample_size"])
    device = get_model_device(compressed_model)
    torch_input = torch.tensor(input_tensor, dtype=torch.float32).to(device)
    with torch.no_grad():
        torch_model_output = compressed_model(torch_input)

    # ONNXRuntime
    sess = rt.InferenceSession(str(onnx_model_without_block_path))
    input_name = sess.get_inputs()[0].name
    onnx_model_output = sess.run(None, {input_name: input_tensor.astype(np.float32)})[0]
    assert np.allclose(torch_model_output.cpu().numpy(), onnx_model_output, atol=1.0e-5)


def test_skip_one_block_resnet50(mocker):
    model = ResNet50()
    device = move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    skipped_blocks = [
        BuildingBlock(
            "ResNet/Sequential[layer1]/Bottleneck[1]/relu_2", "ResNet/Sequential[layer1]/Bottleneck[2]/relu_2"
        )
    ]
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ctx = compressed_model.nncf.get_tracing_context()
    ctx.set_elastic_blocks(skipped_blocks)
    spy_agent = mocker.spy(NNCFConv2d, "__call__")
    ctx.elastic_depth = True  # activate mode with elastic depth
    ctx.set_active_skipped_block([0])
    compressed_model(torch.ones(RESNET50_INPUT_SIZE).to(device))

    assert (
        id(spy_agent.call_args_list[8][0][1])
        == id(spy_agent.call_args_list[9][0][1])
        == id(spy_agent.call_args_list[10][0][1])
    )  # TracedTensor

    # check torch.tensor.data by element
    assert (spy_agent.call_args_list[8][0][1] == spy_agent.call_args_list[9][0][1]).sum() == np.prod(
        spy_agent.call_args_list[9][0][1].shape
    )

    spy_agent.reset_mock()

    ctx.elastic_depth = False
    compressed_model(torch.ones(RESNET50_INPUT_SIZE).to(device))

    assert id(spy_agent.call_args_list[8][0][1]) != id(spy_agent.call_args_list[9][0][1])  # TracedTensor


def get_ref_output_model_after_backward__with_manual_skipping():
    # forward and backward with "manual" mechanism skipping block

    model = DepthBasicConvTestModel(depth=3)
    device = move_model_to_cuda_if_available(model)
    optimizer_for_model = torch.optim.Adam(model.parameters(), lr=0.01)

    # set skipped layer
    model.set_skipped_layers(["conv1"])

    return do_training_step(model, optimizer_for_model, torch.ones(model.INPUT_SIZE).to(device))


def get_model_with_elastic_depth(model, input_sample_sizes, skipped_blocks):
    nncf_config = get_empty_config(input_sample_sizes=input_sample_sizes)

    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)
    ctx = compressed_model.nncf.get_tracing_context()
    ctx.set_elastic_blocks(skipped_blocks)
    ctx.elastic_depth = True
    ctx.set_active_skipped_block([0])
    return compressed_model


def get_ref_output_model_after_backward__with_elastic_depth():
    # forward and backward with elastic_depth
    target_model = DepthBasicConvTestModel(depth=3)
    device = move_model_to_cuda_if_available(target_model)
    skipped_block = [
        BuildingBlock(
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
        )
    ]
    compressed_model = get_model_with_elastic_depth(target_model, target_model.INPUT_SIZE, skipped_block)

    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=0.01)

    ctx = compressed_model.nncf.get_tracing_context()
    ctx.elastic_depth = True
    ctx.set_active_skipped_block([0])

    return do_training_step(compressed_model, optimizer, torch.ones(target_model.INPUT_SIZE).to(device))


def get_ref_output_resnet50_after_backward__with_elastic_depth():
    # forward and backward with elastic_depth
    target_model = ResNet50()
    device = move_model_to_cuda_if_available(target_model)
    with torch.no_grad():
        target_model.apply(init_weight)
    skipped_block = [
        BuildingBlock(
            "ResNet/Sequential[layer2]/Bottleneck[0]/relu_2", "ResNet/Sequential[layer2]/Bottleneck[1]/relu_2"
        )
    ]

    compressed_model = get_model_with_elastic_depth(target_model, RESNET50_INPUT_SIZE, skipped_block)

    with torch.no_grad():
        compressed_model.apply(reset_bn_stats)
    compressed_model.train()
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=0.01)

    ctx = compressed_model.nncf.get_tracing_context()
    ctx.elastic_depth = True
    ctx.set_active_skipped_block([0])

    return do_training_step(compressed_model, optimizer, torch.ones(RESNET50_INPUT_SIZE).to(device))


def get_output_model__with_manual_skipping():
    model = DepthBasicConvTestModel(depth=3)
    model.set_skipped_layers(["conv1"])
    output = model(torch.ones(model.INPUT_SIZE))
    return output


def get_output_model__with_elastic_depth():
    target_model = DepthBasicConvTestModel(depth=3)
    skipped_block = [
        BuildingBlock(
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv0]/conv2d_0",
            "DepthBasicConvTestModel/Sequential[branch_with_blocks]/NNCFConv2d[conv1]/conv2d_0",
        )
    ]
    compressed_model = get_model_with_elastic_depth(target_model, target_model.INPUT_SIZE, skipped_block)
    output = compressed_model(torch.ones(target_model.INPUT_SIZE))

    return output


def get_ref_output_resnet50_after_backward__with_manual_skipping():
    # forward and backward with "manual" mechanism skipping block
    idx_skip_blocks = [4]
    model = ResNet50__elastic(idx_skip_blocks)
    device = move_model_to_cuda_if_available(model)
    with torch.no_grad():
        model.apply(init_weight)
    optimizer_for_model = torch.optim.Adam(model.parameters(), lr=0.01)

    with torch.no_grad():
        model.apply(reset_bn_stats)
    return do_training_step(model, optimizer_for_model, torch.ones(RESNET50_INPUT_SIZE).to(device))


def test_correct_grad_when_block_skipped():
    output_ref = get_ref_output_model_after_backward__with_manual_skipping()
    output = get_ref_output_model_after_backward__with_elastic_depth()

    assert (output_ref == output).sum() == np.prod(output.shape)


def test_correct_output_with_active_skipped_block():
    output_ref = get_output_model__with_manual_skipping()
    output = get_output_model__with_elastic_depth()
    assert (output_ref == output).sum() == np.prod(output.shape)


def test_correct_grad_when_block_skipped__resnet50():
    from torch.backends import cudnn

    from nncf.torch.utils import manual_seed

    manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False

    output_ref = get_ref_output_resnet50_after_backward__with_manual_skipping()
    output = get_ref_output_resnet50_after_backward__with_elastic_depth()

    assert (output_ref == output).sum() == np.prod(output.shape)


class TwoPermute(nn.Module):
    INPUT_SIZE = [1, 2]

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        num_dims = len(x.size())
        permute_dims = tuple(range(num_dims - 1, -1, -1))
        x = torch.permute(x, dims=permute_dims)
        return torch.permute(x, dims=permute_dims)


class ChunkConcat(nn.Module):
    INPUT_SIZE = [2, 2]

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        chunks = torch.chunk(x, 2, 0)
        return torch.cat(chunks)


class TwoBranchesBeforeInput(nn.Module):
    INPUT_SIZE = [1, 2]

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.dummy + x


@register_operator()
def custom_identity(x):
    return x


class TwoBranchesAfterInput(nn.Module):
    INPUT_SIZE = [1, 2]

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = custom_identity(x)
        return x * self.dummy + x


@pytest.mark.parametrize("model_creator", (TwoPermute, ChunkConcat, TwoBranchesBeforeInput, TwoBranchesAfterInput))
def test_can_skip_trivial_block(model_creator):
    model = model_creator()
    input_sample_sizes = model.INPUT_SIZE
    nncf_config = get_empty_config(input_sample_sizes=input_sample_sizes)
    nncf_config["bootstrapNAS"] = {
        "training": {
            "elasticity": {"available_elasticity_dims": [ElasticityDim.DEPTH.value], "depth": {"min_block_size": 1}}
        }
    }
    input_ = torch.ones(input_sample_sizes)
    compressed_model, _ = create_bootstrap_training_model_and_ctrl(model, nncf_config)

    # activate elastic depth to skip the only one possible block
    ctx = compressed_model.nncf.get_tracing_context()
    ctx.elastic_depth = True
    ctx.set_active_skipped_block([0])

    output = compressed_model(input_)

    # both permute should be skipped, input stays the same
    assert output.size() == input_.size()
    # sanity check that elastic depth is working
    assert id(input_) == id(output)


###########################
# Dynamic Graph
###########################


def test_check_dinamic_graph_not_grow():
    model = ResNet50()
    device = move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    skipped_blocks = [
        BuildingBlock(
            "ResNet/Sequential[layer1]/Bottleneck[1]/relu_2", "ResNet/Sequential[layer1]/Bottleneck[2]/relu_2"
        ),
        BuildingBlock(
            "ResNet/Sequential[layer2]/Bottleneck[2]/relu_2", "ResNet/Sequential[layer2]/Bottleneck[3]/relu_2"
        ),
        BuildingBlock(
            "ResNet/Sequential[layer3]/Bottleneck[4]/relu_2", "ResNet/Sequential[layer3]/Bottleneck[5]/relu_2"
        ),
        BuildingBlock(
            "ResNet/Sequential[layer4]/Bottleneck[1]/relu_2", "ResNet/Sequential[layer4]/Bottleneck[2]/relu_2"
        ),
    ]
    compressed_model, _ = create_compressed_model_and_algo_for_test(model, nncf_config)

    ctx = compressed_model.nncf.get_tracing_context()
    ctx.set_elastic_blocks(skipped_blocks)
    nodes_count = ctx.graph.get_nodes_count()
    ctx.elastic_depth = True  # activate mode with elastic depth
    ctx.set_active_skipped_block([0, 1, 2, 3])

    for _ in range(10):
        compressed_model(torch.ones(RESNET50_INPUT_SIZE).to(device))
        assert nodes_count == ctx.graph.get_nodes_count()


###########################
# Depth Indicator
###########################


def test_validate_depth_config():
    model = ResNet50()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    nncf_config["bootstrapNAS"] = {
        "training": {"elasticity": {"available_elasticity_dims": [ElasticityDim.DEPTH.value]}}
    }
    _, controller = create_bootstrap_training_model_and_ctrl(model, nncf_config)
    depth_handler = controller.multi_elasticity_handler.depth_handler

    check = [[1, 4], [1, 3], [2], [0], [1, 9], [1], [2], [3], [2, 3], [5, 6, 7, 8, 9]]
    valid_1 = [[1, 4], [1], [], [], [1, 9], [1], [], [], [], [9]]

    for i, valid_value in enumerate(valid_1):
        assert valid_value == depth_handler._remove_inconsistent_blocks(check[i])

    depth_handler.depth_indicator = 2
    for i, valid_value in enumerate(valid_1):
        assert valid_value == depth_handler._remove_inconsistent_blocks(check[i])

    check = [[0, 1], [2, 3, 4], [0, 1, 3, 4, 8, 9, 10, 11], [0, 1, 3, 4, 8, 9, 10]]
    valid_2 = [[0, 1], [3, 4], [0, 1, 3, 4, 8, 9, 10, 11], [0, 1, 3, 4, 8, 9]]
    for i, valid_value in enumerate(valid_2):
        assert valid_value == depth_handler._remove_inconsistent_blocks(check[i])


def test_change_depth_indicator():
    model = ResNet50()
    move_model_to_cuda_if_available(model)
    nncf_config = get_empty_config(input_sample_sizes=RESNET50_INPUT_SIZE)
    nncf_config["bootstrapNAS"] = {
        "training": {"elasticity": {"available_elasticity_dims": [ElasticityDim.DEPTH.value]}}
    }
    _, ctrl = create_bootstrap_training_model_and_ctrl(model, nncf_config)

    ctrl.multi_elasticity_handler.depth_handler.activate_minimum_subnet()
    assert ctrl.multi_elasticity_handler.depth_handler.get_active_config() == [1, 4, 9, 11]

    ctrl.multi_elasticity_handler.depth_handler.depth_indicator = 2
    ctrl.multi_elasticity_handler.depth_handler.activate_minimum_subnet()
    assert ctrl.multi_elasticity_handler.depth_handler.get_active_config() == [0, 1, 3, 4, 8, 9, 10, 11]

    ctrl.multi_elasticity_handler.depth_handler.depth_indicator = 1
    ctrl.multi_elasticity_handler.depth_handler.activate_subnet_for_config([0, 1])
    assert ctrl.multi_elasticity_handler.depth_handler.get_active_config() == [1]

    ctrl.multi_elasticity_handler.depth_handler.depth_indicator = 2
    ctrl.multi_elasticity_handler.depth_handler.activate_subnet_for_config([0, 1])
    assert ctrl.multi_elasticity_handler.depth_handler.get_active_config() == [0, 1]

    ctrl.multi_elasticity_handler.depth_handler.depth_indicator = 1
    ctrl.multi_elasticity_handler.depth_handler.activate_subnet_for_config([0, 1])
    assert ctrl.multi_elasticity_handler.depth_handler.get_active_config() == [1]
