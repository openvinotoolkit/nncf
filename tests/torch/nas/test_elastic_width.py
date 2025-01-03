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
import copy

import pytest
import torch
from torch import nn

from nncf.experimental.torch.nas.bootstrapNAS.elasticity.visualization import SubnetGraph
from nncf.torch.utils import get_model_device
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_empty_config
from tests.torch.nas.creators import create_bootstrap_nas_training_algo
from tests.torch.nas.creators import create_bootstrap_training_model_and_ctrl
from tests.torch.nas.creators import create_two_conv_width_supernet
from tests.torch.nas.helpers import compare_tensors_ignoring_the_order
from tests.torch.nas.helpers import move_model_to_cuda_if_available
from tests.torch.nas.models.synthetic import ConvTwoFcTestModel
from tests.torch.nas.models.synthetic import TwoConvAddConvTestModel
from tests.torch.nas.models.synthetic import TwoConvMeanModel
from tests.torch.nas.models.synthetic import TwoSequentialConvBNTestModel
from tests.torch.nas.models.synthetic import TwoSequentialFcLNTestModel
from tests.torch.nas.test_all_elasticity import NAS_MODELS_SCOPE
from tests.torch.nas.test_elastic_kernel import do_conv2d

###########################
# Helpers
###########################
from tests.torch.test_compressed_graph import get_full_path_to_the_graph


@pytest.fixture(
    name="basic_model",
    params=(
        TwoSequentialFcLNTestModel,
        ConvTwoFcTestModel,
        TwoConvAddConvTestModel,
        TwoSequentialConvBNTestModel,
    ),
)
def fixture_basic_model(request):
    model_cls = request.param
    model = model_cls()
    move_model_to_cuda_if_available(model)
    return model


BASIC_ELASTIC_WIDTH_PARAMS = {
    "filter_importance": "L1",
    "external_importance_path": None,
    "add_dynamic_inputs": None,
    "max_num_widths": -1,
    "min_width": 1,
    "overwrite_groups": None,
    "overwrite_groups_widths": None,
    "width_multipliers": None,
    "width_step": 1,
}


###########################
# Behavior
###########################


def test_set_elastic_width_by_value_not_from_list():
    width_handler, _ = create_two_conv_width_supernet()
    with pytest.raises(ValueError):
        width_handler.activate_subnet_for_config({0: 16})


def test_add_dynamic_inputs():
    elasticity_params = {
        "width": {
            "overwrite_groups": [["TwoConvMeanModel/NNCFConv2d[conv1]/conv2d_0"]],
            "overwrite_groups_widths": [[3, 1]],
            "add_dynamic_inputs": ["TwoConvMeanModel/NNCFConv2d[last_conv]/conv2d_0"],
        }
    }
    width_handler, _ = create_two_conv_width_supernet(elasticity_params=elasticity_params, model=TwoConvMeanModel)
    width_handler.activate_minimum_subnet()
    input_channel, output_channel = width_handler.get_active_in_out_width_values()
    assert input_channel["TwoConvMeanModel/NNCFConv2d[last_conv]/conv2d_0"] == 1
    assert output_channel["TwoConvMeanModel/NNCFConv2d[conv1]/conv2d_0"] == 1


###########################
# Output checking
###########################


def test_elastic_width_with_maximum_value():
    _, supernet = create_two_conv_width_supernet()
    device = get_model_device(supernet)
    x = torch.ones(supernet.INPUT_SIZE).to(device)
    actual_output = supernet(x)

    shallow_copy = supernet.nncf.get_clean_shallow_copy()
    ref_output = shallow_copy(x)

    assert torch.equal(actual_output, ref_output)


def test_elastic_width_with_intermediate_value():
    width_handler, supernet = create_two_conv_width_supernet(elasticity_params={"width": BASIC_ELASTIC_WIDTH_PARAMS})
    device = get_model_device(supernet)

    ACTIVE_WIDTH = 1
    x = torch.ones(supernet.INPUT_SIZE).to(device)
    width_handler.activate_subnet_for_config({0: ACTIVE_WIDTH})
    actual_output = supernet(x)

    conv1 = supernet.conv1
    ref_weights = conv1.weight[:ACTIVE_WIDTH, :, :, :]
    ref_bias = conv1.bias[:ACTIVE_WIDTH]
    o1 = do_conv2d(conv1, x, weight=ref_weights, bias=ref_bias)
    ref_output = supernet.last_conv(o1)

    assert torch.equal(actual_output, ref_output)


def test_width_activation(basic_model):
    config = get_empty_config(input_sample_sizes=basic_model.INPUT_SIZE)
    config.update(
        {
            "bootstrapNAS": {
                "training": {
                    "elasticity": {"width": BASIC_ELASTIC_WIDTH_PARAMS},
                }
            }
        }
    )
    ref_model = copy.deepcopy(basic_model)
    ref_model.eval()
    model, ctrl = create_bootstrap_training_model_and_ctrl(basic_model, config)
    model.eval()

    device = next(model.parameters()).device
    dummy_input = torch.Tensor([1]).reshape(basic_model.INPUT_SIZE).to(device)
    width_handler = ctrl.multi_elasticity_handler.width_handler
    width_handler.reorganize_weights()
    width_handler.activate_minimum_subnet()
    actual_output = model(dummy_input)
    ref_output = ref_model.get_minimal_subnet_output(dummy_input)
    compare_tensors_ignoring_the_order(ref_output, actual_output)


def test_width_reorg(basic_model):
    config = get_empty_config(input_sample_sizes=basic_model.INPUT_SIZE)
    config["bootstrapNAS"] = {}
    model, ctrl = create_bootstrap_training_model_and_ctrl(basic_model, config)
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.Tensor([1]).reshape(basic_model.INPUT_SIZE).to(device)
    before_reorg = model(dummy_input)
    ctrl.multi_elasticity_handler.width_handler.reorganize_weights()
    after_reorg = model(dummy_input)

    model.check_reorg()
    compare_tensors_ignoring_the_order(after_reorg, before_reorg)


def test_width_custom_external_reorg(basic_model, tmp_path):
    config = get_empty_config(input_sample_sizes=basic_model.INPUT_SIZE)
    external_importance = basic_model.IMPORTANCE
    external_importance_tempfile = tmp_path / "importance_file"
    torch.save(external_importance, external_importance_tempfile)
    config.update(
        {
            "bootstrapNAS": {
                "training": {
                    "elasticity": {
                        "width": {
                            "filter_importance": "external",
                            "external_importance_path": external_importance_tempfile,
                        }
                    },
                }
            }
        }
    )
    model, ctrl = create_bootstrap_training_model_and_ctrl(basic_model, config)
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.Tensor([1]).reshape(basic_model.INPUT_SIZE).to(device)
    before_reorg = model(dummy_input)
    ctrl.multi_elasticity_handler.width_handler.reorganize_weights()
    after_reorg = model(dummy_input)

    model.check_custom_external_reorg()
    compare_tensors_ignoring_the_order(after_reorg, before_reorg)


@pytest.fixture(name="nas_model_name", scope="function", params=NAS_MODELS_SCOPE)
def fixture_nas_model_name(request):
    return request.param


def test_weight_reorg(nas_model_name, _seed):
    if nas_model_name in ["inception_v3"]:
        pytest.skip("Skip test for Inception-V3 because of invalid padding update in elastic kernel (60990)")

    compressed_model, training_ctrl, dummy_forward = create_bootstrap_nas_training_algo(nas_model_name)
    compressed_model.eval()
    before_reorg = dummy_forward(compressed_model)
    width_handler = training_ctrl.multi_elasticity_handler.width_handler
    width_handler.reorganize_weights()
    after_reorg = dummy_forward(compressed_model)
    atol = 1e-2
    if nas_model_name == "efficient_net_b0":
        atol = 4e-1
    if nas_model_name == "ssd_vgg":
        atol = 5e-1
    if nas_model_name == "squeezenet1_0":
        atol = 1
    compare_tensors_ignoring_the_order(after_reorg, before_reorg, atol=atol)


###########################
# Corner case
###########################


def test_multi_forward_nodes():
    class TestModel(nn.Module):
        INPUT_SIZE = [1, 1, 5, 5]

        def __init__(self):
            super().__init__()
            self.conv1 = create_conv(1, 3, 5)
            self.multi_forward_conv = create_conv(3, 3, 1)
            self.last_conv = create_conv(3, 1, 1)

        def forward(self, x):
            #
            #         / ---- \
            # conv1 ->        add -> multi_forward_conv -> last_conv
            #         \      /
            #     multi_forward_conv
            #
            o1 = self.conv1(x)
            o2 = self.multi_forward_conv(o1)
            o3 = o2 + o1
            o4 = self.multi_forward_conv(o3)
            return self.last_conv(o4)

    config = get_empty_config(input_sample_sizes=TestModel.INPUT_SIZE)
    config["compression"] = {"algorithm": "bootstrapNAS"}
    model, ctrl = create_bootstrap_training_model_and_ctrl(TestModel(), config)
    multi_elasticity_handler = ctrl.multi_elasticity_handler
    # multi_elasticity_handler.enable_all()
    # multi_elasticity_handler.activate_supernet()
    width_graph = SubnetGraph(model.nncf.get_graph(), multi_elasticity_handler).get()
    path_to_dot = get_full_path_to_the_graph("multi_forward_node.dot", "nas")
    compare_nx_graph_with_reference(width_graph, path_to_dot)
