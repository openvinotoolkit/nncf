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
import pytest

from nncf.common.graph.utils import get_first_nodes_of_type
from nncf.common.pruning.utils import get_last_nodes_of_type
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.torch.pruning.filter_pruning.algo import FilterPruningBuilder
from nncf.torch.pruning.utils import get_bn_for_conv_node_by_name
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import BranchingModel
from tests.torch.pruning.helpers import get_basic_pruning_config


@pytest.mark.parametrize(
    "total,sparsity_rate,multiple_of,ref",
    [(20, 0.2, None, 4), (20, 0.2, 8, 4), (20, 0.1, 2, 2), (20, 0.1, 5, 0), (20, 0.5, None, 4)],
)
def test_get_rounded_pruned_element_number(total, sparsity_rate, multiple_of, ref):
    if multiple_of is not None:
        result = get_rounded_pruned_element_number(total, sparsity_rate, multiple_of)
    else:
        result = get_rounded_pruned_element_number(total, sparsity_rate)
    assert ref == result

    if multiple_of is not None:
        assert (total - result) % multiple_of == 0


def test_get_bn_for_conv_node():
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    pruned_model, _ = create_compressed_model_and_algo_for_test(BigPruningTestModel(), config)

    conv1_name = "BigPruningTestModel/NNCFConv2d[conv1]/conv2d_0"
    bn = get_bn_for_conv_node_by_name(pruned_model, conv1_name)
    assert bn == pruned_model.bn1

    conv2_name = "BigPruningTestModel/NNCFConv2d[conv2]/conv2d_0"
    bn = get_bn_for_conv_node_by_name(pruned_model, conv2_name)
    assert bn == pruned_model.bn2

    up_name = "BigPruningTestModel/NNCFConvTranspose2d[up]/conv_transpose2d_0"
    bn = get_bn_for_conv_node_by_name(pruned_model, up_name)
    assert bn is None

    conv3_name = "BigPruningTestModel/NNCFConv2d[conv3]/conv2d_0"
    bn = get_bn_for_conv_node_by_name(pruned_model, conv3_name)
    assert bn is None


@pytest.mark.parametrize(
    ("model", "ref_first_module_names"),
    [
        (BigPruningTestModel, ["conv1"]),
        (BranchingModel, ["conv1", "conv2", "conv3"]),
    ],
)
def test_get_first_pruned_layers(model, ref_first_module_names):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    pruned_model, _ = create_compressed_model_and_algo_for_test(model(), config)

    first_pruned_nodes = get_first_nodes_of_type(
        pruned_model.nncf.get_original_graph(), FilterPruningBuilder(config).get_op_types_of_pruned_modules()
    )
    first_pruned_modules = [pruned_model.nncf.get_containing_module(n.node_name) for n in first_pruned_nodes]
    ref_first_modules = [getattr(pruned_model, module_name) for module_name in ref_first_module_names]
    assert set(first_pruned_modules) == set(ref_first_modules)


@pytest.mark.parametrize(
    ("model", "ref_last_module_names"),
    [
        (BigPruningTestModel, ["conv3"]),
        (BranchingModel, ["conv4", "conv5"]),
    ],
)
def test_get_last_pruned_layers(model, ref_last_module_names):
    config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    config["compression"]["algorithm"] = "filter_pruning"
    pruned_model, _ = create_compressed_model_and_algo_for_test(model(), config)

    last_pruned_nodes = get_last_nodes_of_type(
        pruned_model.nncf.get_original_graph(), FilterPruningBuilder(config).get_op_types_of_pruned_modules()
    )
    last_pruned_modules = [pruned_model.nncf.get_containing_module(n.node_name) for n in last_pruned_nodes]
    ref_last_modules = [getattr(pruned_model, module_name) for module_name in ref_last_module_names]
    assert set(last_pruned_modules) == set(ref_last_modules)
