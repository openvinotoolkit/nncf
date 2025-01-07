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

from tests.torch.helpers import load_exported_onnx_version
from tests.torch.pruning.helpers import BigPruningTestModel
from tests.torch.pruning.helpers import DiffConvsModel
from tests.torch.pruning.helpers import GroupNormModel
from tests.torch.pruning.helpers import PruningTestModelConcat
from tests.torch.pruning.helpers import PruningTestModelEltwise
from tests.torch.pruning.helpers import get_basic_pruning_config

pytestmark = pytest.mark.skip(reason="Export as actually deleting filters from the model is currently disabled.")


def find_value_by_name_in_list(obj_list, name):
    for obj in obj_list:
        if obj.name == name:
            return obj
    return None


def check_bias_and_weight_shape(node_name, onnx_model_proto, weight_shape, bias_shape):
    node_weight = find_value_by_name_in_list(onnx_model_proto.graph.initializer, node_name + ".weight")
    node_bias = find_value_by_name_in_list(onnx_model_proto.graph.initializer, node_name + ".bias")
    assert node_weight.dims == weight_shape
    assert node_bias.dims == bias_shape


def test_pruning_export_simple_model(tmp_path):
    model = BigPruningTestModel()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["pruning_init"] = 0.5
    nncf_config["compression"]["algorithm"] = "filter_pruning"
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    # Check that conv2 + BN were pruned by output filters
    # WARNING: starting from at least torch 1.7.0, torch.onnx.export will fuses BN into previous
    # convs if torch.onnx.export is done with `training=False`, so this test might fail.
    check_bias_and_weight_shape("nncf_module.conv2", onnx_model_proto, [16, 16, 3, 3], [16])
    check_bias_and_weight_shape("nncf_module.bn", onnx_model_proto, [16], [16])

    # Check that up was pruned by input filters
    check_bias_and_weight_shape("nncf_module.up", onnx_model_proto, [16, 32, 3, 3], [32])

    # Check that conv3 was pruned by input filters
    check_bias_and_weight_shape("nncf_module.conv3", onnx_model_proto, [1, 32, 5, 5], [1])


@pytest.mark.parametrize(
    ("prune_first", "ref_shapes"),
    [
        (False, [[[16, 1, 2, 2], [16]], [[16, 16, 2, 2], [16]], [[16, 16, 2, 2], [16]], [[16, 32, 3, 3], [16]]]),
        (True, [[[8, 1, 2, 2], [8]], [[16, 8, 2, 2], [16]], [[16, 8, 2, 2], [16]], [[16, 32, 3, 3], [16]]]),
    ],
)
def test_pruning_export_concat_model(tmp_path, prune_first, ref_shapes):
    model = PruningTestModelConcat()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"

    nncf_config["compression"]["params"]["prune_first_conv"] = prune_first
    nncf_config["compression"]["pruning_init"] = 0.5

    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    for i in range(1, 5):
        conv_name = "nncf_module.conv{}".format(i)
        check_bias_and_weight_shape(conv_name, onnx_model_proto, *ref_shapes[i - 1])


@pytest.mark.parametrize(
    ("prune_first", "ref_shapes"),
    [
        (False, [[[16, 1, 2, 2], [16]], [[16, 16, 2, 2], [16]], [[16, 16, 2, 2], [16]], [[16, 16, 3, 3], [16]]]),
        (True, [[[8, 1, 2, 2], [8]], [[16, 8, 2, 2], [16]], [[16, 8, 2, 2], [16]], [[16, 16, 3, 3], [16]]]),
    ],
)
def test_pruning_export_eltwise_model(tmp_path, prune_first, ref_shapes):
    model = PruningTestModelEltwise()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"

    nncf_config["compression"]["params"]["prune_first_conv"] = prune_first
    nncf_config["compression"]["pruning_init"] = 0.5
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    for i in range(1, 5):
        conv_name = "nncf_module.conv{}".format(i)
        check_bias_and_weight_shape(conv_name, onnx_model_proto, *ref_shapes[i - 1])


@pytest.mark.parametrize(
    ("prune_first", "ref_shapes"),
    [
        (False, [[[32, 1, 2, 2], [32]], [[32, 1, 1, 1], [32]], [[32, 32, 3, 3], [32]], [[16, 4, 1, 1], [16]]]),
        (True, [[[16, 1, 2, 2], [16]], [[16, 1, 1, 1], [16]], [[32, 16, 3, 3], [32]], [[16, 4, 1, 1], [16]]]),
    ],
)
def test_pruning_export_diffconvs_model(tmp_path, prune_first, ref_shapes):
    model = DiffConvsModel()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"

    nncf_config["compression"]["params"]["prune_first_conv"] = prune_first
    nncf_config["compression"]["pruning_init"] = 0.5
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)
    for i in range(1, 5):
        conv_name = "nncf_module.conv{}".format(i)
        check_bias_and_weight_shape(conv_name, onnx_model_proto, *ref_shapes[i - 1])


def test_pruning_export_groupnorm_model(tmp_path):
    model = GroupNormModel()
    nncf_config = get_basic_pruning_config(input_sample_size=[1, 1, 8, 8])
    nncf_config["compression"]["algorithm"] = "filter_pruning"

    nncf_config["compression"]["params"]["prune_first_conv"] = True
    nncf_config["compression"]["pruning_init"] = 0.5
    onnx_model_proto = load_exported_onnx_version(nncf_config, model, path_to_storage_dir=tmp_path)

    check_bias_and_weight_shape("nncf_module.conv1", onnx_model_proto, [8, 1, 1, 1], [8])
    check_bias_and_weight_shape("nncf_module.conv2", onnx_model_proto, [16, 8, 1, 1], [16])
