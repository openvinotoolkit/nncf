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

from typing import Any

import onnx
import pytest
import torch
from torch import nn

from nncf import NNCFConfig
from nncf.torch import patch_torch_operators
from nncf.torch.dynamic_graph.patch_pytorch import unpatch_torch_operators
from nncf.torch.exporter import PTExporter
from tests.torch.helpers import MockModel
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_conv
from tests.torch.helpers import get_nodes_by_type
from tests.torch.helpers import load_exported_onnx_version
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import SingleLayerModelDesc
from tests.torch.test_compressed_graph import get_basic_quantization_config


class ModelForIONamingTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 1)
        self.linear = torch.nn.Linear(1, 1)
        self.embedding = torch.nn.Embedding(1, 1)

    def forward(self, conv_input, linear_input, embedding_input):
        return [
            self.conv(conv_input),
            {"linear": self.linear(linear_input), "embedding": self.embedding(embedding_input)},
        ]


def test_io_nodes_naming_scheme(tmp_path):
    config = NNCFConfig.from_dict(
        {
            "input_info": [
                {
                    "sample_size": [1, 1, 1],
                },
                {
                    "sample_size": [1, 1],
                },
                {"sample_size": [1, 1], "type": "long", "filler": "zeros"},
            ]
        }
    )
    onnx_model_proto = load_exported_onnx_version(config, ModelForIONamingTest(), tmp_path)
    conv_node = next(iter(get_nodes_by_type(onnx_model_proto, "Conv")))
    linear_node = next(iter(get_nodes_by_type(onnx_model_proto, "Gemm")))
    embedding_node = next(iter(get_nodes_by_type(onnx_model_proto, "Gather")))

    for idx, node in enumerate([conv_node, linear_node, embedding_node]):
        input_tensor_ids = [x for x in node.input if "input" in x]
        assert len(input_tensor_ids) == 1
        assert input_tensor_ids[0] == f"input.{idx}"

        assert len(node.output) == 1
        assert node.output[0] == f"output.{idx}"


@pytest.mark.parametrize(
    "save_format, refs",
    (
        ("onnx", ("onnx", {"opset_version": PTExporter._ONNX_DEFAULT_OPSET})),
        ("onnx_9", ("onnx", {"opset_version": 9})),
        ("onnx_10", ("onnx", {"opset_version": 10})),
        ("onnx_11", ("onnx", {"opset_version": 11})),
        ("onnx_0", ValueError),
        ("onnx_onnx", ValueError),
    ),
)
def test_exporter_parser_format(save_format: str, refs: Any):
    try:
        save_format, args = PTExporter.parse_format(save_format)
    except Exception as e:
        if not isinstance(refs, tuple):
            assert isinstance(e, refs)
            return

    assert save_format == refs[0]
    assert args == refs[1]


@pytest.mark.parametrize("save_format, ref_opset", (("onnx", 13), ("onnx_9", 9), ("onnx_10", 10), ("onnx_11", 11)))
def test_exported_version(tmp_path: str, save_format: str, ref_opset: int):
    model = MockModel()
    config = NNCFConfig()
    config.update({"input_info": {"sample_size": [1, 1, 1, 1]}})

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    onnx_checkpoint_path = tmp_path / "model.onnx"
    compression_ctrl.export_model(onnx_checkpoint_path, save_format)
    model_proto = onnx.load_model(onnx_checkpoint_path)

    assert model_proto.opset_import[0].version == ref_opset


class MultiParamForwardModel(torch.nn.Module):
    def forward(self, param1, param2, param3=None):
        return param1, param2


def test_can_export_single_batch_bn(tmp_path):
    test_path = tmp_path.joinpath("test.onnx")
    synthetic_model_desc = SingleLayerModelDesc(layer=nn.BatchNorm2d(4), input_sample_sizes=([1, 4, 1, 1]))
    config = get_basic_quantization_config(
        input_sample_sizes=synthetic_model_desc.get_input_sample_sizes(),
        input_info=synthetic_model_desc.create_input_info(),
    )
    register_bn_adaptation_init_args(config)
    model = synthetic_model_desc.get_model()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.export_model(str(test_path))
    assert test_path.exists()


def test_can_export_with_model_args(tmp_path):
    # Torch now parses the function signature and sets up default parameters for unprovided
    # arguments on its own. Need to rethink and possibly deprecate model_args parameter.
    test_path = tmp_path.joinpath("test.onnx")
    model = MultiParamForwardModel()
    config = get_basic_quantization_config(input_info=[{"sample_size": [1, 1, 1, 1]}, {"sample_size": [1, 1, 1, 1]}])
    register_bn_adaptation_init_args(config)
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    compression_ctrl.export_model(str(test_path), model_args=({"param3": 42},))
    assert test_path.exists()


class LinearTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(3, 3, 1)
        self.bn1 = create_bn(3)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = create_conv(3, 1, 1)
        self.bn2 = create_bn(1)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        return x


@pytest.mark.parametrize(
    "compression_section",
    [{}, {"compression": {"algorithm": "quantization"}}, {"compression": {"algorithm": "filter_pruning"}}],
    ids=["none", "quantization", "filter_pruning"],
)
def test_preserves_onnx_node_name_format(tmp_path, compression_section):
    model = LinearTestModel()
    model.eval().cpu()
    try:
        unpatch_torch_operators()
        without_nncf_path = tmp_path / "without_nncf.onnx"
        torch.onnx.export(
            model,
            torch.ones([1, 3, 32, 32]),
            without_nncf_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=False,
        )
        original_model_proto = onnx.load_model(str(without_nncf_path))
        patch_torch_operators()

        config = NNCFConfig.from_dict({"input_info": {"sample_size": [1, 3, 32, 32]}, **compression_section})
        compressed_model_proto = load_exported_onnx_version(config, model, tmp_path)

        compressed_model_onnx_node_names = {node.name for node in compressed_model_proto.graph.node}
        for node in original_model_proto.graph.node:
            if not node.name.startswith("Identity_"):
                # Since torch==2.2.0 identity nodes have different indexes
                assert node.name in compressed_model_onnx_node_names
    finally:
        patch_torch_operators()
