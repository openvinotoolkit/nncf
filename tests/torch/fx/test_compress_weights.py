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

from typing import Dict

import pytest
import torch

import nncf
from nncf import BackupMode
from nncf import CompressWeightsMode
from nncf.common.factory import NNCFGraphFactory
from nncf.data.dataset import Dataset
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.quantization import compress_weights
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
from tests.torch.fx.helpers import get_torch_fx_model
from tests.torch.ptq.test_weights_compression import ALL_SENSITIVITY_METRICS
from tests.torch.ptq.test_weights_compression import DATA_BASED_SENSITIVITY_METRICS
from tests.torch.ptq.test_weights_compression import INT4_MODES
from tests.torch.ptq.test_weights_compression import INT8_MODES
from tests.torch.ptq.test_weights_compression import SUPPORTED_MODES
from tests.torch.ptq.test_weights_compression import UNSUPPORTED_MODES
from tests.torch.ptq.test_weights_compression import ConvolutionModel
from tests.torch.ptq.test_weights_compression import DTypeModel
from tests.torch.ptq.test_weights_compression import EmptyModel
from tests.torch.ptq.test_weights_compression import FunctionalModel
from tests.torch.ptq.test_weights_compression import MatMulModel
from tests.torch.test_models.synthetic import ShortTransformer


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / 1024**2

    return model_size_mb


def get_compressed_modules_weights(
    compressed_model: torch.fx.GraphModule, dtype: torch.dtype, compressed_node_weight_port: Dict[str, int]
):
    n_target_modules = 0
    n_compressed_weights = 0

    for node in compressed_model.graph.nodes:
        if node.op == "call_function" and hasattr(node.target, "overloadpacket"):
            node_type = str(node.target.overloadpacket).split(".")[1]
            if node_type in compressed_node_weight_port:
                n_target_modules += 1
                weight_port_id = compressed_node_weight_port[node_type]
                weight_decompressor_node = node.all_input_nodes[weight_port_id]
                if weight_decompressor_node.all_input_nodes:
                    compressed_weight_node = weight_decompressor_node.all_input_nodes[0]
                    weight = get_tensor_constant_from_node(compressed_weight_node, compressed_model).data
                    if weight.dtype == dtype:
                        n_compressed_weights += 1

    return n_target_modules, n_compressed_weights


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_compress_weights(mode):
    model = ShortTransformer(8, 16)
    input_ids = torch.randint(0, 10, (8,))
    exported_model = get_torch_fx_model(model, input_ids)
    kwargs = {}
    if mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
        kwargs["group_size"] = 4
    compressed_model = compress_weights(exported_model, mode=mode, **kwargs)
    dtype = torch.int8 if mode == CompressWeightsMode.INT8_SYM else torch.uint8
    n_compressed_weights = 0
    n_target_modules = 0
    compressed_node_weight_port = {"linear": 1, "embedding": 0}

    n_target_modules, n_compressed_weights = get_compressed_modules_weights(
        compressed_model, dtype, compressed_node_weight_port
    )
    assert n_target_modules == n_compressed_weights


@pytest.mark.parametrize("mode", INT8_MODES)
def test_compress_weights_graph_edge(mode):
    model = ShortTransformer(5, 10)
    input_ids = torch.randint(0, 10, (5,))
    exported_model = get_torch_fx_model(model, input_ids)
    compressed_model = compress_weights(exported_model, mode=mode)
    nncf_graph = NNCFGraphFactory.create(compressed_model)
    for node in nncf_graph.get_all_nodes():
        if "weights_decompressor" in node.node_name and node.node_type == "call_module":
            decompressor_node_edge = nncf_graph.get_input_edges(node)[0]
            decompressor_constant_edge = nncf_graph.get_edge(node, nncf_graph.get_next_nodes(node)[0])
            assert decompressor_node_edge.tensor_shape == decompressor_constant_edge.tensor_shape


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_compress_weights_shared_weights(mocker, mode):
    with disable_patching():
        model = ShortTransformer(8, 16, share_weights=True)
        input_ids = torch.randint(0, 10, (8,))
        exported_model = get_torch_fx_model(model, input_ids)
        kwargs = {}
        if mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
            kwargs["group_size"] = 4
        compressed_model = compress_weights(exported_model, mode=mode, **kwargs)
    dtype = torch.int8 if mode == CompressWeightsMode.INT8_SYM else torch.uint8
    n_compressed_weights = 0
    n_target_modules = 0
    compressed_node_weight_port = {"linear": 1, "embedding": 0}

    n_target_modules, n_compressed_weights = get_compressed_modules_weights(
        compressed_model, dtype, compressed_node_weight_port
    )
    assert n_target_modules == n_compressed_weights

    num_decompression_nodes = 0
    spies = []
    for node in compressed_model.graph.nodes:
        if node.op == "call_module" and "decompress" in node.name:
            num_decompression_nodes += 1
            decompressor_module = getattr(compressed_model, node.target)
            spy = mocker.spy(decompressor_module, "forward")
            spies.append(spy)
    assert num_decompression_nodes == 2

    compressed_model(input_ids)

    for spy in spies:
        assert spy.call_count == 1


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_compressed_model_inference(mode):
    torch.manual_seed(42)
    model = ShortTransformer(8, 16, share_weights=True)
    input_ids = torch.randint(0, 10, (8,))
    exported_model = get_torch_fx_model(model, input_ids)
    exported_model_output = exported_model(input_ids)
    kwargs = {}
    if mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
        kwargs["group_size"] = 4
    compressed_model = compress_weights(exported_model, mode=mode, **kwargs)
    compressed_model_outputs = compressed_model(input_ids)

    assert (
        exported_model_output.shape == compressed_model_outputs.shape
    ), "Compressed model output shape is not equal to the model output shape"
    assert torch.all(torch.isclose(exported_model_output, compressed_model_outputs, atol=1)).item()


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_compress_weights_model_size_conv(mode):
    dtype = torch.int8 if mode == CompressWeightsMode.INT8_SYM else torch.uint8
    model = ConvolutionModel()

    input_ids = torch.randint(0, 10, [1, 3, 256, 256])
    exported_model = get_torch_fx_model(model, input_ids)
    model_size = get_model_size(exported_model)
    compressed_model = compress_weights(exported_model, mode=mode)
    compressed_model_size = get_model_size(compressed_model)

    n_compressed_weights = 0
    n_target_modules = 0
    compressed_node_weight_port = {"linear": 1, "conv2d": 1, "conv_transpose2d": 1}

    n_target_modules, n_compressed_weights = get_compressed_modules_weights(
        compressed_model, dtype, compressed_node_weight_port
    )

    assert n_compressed_weights == n_target_modules
    assert compressed_model_size < model_size


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_compress_weights_functional_model(mode):
    model = FunctionalModel()
    decompressor_type = (
        "symmetric" if mode in (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT4_SYM) else "asymmetric"
    )

    input_ids = torch.randint(0, 10, [1, 3, 256, 256])
    exported_model = get_torch_fx_model(model, input_ids)
    compressed_model = compress_weights(exported_model, mode=mode)

    n_compressed_weights = 0

    for node in compressed_model.graph.nodes:
        if decompressor_type in node.name:
            n_compressed_weights += 1
    assert n_compressed_weights == 4


@pytest.mark.parametrize("mode", INT8_MODES)
@pytest.mark.parametrize(
    "params",
    (
        {"ratio": 0.5},
        {"group_size": 64},
        {"all_layers": True},
        {"all_layers": False},
        *({"sensitivity_metric": metric} for metric in ALL_SENSITIVITY_METRICS),
        {"gptq": True},
        {"awq": True},
        {"scale_estimation": True},
        {"lora_correction": True},
        {"dataset": Dataset([1])},
        {"backup_mode": BackupMode.NONE},
        {"backup_mode": BackupMode.INT8_ASYM},
        {"backup_mode": BackupMode.INT8_SYM},
        {"advanced_parameters": AdvancedCompressionParameters(statistics_path="anything")},
    ),
)
def test_raise_error_with_unsupported_params_for_int8(mode, params):
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    exported_model = get_torch_fx_model(dummy_torch_model, dummy_input)
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(exported_model, mode=mode, **params)


@pytest.mark.parametrize("mode", INT4_MODES)
@pytest.mark.parametrize(
    "params",
    (
        *({"sensitivity_metric": metric} for metric in DATA_BASED_SENSITIVITY_METRICS),
        {"gptq": True},
        {"awq": True},
        {"scale_estimation": True},
        {"lora_correction": True},
        {"dataset": Dataset([1])},
    ),
)
def test_raise_error_with_unsupported_params_for_int4(mode, params):
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    exported_model = get_torch_fx_model(dummy_torch_model, dummy_input)
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(exported_model, mode=mode, **params)


@pytest.mark.parametrize("mode", UNSUPPORTED_MODES)
def test_raise_error_with_not_int8(mode):
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    exported_model = get_torch_fx_model(dummy_torch_model, dummy_input)
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(exported_model, mode=mode)


def test_raise_error_for_statistics_caching():
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    exported_model = get_torch_fx_model(dummy_torch_model, dummy_input)
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(exported_model, advanced_parameters=AdvancedCompressionParameters(statistics_path="anything"))


def test_get_dtype_attribute_of_parameter():
    model = DTypeModel()
    dummy_input = torch.randint(0, 10, [3, 3])
    exported_model = get_torch_fx_model(model, dummy_input)
    compressed_model = compress_weights(exported_model)
    assert compressed_model.weight_updated_constant0.dtype == torch.uint8
    compressed_model(dummy_input)
    assert compressed_model.weight_updated_constant0.dtype == torch.uint8


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_model_devices_and_precisions(use_cuda, dtype):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Skipping for CPU-only setups")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if dtype == "float16" else torch.float32

    model = MatMulModel().to(device)
    if dtype == torch.float16:
        model.half()
    dummy_input = torch.rand((1, 256), dtype=dtype, device=device)
    exported_model = get_torch_fx_model(model, dummy_input)
    compressed_model = compress_weights(exported_model)
    result = compressed_model(dummy_input)

    # Scale should always be in float16
    assert compressed_model.state_dict()["asymmetric_weights_decompressor_w._scale"].dtype == torch.float16
    # Result should be in the precision of the model
    assert result.dtype == dtype
