# Copyright (c) 2024 Intel Corporation
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
import torch
import torch.nn.functional as F

from nncf import CompressWeightsMode
from nncf import SensitivityMetric
from nncf.quantization import compress_weights
from nncf.torch import wrap_model
from nncf.torch.quantization.layers import AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import SymmetricWeightsDecompressor

DATA_BASED_SENSITIVITY_METRICS = (
    SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    SensitivityMetric.MEAN_ACTIVATION_VARIANCE,
    SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE,
)

ALL_SENSITIVITY_METRICS = DATA_BASED_SENSITIVITY_METRICS + (SensitivityMetric.WEIGHT_QUANTIZATION_ERROR,)

SUPPORTED_MODES = (CompressWeightsMode.INT8, CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM)
UNSUPPORTED_MODES = (
    CompressWeightsMode.INT4_SYM,
    CompressWeightsMode.INT4_ASYM,
    CompressWeightsMode.NF4,
)


class ShortTransformer(torch.nn.Module):
    def __init__(self, in_features, num_embeddings, share_weights=False):
        super().__init__()
        self.wte = torch.nn.Embedding(num_embeddings, in_features)
        self.linear = torch.nn.Linear(in_features, in_features)
        self.lm_head = torch.nn.Linear(in_features, num_embeddings)

        if share_weights:
            self.lm_head.weight = self.wte.weight

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = self.linear(x)
        res = self.lm_head(x)
        return res


class MatMulModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(size=(300, 300), dtype=torch.float32))

    def forward(self, input):
        return input @ self.w


class FunctionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_w = torch.nn.Parameter(torch.ones(size=(5, 3, 3, 3), dtype=torch.float32))
        self.matmul_w = torch.nn.Parameter(torch.ones(size=(1, 3, 300, 300), dtype=torch.float32))
        self.conv_tr_w = torch.nn.Parameter(torch.rand(size=(5, 4, 3, 3)))
        self.nested_matmul = MatMulModel()

    def forward(self, input_):
        x = input_.to(torch.float32)
        x = x @ self.matmul_w
        x = self.nested_matmul(x)
        x = F.conv2d(x, self.conv_w)
        x = F.conv_transpose2d(x, self.conv_tr_w)
        return x


class ConvolutionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_regular = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3)
        self.conv_depthwise = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, groups=8)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = torch.nn.Linear(in_features=8, out_features=8)

    def forward(self, input_):
        input_ = input_.to(torch.float32)
        x = self.conv_regular(input_)
        x = F.relu(x)
        x.transpose_(2, 3)
        x = self.max_pool2d(x)
        y = self.conv_transpose(x)
        z = F.conv_transpose2d(x, self.conv_transpose.weight)
        x = y + z
        x = self.conv_depthwise(x)
        x = F.conv2d(x, self.conv_depthwise.weight, groups=self.conv_depthwise.groups)
        x += torch.ones_like(x)
        x = self.adaptive_avg_pool(x)
        x = self.linear(x.flatten())
        return x


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM))
def test_compress_weights(mode):
    model = ShortTransformer(5, 10)
    dtype = torch.int8 if mode == CompressWeightsMode.INT8_SYM else torch.uint8

    input_ids = torch.randint(0, 10, (5,))
    wrapped_model = wrap_model(model, example_input=input_ids, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model, mode=mode)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype == dtype:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM))
def test_compress_weights_functional_model(mode):
    model = FunctionalModel()
    decompressor_type = (
        SymmetricWeightsDecompressor if mode == CompressWeightsMode.INT8_SYM else AsymmetricWeightsDecompressor
    )

    input_ids = torch.randint(0, 10, [1, 3, 300, 300])
    wrapped_model = wrap_model(model, example_input=input_ids, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model, mode=mode)

    n_compressed_weights = 0
    for layer in compressed_model.nncf.external_op.values():
        if isinstance(layer, decompressor_type):
            n_compressed_weights += 1
    assert n_compressed_weights == 4


def test_compress_weights_conv():
    model = ConvolutionModel()

    input_ids = torch.randint(0, 10, [1, 3, 300, 300])
    wrapped_model = wrap_model(model, example_input=input_ids, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            n_target_modules += 1
            if module.weight.dtype in [torch.uint8, torch.int8]:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules


@pytest.mark.parametrize("mode", (CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM))
def test_compress_shared_weights(mocker, mode):
    model = ShortTransformer(5, 10, share_weights=True)
    dtype = torch.int8 if mode == CompressWeightsMode.INT8_SYM else torch.uint8

    input_ids = torch.randint(0, 10, (5,))
    wrapped_model = wrap_model(model, example_input=input_ids, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model, mode=mode)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype == dtype:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules
    assert len(compressed_model.nncf.external_op) == 2

    # check that the weight decompressors are called only once
    for val in compressed_model.nncf.external_op.values():
        mocker.spy(val, "forward")

    compressed_model(input_ids)

    for val in compressed_model.nncf.external_op.values():
        assert val.forward.call_count == 1


class EmptyModel(torch.nn.Module):
    def forward(self, input):
        return input


@pytest.mark.parametrize("mode", SUPPORTED_MODES)
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
    ),
)
def test_raise_error_with_unsupported_params_for_int8(mode, params):
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    wrapped_model = wrap_model(dummy_torch_model, example_input=dummy_input, trace_parameters=True)
    with pytest.raises(AttributeError):
        compress_weights(wrapped_model, mode=mode, **params)


@pytest.mark.parametrize("mode", UNSUPPORTED_MODES)
def test_raise_error_with_not_int8(mode):
    dummy_torch_model = EmptyModel()
    dummy_input = torch.Tensor()
    wrapped_model = wrap_model(dummy_torch_model, example_input=dummy_input, trace_parameters=True)
    with pytest.raises(AttributeError):
        compress_weights(wrapped_model, mode=mode)


class DTypeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size=(3, 3), dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)
        x = x @ self.weight
        return x


def test_get_dtype_attribute_of_parameter():
    model = DTypeModel()
    dummy_input = torch.randint(0, 10, [3, 3])
    wrapped_model = wrap_model(model, example_input=dummy_input, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model)
    assert compressed_model.weight.dtype == torch.uint8
    compressed_model(dummy_input)
    assert compressed_model.weight.dtype == torch.uint8


@pytest.mark.parametrize("dtype", ("float16", "float32"))
def test_model_devices_and_precisions(use_cuda, dtype):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("Skipping for CPU-only setups")
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float16 if dtype == "float16" else torch.float32

    model = MatMulModel().to(device)
    if dtype == torch.float16:
        model.half()

    dummy_input = torch.rand((1, 300), dtype=dtype, device=device)
    wrapped_model = wrap_model(model, example_input=dummy_input, trace_parameters=True)
    compressed_model = compress_weights(wrapped_model)
    result = compressed_model(dummy_input)

    # Scale should always be in float16
    assert compressed_model.state_dict()["_nncf.external_op.weights_decompressor_w._scale"].dtype == torch.float16
    # Result should be in the precision of the model
    assert result.dtype == dtype
