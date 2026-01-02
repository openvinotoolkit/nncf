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

import json

import pytest
import torch

from nncf.common.quantization.structs import QuantizationScheme
from nncf.quantization.algorithms.smooth_quant.torch_backend import SQMultiply
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.graph.transformations.commands import TransformationType
from nncf.torch.graph.transformations.serialization import deserialize_command
from nncf.torch.graph.transformations.serialization import serialize_command
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer


def test_non_supported_command_serialization():
    class NonSupportedCommand(PTTransformationCommand):
        def __init__(self):
            super().__init__(TransformationType.INSERT, None)

    command = NonSupportedCommand()

    with pytest.raises(RuntimeError):
        serialize_command(command)

    serialized_command = {"type": NonSupportedCommand.__name__}
    with pytest.raises(RuntimeError):
        deserialize_command(serialized_command)


@pytest.mark.parametrize("quantizer_class", (SymmetricQuantizer, AsymmetricQuantizer))
def test_quantizer_serialization(quantizer_class: BaseQuantizer):
    scale_shape = [1, 3, 1, 1]
    ref_qspec = PTQuantizerSpec(
        num_bits=4,
        mode=QuantizationScheme.ASYMMETRIC,
        signedness_to_force=False,
        narrow_range=True,
        half_range=False,
        scale_shape=scale_shape,
        logarithm_scale=False,
        is_quantized_on_export=False,
        compression_lr_multiplier=2.0,
    )
    quantizer = quantizer_class(ref_qspec)
    if isinstance(quantizer, SymmetricQuantizer):
        quantizer.scale = torch.nn.Parameter(torch.fill(torch.empty(scale_shape), 5))
    elif isinstance(quantizer, AsymmetricQuantizer):
        quantizer.input_low = torch.nn.Parameter(torch.fill(torch.empty(scale_shape), 6))
        quantizer.input_range = torch.nn.Parameter(torch.fill(torch.empty(scale_shape), 7))

    state_dict = quantizer.state_dict()

    state = quantizer.get_config()
    json_state = json.dumps(state)
    state = json.loads(json_state)

    recovered_quantizer = quantizer_class.from_config(state)
    recovered_quantizer.load_state_dict(state_dict)

    assert recovered_quantizer._qspec == ref_qspec

    assert torch.all(quantizer._num_bits == recovered_quantizer._num_bits)
    assert torch.all(quantizer.enabled == recovered_quantizer.enabled)
    if isinstance(quantizer, SymmetricQuantizer):
        assert torch.all(quantizer.signed_tensor == recovered_quantizer.signed_tensor)
        assert torch.all(quantizer.scale == recovered_quantizer.scale)
    elif isinstance(quantizer, AsymmetricQuantizer):
        assert torch.all(quantizer.input_low == recovered_quantizer.input_low)
        assert torch.all(quantizer.input_range == recovered_quantizer.input_range)
    else:
        raise RuntimeError()


def test_sq_multiply_serialization():
    tensor_shape = [1, 3, 5]
    tensor_value = torch.fill(torch.empty(tensor_shape, dtype=torch.float16), 5)
    sq_multiply = SQMultiply(tensor_shape)
    sq_multiply.scale = tensor_value
    state_dict = sq_multiply.state_dict()

    state = sq_multiply.get_config()
    json_state = json.dumps(state)
    state = json.loads(json_state)

    recovered_sq_multiply = SQMultiply.from_config(state)
    recovered_sq_multiply.load_state_dict(state_dict)

    assert torch.all(sq_multiply.scale == recovered_sq_multiply.scale)
    assert sq_multiply.scale.shape == recovered_sq_multiply.scale.shape
