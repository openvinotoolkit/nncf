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
from copy import deepcopy

import pytest
import torch

import nncf
from nncf.common.factory import ModelTransformerFactory
from nncf.common.quantization.structs import QuantizationScheme
from nncf.quantization.algorithms.smooth_quant.torch_backend import SQMultiply
from nncf.torch import wrap_model
from nncf.torch.graph.transformations.commands import PTTransformationCommand
from nncf.torch.graph.transformations.commands import TransformationType
from nncf.torch.graph.transformations.serialization import deserialize_command
from nncf.torch.graph.transformations.serialization import deserialize_transformations
from nncf.torch.graph.transformations.serialization import serialize_command
from nncf.torch.graph.transformations.serialization import serialize_transformations
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.pruning.filter_pruning.layers import FilterPruningMask
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.sparsity.layers import BinaryMask
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from tests.torch.helpers import DummyOpWithState
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import commands_are_equal
from tests.torch.nncf_network.helpers import AVAILABLE_TARGET_TYPES
from tests.torch.nncf_network.helpers import InsertionCommandBuilder


@pytest.mark.parametrize("target_type", AVAILABLE_TARGET_TYPES)
@pytest.mark.parametrize("command_builder", InsertionCommandBuilder(TwoConvTestModel).get_command_builders())
@pytest.mark.parametrize("priority", InsertionCommandBuilder.PRIORITIES)
def test_serialize_load_command(target_type, command_builder, priority):
    dummy_op_state = "DUMMY_OP_STATE"
    op_unique_name = "UNIQUE_NAME"
    # The only difference for trace_parameters param in this test is taget nodes names
    command = InsertionCommandBuilder(TwoConvTestModel).create_one_command(
        command_builder[0], target_type, priority, dummy_op_state, trace_parameters=False, op_unique_name=op_unique_name
    )

    serialized_command = serialize_command(command)

    # Check serialized transformation are json compatible
    j_str = json.dumps(serialized_command)
    serialized_command = json.loads(j_str)

    recovered_command = deserialize_command(serialized_command)
    _check_commands_after_serialization(command, recovered_command, dummy_op_state)


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


def test_serialize_transformations():
    dummy_op_state = "DUMMY_OP_STATE"
    # The only difference for trace_parameters param in this test is taget nodes names
    layout = InsertionCommandBuilder(TwoConvTestModel).get_all_available_commands(
        dummy_op_state=dummy_op_state, trace_parameters=False
    )

    serialized_transformations = serialize_transformations(layout)

    # Check serialized transformation are json compatible
    j_str = json.dumps(serialized_transformations)
    serialized_transformations = json.loads(j_str)

    recovered_layout = deserialize_transformations(serialized_transformations)
    assert len(layout.transformations) == len(recovered_layout.transformations)
    # Can zip layouts because the order should not be altered
    for command, recovered_command in zip(layout.transformations, recovered_layout.transformations):
        _check_commands_after_serialization(command, recovered_command, dummy_op_state)


@pytest.mark.parametrize("model_cls", InsertionCommandBuilder.AVAILABLE_MODELS)
@pytest.mark.parametrize("trace_parameters", (False, True))
def test_get_apply_serialization_from_a_model(model_cls, trace_parameters):
    dummy_op_state = "DUMMY_OP_STATE"
    layout = InsertionCommandBuilder(model_cls).get_all_available_commands(
        dummy_op_state, trace_parameters, skip_model_transformer_unsupported=True
    )
    model = model_cls()
    example_input = torch.ones((1, 1, 4, 4))
    nncf_model = wrap_model(deepcopy(model), example_input=example_input, trace_parameters=trace_parameters)
    modified_model = ModelTransformerFactory.create(nncf_model).transform(layout)

    serialized_transformations = modified_model.nncf.get_config()

    # Check serialized transformation are json compatible
    j_str = json.dumps(serialized_transformations)
    serialized_transformations = json.loads(j_str)

    recovered_model = nncf.torch.load_from_config(model, serialized_transformations, example_input)

    assert modified_model.state_dict().keys() == recovered_model.state_dict().keys()
    if not trace_parameters:
        _check_pre_post_ops(modified_model, recovered_model)

    context = modified_model.nncf._compressed_context
    recovered_context = recovered_model.nncf._compressed_context
    for hooks_attr in ["_pre_hooks", "_post_hooks"]:
        container = getattr(context, hooks_attr)
        recovered_container = getattr(recovered_context, hooks_attr)
        assert len(container) == len(recovered_container)
        for op_address, hooks in container.items():
            recovered_hooks = recovered_container[op_address]
            for k, hook in hooks.items():
                recovered_hook = recovered_hooks[k]
                _check_hook_are_equal(hook, recovered_hook)

    for attr_name in ["external_quantizers", "external_op"]:
        container = getattr(modified_model.nncf, attr_name)
        recovered_container = getattr(recovered_model.nncf, attr_name)
        assert len(container) == len(recovered_container)
        for k, module in container.items():
            recovered_module = recovered_container[k]
            _check_hook_are_equal(module, recovered_module)


def _check_pre_post_ops(modified_model, recovered_model):
    for conv, recovered_conv in zip(modified_model.features, recovered_model.features):
        for hooks_attr in ["pre_ops", "post_ops"]:
            hooks = getattr(conv[0], hooks_attr)
            recovered_hooks = getattr(recovered_conv[0], hooks_attr)
            assert len(hooks) == len(recovered_hooks)
            for k, hook in hooks.items():
                recovered_hook = recovered_hooks[k]
                if isinstance(hook, UpdateWeight):
                    assert isinstance(recovered_hook, UpdateWeight)
                    hook = hook.op
                    recovered_hook = recovered_hook.op
                _check_hook_are_equal(hook, recovered_hook)


def _check_hook_are_equal(hook, recovered_hook):
    assert type(hook) == type(recovered_hook)
    if isinstance(hook, DummyOpWithState):
        assert hook.get_config() == recovered_hook.get_config()
        return
    # Hook is external op call hook then
    assert hook._storage_name == recovered_hook._storage_name
    assert hook._storage_key == recovered_hook._storage_key


def _check_commands_after_serialization(command, recovered_command, dummy_op_state=None):
    commands_are_equal(recovered_command, command, check_fn_ref=False)
    assert isinstance(command.fn, DummyOpWithState)
    assert command.fn.get_config() == recovered_command.fn.get_config()
    if dummy_op_state is not None:
        assert command.fn.get_config() == dummy_op_state


@pytest.mark.parametrize("size", (4, [3, 4]))
def test_pruning_mask_serialization(size):
    node_name = "dummy_node_name"
    dim = 2
    mask = FilterPruningMask(size=size, node_name=node_name, dim=dim)
    mask.binary_filter_pruning_mask = torch.fill(torch.empty(size), 5)
    state_dict = mask.state_dict()

    state = mask.get_config()
    json_state = json.dumps(state)
    state = json.loads(json_state)

    recovered_mask = FilterPruningMask.from_config(state)
    recovered_mask.load_state_dict(state_dict)

    ref_size = size if isinstance(size, list) else [size]
    assert list(recovered_mask.binary_filter_pruning_mask.size()) == ref_size
    assert recovered_mask.node_name == node_name
    assert recovered_mask.mask_applying_dim == dim

    assert torch.all(mask.binary_filter_pruning_mask == recovered_mask.binary_filter_pruning_mask)


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


def test_sparsity_binary_mask_serialization():
    ref_shape = [4, 2, 1, 3]
    mask = BinaryMask(ref_shape)
    mask.binary_mask = torch.zeros(ref_shape)
    state_dict = mask.state_dict()

    state = mask.get_config()
    json_state = json.dumps(state)
    state = json.loads(json_state)

    recovered_mask = BinaryMask.from_config(state)
    recovered_mask.load_state_dict(state_dict)

    assert list(recovered_mask.binary_mask.shape) == ref_shape
    assert torch.all(mask.binary_mask == recovered_mask.binary_mask)


def test_rb_sparsity_mask_serialization():
    ref_weights_shape = [3, 2, 4, 1]
    ref_frozen = False
    ref_compression_lr_multiplier = 2.0
    ref_eps = 0.3
    mask = RBSparsifyingWeight(
        weight_shape=ref_weights_shape,
        frozen=ref_frozen,
        compression_lr_multiplier=ref_compression_lr_multiplier,
        eps=ref_eps,
    )
    mask.binary_mask = torch.zeros(ref_weights_shape)
    mask.mask = torch.fill(torch.empty(ref_weights_shape), 5)
    state_dict = mask.state_dict()

    state = mask.get_config()
    json_state = json.dumps(state)
    state = json.loads(json_state)

    recovered_mask = RBSparsifyingWeight.from_config(state)
    recovered_mask.load_state_dict(state_dict)

    assert list(recovered_mask.mask.shape) == ref_weights_shape
    assert recovered_mask.frozen == ref_frozen
    assert recovered_mask._compression_lr_multiplier == ref_compression_lr_multiplier
    assert recovered_mask.eps == ref_eps

    assert torch.all(mask.mask == recovered_mask.mask)
    assert torch.all(mask.binary_mask == recovered_mask.binary_mask)
    assert torch.all(mask.uniform == recovered_mask.uniform)


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
