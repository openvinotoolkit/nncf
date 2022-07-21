"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from copy import deepcopy
import pytest
import torch

from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.dynamic_graph.scope import Scope, ScopeElement
from nncf.torch.model_creation import create_compression_algorithm_builder
from nncf.torch.module_operations import UpdateInputs, UpdateWeight
from nncf.torch.utils import get_all_modules_by_type

from tests.torch.helpers import BasicConvTestModel, create_compressed_model_and_algo_for_test, register_bn_adaptation_init_args
from tests.torch.quantization.test_quantization_helpers import get_empty_config
from nncf.experimental.torch.fracbits.builder import FracBitsQuantizationBuilder
from nncf.experimental.torch.fracbits.quantizer import FracBitsSymmetricQuantizer

#pylint: disable=redefined-outer-name


@pytest.fixture
def config(model_size: int = 4):
    config = get_empty_config(model_size)

    config["compression"] = {
        "algorithm": "fracbits_quantization",
        "initializer": {
            "range": {
                "num_init_samples": 0
            }
        },
        "loss": {
            "type": "model_size",
            "compression_rate": 1.5,
            "criteria": "L1"
        }
    }
    register_bn_adaptation_init_args(config)
    return config


@pytest.fixture
def model():
    return BasicConvTestModel()


def test_create_builder(config):
    builder = create_compression_algorithm_builder(config)
    assert isinstance(builder, FracBitsQuantizationBuilder)


def test_can_load_quant_algo__with_defaults(config, model):
    quant_model, _ = create_compressed_model_and_algo_for_test(
        deepcopy(model), config)

    model_conv = get_all_modules_by_type(model, 'Conv2d')
    quant_model_conv = get_all_modules_by_type(
        quant_model.get_nncf_wrapped_model(), 'NNCFConv2d')
    assert len(model_conv) == len(quant_model_conv)

    for module_scope, _ in model_conv.items():
        true_quant_scope: Scope = deepcopy(module_scope)
        true_quant_scope.pop()
        true_quant_scope.push(ScopeElement('NNCFConv2d', 'conv'))
        assert true_quant_scope in quant_model_conv.keys()

        store = []
        for op in quant_model_conv[true_quant_scope].pre_ops.values():
            if isinstance(op, (UpdateInputs, UpdateWeight)) and isinstance(op.operand, FracBitsSymmetricQuantizer):
                assert op.__class__.__name__ not in store
                store.append(op.__class__.__name__)
        assert UpdateWeight.__name__ in store


def test_quant_loss(config, model):
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    loss = compression_ctrl.loss
    assert isinstance(loss, PTCompressionLoss)

    loss_value = compression_ctrl.loss.calculate()
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.grad_fn is not None

    # Check whether bit_width gradient is not None
    loss_value.backward()
    for qinfo in compression_ctrl.weight_quantizers.values():
        q = qinfo.quantizer_module_ref
        assert q._num_bits.grad.data is not None
