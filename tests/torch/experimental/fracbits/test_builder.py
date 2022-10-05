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
import torch
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.fracbits.loss import ModelSizeCompressionLoss

from nncf.torch.compression_method_api import PTCompressionLoss
from nncf.torch.dynamic_graph.scope import Scope, ScopeElement
from nncf.torch.model_creation import create_compression_algorithm_builder
from nncf.torch.module_operations import UpdateInputs, UpdateWeight
from nncf.torch.utils import get_all_modules_by_type

from tests.torch.helpers import create_compressed_model_and_algo_for_test
from nncf.experimental.torch.fracbits.builder import FracBitsQuantizationBuilder
from nncf.experimental.torch.fracbits.quantizer import FracBitsSymmetricQuantizer

#pylint: disable=protected-access


def test_create_builder(config):
    builder = create_compression_algorithm_builder(config)
    assert isinstance(builder, FracBitsQuantizationBuilder)


def test_can_load_quant_algo_with_defaults(config, conv_model):
    quant_model, _ = create_compressed_model_and_algo_for_test(deepcopy(conv_model), config)

    model_conv = get_all_modules_by_type(conv_model, 'Conv2d')
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


def test_quant_loss(config, conv_model):
    _, compression_ctrl = create_compressed_model_and_algo_for_test(conv_model, config)

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


def test_quant_loss_params(config, conv_model):
    _, compression_ctrl = create_compressed_model_and_algo_for_test(conv_model, config)

    loss: ModelSizeCompressionLoss = compression_ctrl.loss
    assert isinstance(loss, ModelSizeCompressionLoss)

    loss_config = config["compression"]["loss"]
    assert isinstance(loss._criteria, torch.nn.L1Loss)
    assert loss._alpha == loss_config["alpha"]
    assert loss._flip_loss == loss_config["flip_loss"]
    assert loss._compression_rate.item() == loss_config["compression_rate"]


def test_e2e_quant_loss(config, conv_model_with_input_output):
    conv_model, x, y = conv_model_with_input_output
    criterion = torch.nn.MSELoss()

    quant_model, compression_ctrl = create_compressed_model_and_algo_for_test(conv_model, config)

    optimizer = torch.optim.SGD(quant_model.parameters(), lr=1e-1)

    for i in range(500):
        optimizer.zero_grad()
        target_loss = criterion(quant_model(x), y)
        comp_loss = compression_ctrl.loss.calculate()

        loss = target_loss + comp_loss
        loss.backward()
        optimizer.step()

        if i == 300:
            compression_ctrl.freeze_bit_widths()

    target_comp_rate = config["compression"]["loss"]["compression_rate"]
    for qinfo in compression_ctrl.weight_quantizers.values():
        q = qinfo.quantizer_module_ref
        assert q.num_bits <= int(8 / target_comp_rate)


def test_statistics(config, conv_model):
    _, ctrl = create_compressed_model_and_algo_for_test(conv_model, config)

    stats: NNCFStatistics = ctrl.statistics()
    assert stats.quantization is not None

    dict_stats = dict(stats)
    assert dict_stats[ctrl.name] is not None
