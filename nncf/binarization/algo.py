"""
 Copyright (c) 2019-2020 Intel Corporation
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
from collections import OrderedDict
from typing import List, Callable

import torch
from texttable import Texttable
from torch import nn

from nncf.algo_selector import COMPRESSION_ALGORITHMS
from nncf.binarization.layers import BINARIZATION_MODULES, BinarizationMode, WeightBinarizer, ActivationBinarizer, \
    ActivationBinarizationScaleThreshold, BaseBinarizer
from nncf.compression_method_api import CompressionAlgorithmBuilder, CompressionAlgorithmController, CompressionLevel
from nncf.config import NNCFConfig
from nncf.layers import NNCFConv2d
from nncf.module_operations import UpdateWeight, UpdateInputs
from nncf.nncf_logger import logger as nncf_logger
from nncf.nncf_network import InsertionCommand, InsertionPoint, InsertionType, OperationPriority
from nncf.nncf_network import NNCFNetwork
from nncf.quantization.algo import QuantizationControllerBase
from nncf.quantization.schedulers import QUANTIZATION_SCHEDULERS


@COMPRESSION_ALGORITHMS.register('binarization')
class BinarizationBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self.mode = self.config.get('mode', BinarizationMode.XNOR)

    def _apply_to(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        return self._binarize_weights_and_module_inputs(target_model)

    def __create_binarize_module(self):
        return BINARIZATION_MODULES.get(self.mode)()

    def _binarize_weights_and_module_inputs(self, target_model: NNCFNetwork) -> List[InsertionCommand]:
        device = next(target_model.parameters()).device
        modules = target_model.get_nncf_modules_by_module_names(self.compressed_nncf_module_names)

        insertion_commands = []
        for scope, module in modules.items():
            scope_str = str(scope)

            if not self._should_consider_scope(scope_str):
                nncf_logger.info("Ignored adding binarizers in scope: {}".format(scope_str))
                continue

            if isinstance(module, torch.nn.modules.Conv2d):
                nncf_logger.info("Adding Weight binarizer in scope: {}".format(scope_str))
                op_weights = UpdateWeight(
                    self.__create_binarize_module()
                ).to(device)

                nncf_logger.info("Adding Activation binarizer in scope: {}".format(scope_str))
                op_inputs = UpdateInputs(ActivationBinarizationScaleThreshold(module.weight.shape)).to(device)

                ip = InsertionPoint(InsertionType.NNCF_MODULE_PRE_OP,
                                    module_scope=scope)
                insertion_commands.append(InsertionCommand(ip, op_weights, OperationPriority.QUANTIZATION_PRIORITY))

                insertion_commands.append(InsertionCommand(ip, op_inputs, OperationPriority.QUANTIZATION_PRIORITY))
        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> CompressionAlgorithmController:
        return BinarizationController(target_model, self.config)


class BinarizationController(QuantizationControllerBase):
    def __init__(self, target_model: NNCFNetwork, config: NNCFConfig):
        super().__init__(target_model)

        scheduler_cls = QUANTIZATION_SCHEDULERS.get("staged")
        self._scheduler = scheduler_cls(self, config.get("params", {}))
        from nncf.utils import is_main_process
        if is_main_process():
            self._compute_and_display_flops_binarization_rate()

    def _set_binarization_status(self, condition_fn: Callable[[BaseBinarizer], bool],
                                 apply_fn: Callable[[BaseBinarizer], None]):
        if self._model is not None:
            for _, m in self._model.named_modules():
                if condition_fn(m):
                    apply_fn(m)

    def enable_activation_quantization(self):
        self._set_binarization_status(lambda x: isinstance(x, ActivationBinarizer), lambda x: x.enable())

    def enable_weight_quantization(self):
        self._set_binarization_status(lambda x: isinstance(x, WeightBinarizer), lambda x: x.enable())

    def disable_activation_quantization(self):
        self._set_binarization_status(lambda x: isinstance(x, ActivationBinarizer), lambda x: x.disable())

    def disable_weight_quantization(self):
        self._set_binarization_status(lambda x: isinstance(x, WeightBinarizer), lambda x: x.disable())

    def init_range(self):
        pass

    def compression_level(self) -> CompressionLevel:
        return self.scheduler.compression_level()

    def _compute_and_display_flops_binarization_rate(self):
        net = self._model
        weight_list = {}
        state_dict = net.state_dict()
        for n, v in state_dict.items():
            weight_list[n] = v.clone()

        ops_dict = OrderedDict()

        def get_hook(name):
            def compute_flops_hook(self, input_, output):
                name_type = str(type(self).__name__)
                if isinstance(self, (nn.Conv2d, nn.ConvTranspose2d)):
                    ks = self.weight.data.shape
                    ops_count = ks[0] * ks[1] * ks[2] * ks[3] * output.shape[3] * output.shape[2]
                elif isinstance(self, nn.Linear):
                    ops_count = input_[0].shape[1] * output.shape[1]
                else:
                    return
                ops_dict[name] = (name_type, ops_count, isinstance(self, NNCFConv2d))

            return compute_flops_hook

        hook_list = [m.register_forward_hook(get_hook(n)) for n, m in net.named_modules()]

        net.do_dummy_forward(force_eval=True)

        for h in hook_list:
            h.remove()

        # restore all parameters that can be corrupted due forward pass
        for n, v in state_dict.items():
            state_dict[n].data.copy_(weight_list[n].data)

        ops_bin = 0
        ops_total = 0

        for layer_name, (layer_type, ops, is_binarized) in ops_dict.items():
            ops_total += ops
            if is_binarized:
                ops_bin += ops

        table = Texttable()
        header = ["Layer name", "Layer type", "Binarized", "MAC count", "MAC share"]
        table_data = [header]

        for layer_name, (layer_type, ops, is_binarized) in ops_dict.items():
            drow = {h: 0 for h in header}
            drow["Layer name"] = layer_name
            drow["Layer type"] = layer_type
            drow["Binarized"] = 'Y' if is_binarized else 'N'
            drow["MAC count"] = "{:.3f}G".format(ops * 1e-9)
            drow["MAC share"] = "{:2.1f}%".format(ops / ops_total * 100)
            row = [drow[h] for h in header]
            table_data.append(row)

        table.add_rows(table_data)
        nncf_logger.info(table.draw())
        nncf_logger.info("Total binarized MAC share: {:.1f}%".format(ops_bin / ops_total * 100))
