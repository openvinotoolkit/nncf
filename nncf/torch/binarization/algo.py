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

from texttable import Texttable
from torch import nn

from nncf.torch.algo_selector import COMPRESSION_ALGORITHMS, ZeroCompressionLoss
from nncf.api.compression import CompressionStage
from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.torch.binarization.layers import BINARIZATION_MODULES
from nncf.torch.binarization.layers import BinarizationMode
from nncf.torch.binarization.layers import WeightBinarizer
from nncf.torch.binarization.layers import ActivationBinarizer
from nncf.torch.binarization.layers import ActivationBinarizationScaleThreshold
from nncf.torch.binarization.layers import BaseBinarizer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.statistics import NNCFStatistics
from nncf.torch.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.torch.compression_method_api import PTCompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdateInputs
from nncf.common.utils.logger import logger as nncf_logger
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.algo import QuantizationControllerBase
from nncf.torch.quantization.schedulers import QUANTIZATION_SCHEDULERS


@COMPRESSION_ALGORITHMS.register('binarization')
class BinarizationBuilder(PTCompressionAlgorithmBuilder):
    def __init__(self, config, should_init: bool = True):
        super().__init__(config, should_init)
        self.mode = self.config.get('mode', BinarizationMode.XNOR)

    def _get_transformation_layout(self, target_model: NNCFNetwork) -> PTTransformationLayout:
        layout = PTTransformationLayout()
        commands = self._binarize_weights_and_module_inputs(target_model)
        for command in commands:
            layout.register(command)
        return layout

    def __create_binarize_module(self):
        return BINARIZATION_MODULES.get(self.mode)()

    def _nncf_module_types_to_compress(self) -> List[str]:
        return [NNCFConv2d.__name__, ]

    def _binarize_weights_and_module_inputs(self, target_model: NNCFNetwork) -> List[PTInsertionCommand]:
        device = next(target_model.parameters()).device

        module_nodes = target_model.get_weighted_original_graph_nodes(
            nncf_module_names=self.compressed_nncf_module_names)

        insertion_commands = []
        for module_node in module_nodes:
            if not self._should_consider_scope(module_node.node_name):
                nncf_logger.info("Ignored adding binarizers in scope: {}".format(module_node.node_name))
                continue

            nncf_logger.info("Adding Weight binarizer in scope: {}".format(module_node.node_name))
            op_weights = self.__create_binarize_module().to(device)

            nncf_logger.info("Adding Activation binarizer in scope: {}".format(module_node.node_name))
            compression_lr_multiplier = self.config.get("compression_lr_multiplier", None)
            op_inputs = UpdateInputs(ActivationBinarizationScaleThreshold(
                module_node.layer_attributes.get_weight_shape(),
                compression_lr_multiplier=compression_lr_multiplier
            )).to(device)

            ip_w = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS,
                                 target_node_name=module_node.node_name)
            insertion_commands.append(PTInsertionCommand(ip_w, op_weights,
                                                         TransformationPriority.QUANTIZATION_PRIORITY))

            ip_i = PTTargetPoint(TargetType.PRE_LAYER_OPERATION,
                                 target_node_name=module_node.node_name, input_port_id=0)
            insertion_commands.append(PTInsertionCommand(ip_i, op_inputs,
                                                         TransformationPriority.QUANTIZATION_PRIORITY))
        return insertion_commands

    def build_controller(self, target_model: NNCFNetwork) -> PTCompressionAlgorithmController:
        return BinarizationController(target_model, self.config)

    def initialize(self, model: NNCFNetwork) -> None:
        pass


class BinarizationController(QuantizationControllerBase):
    def __init__(self, target_model: NNCFNetwork, config: NNCFConfig):
        super().__init__(target_model)

        self._loss = ZeroCompressionLoss(next(target_model.parameters()).device)
        scheduler_cls = QUANTIZATION_SCHEDULERS.get("staged")
        self._scheduler = scheduler_cls(self, config.get("params", {}))
        from nncf.torch.utils import is_main_process
        if is_main_process():
            self._compute_and_display_flops_binarization_rate()

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

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

    def compression_stage(self) -> CompressionStage:
        return self.scheduler.compression_stage()

    def statistics(self, quickly_collected_only: bool = False) -> NNCFStatistics:
        return NNCFStatistics()

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
