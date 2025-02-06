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

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Union

from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.structs import WeightQuantizerInfo
from nncf.torch.structures import NNCFExtraConfigStruct
from nncf.torch.utils import get_all_modules_by_type


class BasePrecisionInitParams:
    def __init__(self, user_init_args: NNCFExtraConfigStruct = None):
        self.user_init_args = user_init_args


class BasePrecisionInitializer:
    def __init__(
        self,
        algo: "ExperimentalQuantizationController",  # noqa: F821
        params: BasePrecisionInitParams,
        hw_precision_constraints: HardwareQuantizationConstraints = None,
    ):
        self._algo = algo
        self._model: NNCFNetwork = self._algo._model
        all_quantizers = algo.all_quantizations
        self._hw_precision_constraints = hw_precision_constraints
        self.original_precisions = {q_id: quantizer.num_bits for q_id, quantizer in all_quantizers.items()}
        self._quantizers_handler = WeightQuantizersHandler(
            self._model, self._algo.weight_quantizers, self._hw_precision_constraints
        )
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self._weight_quantizations_by_execution_order = (
            self._quantizers_handler.get_weight_quantizers_in_execution_order_per_id()
        )

        self._all_quantizers_per_scope = get_all_modules_by_type(
            self._model.nncf.get_compression_modules_by_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER),
            quantization_types,
        )
        self._all_quantizers_per_scope.update(
            self._quantizers_handler.get_all_weight_quantizers_in_execution_order_per_scope()
        )

    def apply_init(self) -> SingleConfigQuantizerSetup:
        raise NotImplementedError

    @staticmethod
    def get_bitwidth_per_scope(quantizer_setup: SingleConfigQuantizerSetup) -> List[List[Union[int, str]]]:
        scope_vs_bitwidth = {}
        for qp in quantizer_setup.quantization_points.values():
            scope_vs_bitwidth[str(qp.insertion_point)] = qp.qconfig.num_bits
        sorted_scope_vs_bitwidth = OrderedDict(sorted(scope_vs_bitwidth.items(), key=lambda x: x[0]))
        full_bitwidth_per_scope = []
        for scope, bitwidth in sorted_scope_vs_bitwidth.items():
            full_bitwidth_per_scope.append([bitwidth, scope])
        return full_bitwidth_per_scope


class WeightQuantizersHandler:
    """
    Defines weight quantizers for precision initialization in the order of execution.
    """

    def is_wq_scope(self, scope: Scope) -> bool:
        return scope[-2].calling_module_class_name == UpdateWeight.__name__

    @staticmethod
    def get_owning_module_scope_from_wq_scope(wq_scope: Scope) -> Scope:
        retval = deepcopy(wq_scope)
        retval.pop()
        retval.pop()
        retval.pop()
        return retval

    def __init__(
        self,
        model: NNCFNetwork,
        weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
        constraints: HardwareQuantizationConstraints,
    ):
        self._wq_affected_module_node_name_vs_qid_dict = {k.target_node_name: k for k in weight_quantizers}
        self._quantizer_module_scope_vs_qid_dict: Dict[Scope, WeightQuantizerId] = {}
        self._skipped_quantized_weight_node_names = []
        self._skipped_weight_quantizers: Dict[WeightQuantizerId, BaseQuantizer] = {}
        self._weight_quantizers_in_execution_order_per_scope: Dict[Scope, BaseQuantizer] = OrderedDict()
        self._weight_quantizers_in_execution_order: Dict[WeightQuantizerId, BaseQuantizer] = OrderedDict()

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        weight_module_dict = model
        quantizers_in_execution_order_per_scope = get_all_modules_by_type(weight_module_dict, quantization_types)

        for scope, quantizer in quantizers_in_execution_order_per_scope.items():
            if self.is_wq_scope(scope):
                affected_module_scope = self.get_owning_module_scope_from_wq_scope(scope)
                affected_module_node = model.nncf.get_original_graph().get_op_nodes_in_scope(affected_module_scope)[0]
                if affected_module_node.node_name in self._wq_affected_module_node_name_vs_qid_dict:
                    qid = self._wq_affected_module_node_name_vs_qid_dict[affected_module_node.node_name]
                    if len(constraints.get_all_unique_bitwidths(qid)) != 1:
                        self._weight_quantizers_in_execution_order_per_scope[scope] = quantizer
                        self._weight_quantizers_in_execution_order[qid] = quantizer
                    else:
                        self._skipped_quantized_weight_node_names.append(affected_module_node.node_name)
                        self._skipped_weight_quantizers[qid] = quantizer

    def get_skipped_quantized_weight_node_names(self) -> List[NNCFNodeName]:
        return self._skipped_quantized_weight_node_names

    def get_all_weight_quantizers_in_execution_order_per_scope(self) -> Dict[Scope, BaseQuantizer]:
        return self._weight_quantizers_in_execution_order_per_scope

    def get_weight_quantizers_in_execution_order_per_id(self) -> Dict[WeightQuantizerId, BaseQuantizer]:
        return self._weight_quantizers_in_execution_order

    def get_quantizer_id_by_scope(self, scope: Scope) -> QuantizerId:
        affected_module_scope = self.get_owning_module_scope_from_wq_scope(scope)
        return self._wq_affected_module_node_name_vs_qid_dict[affected_module_scope]

    def get_skipped_weight_quantizers_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._skipped_weight_quantizers
