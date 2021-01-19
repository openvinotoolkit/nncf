"""
 Copyright (c) 2020 Intel Corporation
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
from copy import deepcopy
from typing import Dict, List

from nncf.module_operations import UpdateWeight

from nncf.nncf_network import ExtraCompressionModuleType
from nncf.quantization.layers import QUANTIZATION_MODULES, BaseQuantizer
from nncf.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.quantization.quantizer_id import QuantizerId, WeightQuantizerId
from nncf.quantization.structs import WeightQuantizerInfo
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.structures import NNCFExtraConfigStruct
from nncf.utils import get_all_modules_by_type


class BasePrecisionInitParams:
    def __init__(self,
                 user_init_args: NNCFExtraConfigStruct = None):
        self.user_init_args = user_init_args


class BasePrecisionInitializer:
    def __init__(self, algo: 'ExperimentalQuantizationController',
                 params: BasePrecisionInitParams,
                 hw_precision_constraints: HardwareQuantizationConstraints = None):
        self._algo = algo
        self._model = self._algo._model  # type: NNCFNetwork
        all_quantizers = algo.all_quantizations
        self._hw_precision_constraints = hw_precision_constraints
        self.original_precisions = {q_id: quantizer.num_bits for q_id, quantizer in all_quantizers.items()}
        self._quantizers_handler = WeightQuantizersHandler(self._model, self._algo.weight_quantizers,
                                                           self._hw_precision_constraints)
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self._weight_quantizations_by_execution_order = self._quantizers_handler. \
            get_weight_quantizers_in_execution_order_per_id()

        self._all_quantizers_per_scope = get_all_modules_by_type(
            self._model.get_compression_modules_by_type(ExtraCompressionModuleType.ACTIVATION_QUANTIZER),
            quantization_types)
        self._all_quantizers_per_scope.update(
            self._quantizers_handler.get_all_weight_quantizers_in_execution_order_per_scope())

    def apply_init(self) -> SingleConfigQuantizerSetup:
        raise NotImplementedError


class WeightQuantizersHandler:
    """
    Defines weight quantizers for precision initialization in the order of execution.
    """
    def is_wq_scope(self, scope: 'Scope') -> bool:
        return scope[-2].calling_module_class_name == UpdateWeight.__name__

    def get_owning_module_scope_from_wq_scope(self, wq_scope: 'Scope') -> 'Scope':
        retval = deepcopy(wq_scope)
        retval.pop()
        retval.pop()
        retval.pop()
        return retval

    def __init__(self, model, weight_quantizers: Dict[WeightQuantizerId, WeightQuantizerInfo],
                 constraints: HardwareQuantizationConstraints):
        self._wq_affected_module_scope_vs_qid_dict = {k.get_scope(): k for k in weight_quantizers.keys()}
        self._quantizer_module_scope_vs_qid_dict = {}  # type: Dict[Scope, WeightQuantizerId]
        self._scopes_of_skipped_weight_quantizers = []
        self._skipped_weight_quantizers = {}  # type: Dict[WeightQuantizerId, BaseQuantizer]
        self._weight_quantizers_in_execution_order_per_scope = OrderedDict()  # type: Dict[Scope, BaseQuantizer]
        self._weight_quantizers_in_execution_order = OrderedDict()  # type: Dict[WeightQuantizerId, BaseQuantizer]

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        weight_module_dict = model.get_nncf_wrapped_model()
        quantizers_in_execution_order_per_scope = get_all_modules_by_type(weight_module_dict,
                                                                          quantization_types)

        for scope, quantizer in quantizers_in_execution_order_per_scope.items():
            if self.is_wq_scope(scope):
                affected_module_scope = self.get_owning_module_scope_from_wq_scope(scope)
                if affected_module_scope in self._wq_affected_module_scope_vs_qid_dict:
                    qid = self._wq_affected_module_scope_vs_qid_dict[affected_module_scope]
                    if len(constraints.get_all_unique_bits(qid)) != 1:
                        self._weight_quantizers_in_execution_order_per_scope[scope] = quantizer
                        self._weight_quantizers_in_execution_order[qid] = quantizer
                    else:
                        self._scopes_of_skipped_weight_quantizers.append(scope)
                        self._skipped_weight_quantizers[qid] = quantizer

    def get_scope_of_skipped_weight_quantizers(self) -> List['Scope']:
        return self._scopes_of_skipped_weight_quantizers

    def get_all_weight_quantizers_in_execution_order_per_scope(self) -> Dict['Scope', BaseQuantizer]:
        return self._weight_quantizers_in_execution_order_per_scope

    def get_weight_quantizers_in_execution_order_per_id(self) -> Dict[WeightQuantizerId, BaseQuantizer]:
        return self._weight_quantizers_in_execution_order

    def get_quantizer_id_by_scope(self, scope: 'Scope') -> QuantizerId:
        affected_module_scope = self.get_owning_module_scope_from_wq_scope(scope)
        return self._wq_affected_module_scope_vs_qid_dict[affected_module_scope]

    def get_skipped_weight_quantizers_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._skipped_weight_quantizers
