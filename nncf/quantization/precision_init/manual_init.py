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
from typing import List, Dict

from nncf.nncf_network import NNCFNetwork, CompressionModuleType
from nncf.quantization.layers import QUANTIZATION_MODULES, BaseQuantizer
from ..hw_precision_constraints import HWPrecisionConstraints
from ..quantizer_id import QuantizerId
from ...structures import QuantizationPrecisionInitArgs
from ...utils import in_scope_list, get_all_modules_by_type


class ManualPrecisionInitializer:
    def __init__(self, algo: 'QuantizationController', config: 'NNCFConfig',
                 init_args: QuantizationPrecisionInitArgs = None):
        self._algo = algo
        self._model = self._algo._model  # type: NNCFNetwork
        all_quantizers = algo.all_quantizations
        self._bitwidth_per_scope = config.get('bitwidth_per_scope', {})  # type: List[List]
        self._hw_precision_constraints = algo._hw_precision_constraints
        self.original_precisions = {q_id: quantizer.num_bits for q_id, quantizer in all_quantizers.items()}
        self._quantizers_handler = WeightQuantizersHandler(self._model, all_quantizers,
                                                           self._hw_precision_constraints)

        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        self._weight_quantizations_by_execution_order = self._quantizers_handler. \
            get_weight_quantizers_in_execution_order_per_id()

        self._all_quantizers_per_scope = get_all_modules_by_type(
            self._model.get_compression_modules_by_type(CompressionModuleType.ACTIVATION_QUANTIZER), quantization_types)
        self._all_quantizers_per_scope.update(get_all_modules_by_type(
            self._model.get_compression_modules_by_type(CompressionModuleType.FUNCTION_QUANTIZER), quantization_types))
        self._all_quantizers_per_scope.update(
            self._quantizers_handler.get_all_weight_quantizers_in_execution_order_per_scope())

    def apply_init(self):
        for pair in self._bitwidth_per_scope:
            if len(pair) != 2:
                raise ValueError('Invalid format of bitwidth per scope: [int, str] is expected')
            bitwidth = pair[0]
            scope_name = pair[1]
            is_matched = False
            for scope, quantizer in self._all_quantizers_per_scope.items():
                if in_scope_list(str(scope), scope_name):
                    quantizer.num_bits = bitwidth
                    is_matched = True
            if not is_matched:
                raise ValueError(
                    'Invalid scope name `{}`, failed to assign bitwidth {} to it'.format(scope_name, bitwidth))


class WeightQuantizersHandler:
    """
    Defines weight quantizers for precision initialization in the order of execution.
    """

    def __init__(self, model, all_quantizers: Dict[QuantizerId, BaseQuantizer], constraints: HWPrecisionConstraints):
        self._quantizer_address_to_id_mapping = {id(quantizer): q_id for q_id, quantizer in all_quantizers.items()}
        quantization_types = [class_type.__name__ for class_type in QUANTIZATION_MODULES.registry_dict.values()]
        weight_module_dict = model.get_nncf_wrapped_model()
        self._weight_quantizers_in_execution_order_per_scope = get_all_modules_by_type(weight_module_dict,
                                                                                       quantization_types)
        ordered_weight_quantization_list = []
        self._scopes_of_skipped_weight_quantizers = []
        self._skipped_weight_quantizers = {}
        for scope, quantizer in self._weight_quantizers_in_execution_order_per_scope.items():
            address = id(quantizer)
            if quantizer.is_weights:
                quantizer_id = self._quantizer_address_to_id_mapping[address]
                # no need to init quantizer with single precision constraint
                if len(constraints.get(quantizer_id)) != 1:
                    ordered_weight_quantization_list.append((quantizer_id, quantizer))
                else:
                    self._scopes_of_skipped_weight_quantizers.append(scope)
                    self._skipped_weight_quantizers[quantizer_id] = quantizer
        self._weight_quantizers_in_execution_order = OrderedDict(ordered_weight_quantization_list)

    def get_scope_of_skipped_weight_quantizers(self) -> List['Scope']:
        return self._scopes_of_skipped_weight_quantizers

    def get_all_weight_quantizers_in_execution_order_per_scope(self) -> Dict['Scope', BaseQuantizer]:
        return self._weight_quantizers_in_execution_order_per_scope

    def get_weight_quantizers_in_execution_order_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._weight_quantizers_in_execution_order

    def get_id(self, quantizer: BaseQuantizer) -> QuantizerId:
        address = id(quantizer)
        return self._quantizer_address_to_id_mapping[address]

    def get_skipped_weight_quantizers_per_id(self) -> Dict[QuantizerId, BaseQuantizer]:
        return self._skipped_weight_quantizers
