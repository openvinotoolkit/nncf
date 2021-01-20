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

from typing import List, Dict

from nncf.quantization.precision_init.base_init import BasePrecisionInitParams, BasePrecisionInitializer
from nncf.quantization.quantizer_setup import SingleConfigQuantizerSetup

from ..precision_constraints import HardwareQuantizationConstraints
from ...structures import QuantizationPrecisionInitArgs
from ...utils import in_scope_list


from nncf.utils import get_all_modules

class ManualPrecisionInitParams(BasePrecisionInitParams):
    def __init__(self,
                 user_init_args: QuantizationPrecisionInitArgs = None,
                 bitwidth_per_scope: List[List] = None):
        super().__init__(user_init_args)
        self.bitwidth_per_scope = bitwidth_per_scope

    @classmethod
    def from_config(cls,
                    manual_init_params_dict: Dict):
        return cls(user_init_args=None,
                   bitwidth_per_scope=manual_init_params_dict.get("bitwidth_per_scope", [[]]))


class ManualPrecisionInitializer(BasePrecisionInitializer):
    def __init__(self,
                 algo: 'ExperimentalQuantizationController',
                 params: ManualPrecisionInitParams,
                 init_args: QuantizationPrecisionInitArgs = None,
                 hw_precision_constraints: HardwareQuantizationConstraints = None):
        super().__init__(algo, params, init_args)
        self._bitwidth_per_scope = params.bitwidth_per_scope

    def apply_init(self) -> SingleConfigQuantizerSetup:
        for pair in self._bitwidth_per_scope:
            if len(pair) != 2:
                raise ValueError('Invalid format of bitwidth per scope: [int, str] is expected')
            bitwidth = pair[0]
            scope_name = pair[1]
            print(str(scope_name))
            is_matched = False
            for scope, quantizer in self._all_quantizers_per_scope.items():
                print(str(scope))
                if in_scope_list(str(scope), scope_name):
                    quantizer.num_bits = bitwidth
                    is_matched = True
            if not is_matched:
                #print(get_all_modules(self._model).keys())
                raise ValueError(
                    'Invalid scope name `{}`, failed to assign bitwidth {} to it'.format(scope_name, bitwidth))
        return self._algo.get_quantizer_setup_for_current_state()
