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
from typing import Dict, List

from nncf.common.quantization.quantizer_setup import SingleConfigQuantizerSetup
from nncf.torch.quantization.precision_constraints import HardwareQuantizationConstraints
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitializer
from nncf.torch.quantization.precision_init.base_init import BasePrecisionInitParams
from nncf.torch.structures import QuantizationPrecisionInitArgs


class ManualPrecisionInitParams(BasePrecisionInitParams):
    def __init__(self, user_init_args: QuantizationPrecisionInitArgs = None, bitwidth_per_scope: List[List] = None):
        super().__init__(user_init_args)
        self.bitwidth_per_scope = bitwidth_per_scope

    @classmethod
    def from_config(cls, manual_init_params_dict: Dict):
        return cls(user_init_args=None, bitwidth_per_scope=manual_init_params_dict.get("bitwidth_per_scope", []))


class ManualPrecisionInitializer(BasePrecisionInitializer):
    def __init__(
        self,
        algo: "ExperimentalQuantizationController",  # noqa: F821
        params: ManualPrecisionInitParams,
        hw_precision_constraints: HardwareQuantizationConstraints = None,
    ):
        super().__init__(algo, params, hw_precision_constraints)
        self._bitwidth_per_scope = params.bitwidth_per_scope

    def apply_init(self) -> SingleConfigQuantizerSetup:
        quantizer_setup = self._algo.get_quantizer_setup_for_current_state()
        for pair in self._bitwidth_per_scope:
            bitwidth, scope_name = pair
            is_matched = False
            msg = (
                "Failed to assign bitwidth={} to `{}`,\n"
                "because it is incompatible for the specified target hardware\n"
                "Supported quantization configs: {}"
            )
            for qp_id, qp in quantizer_setup.quantization_points.items():
                if scope_name in str(qp.insertion_point):
                    if self._hw_precision_constraints:
                        q_id = self._algo.setup_to_module_id_translation_dict[qp_id]
                        q_configs = self._hw_precision_constraints.get(q_id)
                        matched_q_configs = list(filter(lambda x: x.num_bits == bitwidth, q_configs))
                        if not matched_q_configs:
                            raise ValueError(msg.format(bitwidth, scope_name, list(map(str, q_configs))))
                        qp.qconfig = matched_q_configs[0]
                    else:
                        qp.qconfig.num_bits = bitwidth
                    is_matched = True
                    break
            if not is_matched:
                raise ValueError(
                    "Could not find a quantization point at scope name `{}`, failed to assign bitwidth {} "
                    "to it".format(scope_name, bitwidth)
                )
        return quantizer_setup
