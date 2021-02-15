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

from typing import TypeVar

import torch.nn
from copy import deepcopy

from nncf.api.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.api.composite_compression import CompositeCompressionAlgorithmController
from nncf.api.composite_compression import CompositeCompressionLoss
from nncf.api.composite_compression import CompositeCompressionScheduler
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.compression_method_api import PTCompressionLoss
from nncf.compression_method_api import PTCompressionScheduler
from nncf.hw_config import HWConfigType, HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoController

ModelType = TypeVar('ModelType')


class PTCompositeCompressionLoss(CompositeCompressionLoss, PTCompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self) -> torch.nn.ModuleList:
        return self._child_losses


class PTCompositeCompressionScheduler(CompositeCompressionScheduler, PTCompressionScheduler):
    def state_dict(self):
        result = {}
        for child_scheduler in self._child_schedulers:
            result.update(child_scheduler.state_dict())
        return result

    def load_state_dict(self, state_dict):
        for child_scheduler in self._child_schedulers:
            child_scheduler.load_state_dict(state_dict)


class PTCompositeCompressionAlgorithmBuilder(
        CompositeCompressionAlgorithmBuilder, PTCompressionAlgorithmBuilder):
    def __init__(self, config: 'NNCFConfig', should_init: bool = True):
        from nncf import NNCFConfig
        from nncf.quantization.structs import QuantizerSetupType
        from nncf.model_creation import get_compression_algorithm

        super().__init__(config, should_init)

        compression_config_json_section = config.get('compression', {})
        compression_config_json_section = deepcopy(compression_config_json_section)

        hw_config_type = None
        quantizer_setup_type_str = config.get("quantizer_setup_type", "propagation_based")
        quantizer_setup_type = QuantizerSetupType.from_str(quantizer_setup_type_str)
        if quantizer_setup_type == QuantizerSetupType.PROPAGATION_BASED:
            target_device = config.get("target_device", "ANY")
            if target_device != 'TRIAL':
                hw_config_type = HWConfigType.from_str(HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device])

        if isinstance(compression_config_json_section, dict):
            compression_config = NNCFConfig(compression_config_json_section)
            compression_config.register_extra_structs(config.get_all_extra_structs_for_copy())
            compression_config["hw_config_type"] = hw_config_type
            compression_config['quantizer_setup_type'] = quantizer_setup_type
            self._child_builders = [
                get_compression_algorithm(compression_config)(compression_config, should_init=should_init), ]
        else:
            for algo_config in compression_config_json_section:
                algo_config = NNCFConfig(algo_config)
                algo_config.register_extra_structs(config.get_all_extra_structs_for_copy())
                algo_config["hw_config_type"] = hw_config_type
                algo_config['quantizer_setup_type'] = quantizer_setup_type
                self._child_builders.append(
                    get_compression_algorithm(algo_config)(algo_config, should_init=should_init))

    def __bool__(self):
        return bool(self.child_builders)

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        for ctrl in self._child_builders:
            target_model = ctrl.apply_to(target_model)
        return target_model


class PTCompositeCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, PTCompressionAlgorithmController):
    def __init__(self, target_model: ModelType):
        super().__init__(target_model)
        self._loss = PTCompositeCompressionLoss()
        self._scheduler = PTCompositeCompressionScheduler()

    def distributed(self):
        for ctrl in self.child_ctrls:
            ctrl.distributed()

    def prepare_for_export(self):
        if len(self.child_ctrls) > 1 and any(isinstance(x, BasePruningAlgoController) for x in self.child_ctrls):
            # Waiting for the implementation of the related function in OpenVINO
            raise NotImplementedError("Exporting a model that was compressed by filter pruning algorithm "
                                      "in combination with another compression algorithm is not yet supporting")

        for child_ctrl in self.child_ctrls:
            child_ctrl.prepare_for_export()

    def apply_to(self, target_model: NNCFNetwork) -> NNCFNetwork:
        for ctrl in self.child_ctrls:
            target_model = ctrl.apply_to(target_model)
        return target_model
