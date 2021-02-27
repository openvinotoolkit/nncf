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

from nncf.api.compression import CompressionScheduler
from nncf.api.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.api.composite_compression import CompositeCompressionAlgorithmController
from nncf.api.composite_compression import CompositeCompressionLoss
from nncf.api.composite_compression import CompositeCompressionScheduler
from nncf.compression_method_api import PTCompressionAlgorithmBuilder
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.compression_method_api import PTCompressionLoss
from nncf.hw_config import HWConfigType, HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.nncf_network import NNCFNetwork
from nncf.pruning.base_algo import BasePruningAlgoController
from nncf.initialization import TrainEpochArgs
from nncf.compression_method_api import PTStubCompressionScheduler

ModelType = TypeVar('ModelType')


class PTAccuracyAwareCompressionLoss(CompositeCompressionLoss, PTCompressionLoss):
    def __init__(self):
        super().__init__()
        self._child_losses = torch.nn.ModuleList()

    @property
    def child_losses(self) -> torch.nn.ModuleList:
        return self._child_losses


class PTAccuracyAwareCompressionScheduler(CompositeCompressionScheduler, CompressionScheduler):
    def get_state(self):
        result = {}
        for child_scheduler in self._child_schedulers:
            result.update(child_scheduler.get_state())
        return result

    def load_state(self, state):
        for child_scheduler in self._child_schedulers:
            child_scheduler.load_state(state)


class PTAccuracyAwareCompressionAlgorithmBuilder(
        CompositeCompressionAlgorithmBuilder, PTCompressionAlgorithmBuilder):
    def __init__(self, config: 'NNCFConfig', should_init: bool = True):
        from nncf import NNCFConfig
        from nncf.quantization.structs import QuantizerSetupType
        from nncf.model_creation import get_compression_algorithm

        print(config)
        raise RuntimeError

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
            self._child_builders = [get_compression_algorithm(config)(config, should_init=should_init), ]
            #self._child_builders = [
            #    get_compression_algorithm(compression_config)(compression_config, should_init=should_init), ]
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


class PTAccuracyAwareCompressionAlgorithmController(
    CompositeCompressionAlgorithmController, PTCompressionAlgorithmController):
    def __init__(self, target_model: ModelType, config):
        super().__init__(target_model)
        self._loss = PTAccuracyAwareCompressionLoss()
        self._scheduler = PTAccuracyAwareCompressionScheduler()
        self.config = config

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

    def run_accuracy_aware_training(self):
        train_epoch_args = self.config.get_extra_struct(TrainEpochArgs)
        #self.config.device = train_epoch_args.device
        #self.config.print_freq = 10
        #self.config.multiprocessing_distributed = False
        #self.config.distributed = False

        self.config.full_training_epochs = 3
        self.maximal_accuracy_drop = 1.
        self.sparsity_aa_step = 0.1
        self.new_sparsity_target = None
        self.patience_epochs = 3

        compressed_accuracy = train_epoch_args.eval_fn(self._model, epoch=0, log_epoch=-1)
        acc_drop = train_epoch_args.uncompressed_model_accuracy - compressed_accuracy
        print('ACCURACY DROP {}'.format(acc_drop))

        for epoch in range(self.config.full_training_epochs):
            train_epoch_args.train_epoch_fn(train_epoch_args.config, self._child_ctrls[0],
                                            self._model, train_epoch_args.criterion_fn,
                                            epoch=epoch,
                                            log_epoch=epoch)

            compressed_accuracy = train_epoch_args.eval_fn(self._model,
                                                           epoch=epoch,
                                                           log_epoch=epoch)
            acc_drop = train_epoch_args.uncompressed_model_accuracy - compressed_accuracy
            print('ACCURACY DROP {}'.format(acc_drop))

        epoch_count = 0
        self.step_reduction_factor = 0.5
        self.compression_increasing = None
        log_epoch_count = self.config.full_training_epochs

        while (self.sparsity_aa_step >= 0.05) or (epoch_count < self.patience_epochs):

            if acc_drop > self.maximal_accuracy_drop:
                if self.new_sparsity_target is None:
                    self.new_sparsity_target = self._child_ctrls[0]._scheduler.current_sparsity_level - self.sparsity_aa_step
                    print('Setting new target sparsity {}'.format(self.new_sparsity_target))
                    self._child_ctrls[0]._scheduler = PTStubCompressionScheduler()
                elif epoch_count > self.patience_epochs:
                    if self.compression_increasing is not None and self.compression_increasing:
                        self.sparsity_aa_step *= self.step_reduction_factor
                    self.new_sparsity_target -= self.sparsity_aa_step
                    epoch_count = 0
                    print('Setting new target sparsity {}'.format(self.new_sparsity_target))
                    self.compression_increasing = False
            else:
                if self.new_sparsity_target is None:
                    self.new_sparsity_target = self._child_ctrls[0]._scheduler.current_sparsity_level + self.sparsity_aa_step
                    self._child_ctrls[0]._scheduler = PTStubCompressionScheduler()
                    print('Setting new target sparsity {}'.format(self.new_sparsity_target))
                elif epoch_count > self.patience_epochs:
                    if self.compression_increasing is not None and not self.compression_increasing:
                        self.sparsity_aa_step *= self.step_reduction_factor
                    self.new_sparsity_target += self.sparsity_aa_step
                    epoch_count = 0
                    print('Setting new target sparsity {}'.format(self.new_sparsity_target))
                    self.compression_increasing = True

            self._child_ctrls[0].set_sparsity_level(self.new_sparsity_target)
            train_epoch_args.train_epoch_fn(train_epoch_args.config, self._child_ctrls[0],
                                            self._model, train_epoch_args.criterion_fn,
                                            epoch=epoch_count,
                                            log_epoch=log_epoch_count)
            compressed_accuracy = train_epoch_args.eval_fn(self._model,
                                                           epoch=epoch_count,
                                                           log_epoch=log_epoch_count)
            acc_drop = train_epoch_args.uncompressed_model_accuracy - compressed_accuracy

            epoch_count += 1
            log_epoch_count += 1
            print('ACCURACY DROP {}'.format(acc_drop))
            print('STEP VALUE {}'.format(self.sparsity_aa_step))

        # reset optimizer & lr scheduler
        # train for N epochs
        # A) check accuracy; if higher that acc drop, decrease by step, if lower increase.
        # if direction changed - decrease step
        # B) reset optimizer & lr scheduler
        # C) train for #patience epochs
        # D) repeat A)
