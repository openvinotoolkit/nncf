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

import numpy as np

from nncf.composite_compression import PTCompositeCompressionAlgorithmController
from nncf.composite_compression import PTCompositeCompressionAlgorithmBuilder
from nncf.api.composite_compression import CompositeCompressionScheduler
from nncf.compression_method_api import PTCompressionAlgorithmController
from nncf.initialization import TrainEpochArgs
from nncf.compression_method_api import PTStubCompressionScheduler
from nncf.common.utils.logger import logger as nncf_logger
from nncf.pruning.filter_pruning.algo import FilterPruningController
from nncf.sparsity.base_algo import BaseSparsityAlgoController


ModelType = TypeVar('ModelType')


ACCURACY_AWARE_CONTROLLER_TYPES = [BaseSparsityAlgoController, FilterPruningController]


class PTAccuracyAwareCompressionAlgorithmBuilder(PTCompositeCompressionAlgorithmBuilder):
    def __init__(self, config: 'NNCFConfig', should_init: bool = True):
        super().__init__(config, should_init)
        self.accuracy_aware_config = config.get('accuracy_aware_training_config', None)

    def build_controller(self, model: ModelType) -> 'PTAccuracyAwareCompressionAlgorithmController':
        composite_ctrl = PTAccuracyAwareCompressionAlgorithmController(model, self.accuracy_aware_config)
        for builder in self.child_builders:
            composite_ctrl.add(builder.build_controller(model))
        return composite_ctrl


class PTAccuracyAwareCompressionAlgorithmController(PTCompositeCompressionAlgorithmController):
    def __init__(self, target_model: ModelType, accuracy_aware_config):
        super().__init__(target_model)
        self.accuracy_aware_config = accuracy_aware_config

        self.minimal_tolerable_accuracy = self.accuracy_aware_config.get('minimal_tolerable_accuracy')
        self.initial_training_phase_epochs = self.accuracy_aware_config.get('initial_training_phase_epochs')
        self.compression_level_step = self.accuracy_aware_config.get('initial_compression_level_step')
        self.step_reduction_factor = self.accuracy_aware_config.get('compression_level_step_reduction_factor')
        self.minimal_compression_level_step = self.accuracy_aware_config.get('minimal_compression_level_step')
        self.patience_epochs = self.accuracy_aware_config.get('patience_epochs')

        self.accuracy_aware_controller = None
        self.compression_level_target = None
        self.compression_increasing = None
        self.accuracy_bugdet = None
        self.search_epoch_count = None

        self.compression_level_setter_map = {
            BaseSparsityAlgoController: self.set_sparsity_level_to_controller,
            FilterPruningController: self.set_pruning_level_to_controller,
        }

        self.compression_level_getter_map = {
            BaseSparsityAlgoController: self.get_sparsity_level_from_controller,
            FilterPruningController: self.get_pruning_level_from_controller,
        }

        if self.minimal_tolerable_accuracy is None:
            raise RuntimeError('Minimal tolerable accuracy value is not set in the config')

    def run_accuracy_aware_training(self, nncf_config):

        self.train_epoch_args = nncf_config.get_extra_struct(TrainEpochArgs)
        self.accuracy_aware_controller = self.determine_compression_variable_controller(self._child_ctrls)
        if self.accuracy_aware_controller is None:
            raise RuntimeError('No controller defined to vary compression level with')

        optimizers, lr_schedulers = self.train_epoch_args.configure_optimizers_fn()
        self.run_initial_training_phase(optimizers, lr_schedulers)

        self.reset_search_epoch_count()
        log_epoch_count = self.initial_training_phase_epochs
        self.train_epoch_args.config.log_epoch = log_epoch_count

        while (self.compression_level_step >= self.minimal_compression_level_step) or \
            (self.search_epoch_count < self.patience_epochs):

            compression_level_change_sign = np.sign(self.accuracy_bugdet)
            compression_level_changed = self.update_target_compression_level(compression_level_change_sign)

            nncf_logger.info('-' * 40)
            nncf_logger.info('Current target compression level value {}'.format(self.compression_level_target))
            nncf_logger.info('Current accuracy budget value = {}'.format(self.accuracy_bugdet))
            nncf_logger.info('Current compression level step value = {}'.format(self.compression_level_step))
            nncf_logger.info('-' * 40)

            if compression_level_changed:
                optimizers, lr_schedulers = self.train_epoch_args.configure_optimizers_fn()

            self.set_compression_level(self.compression_level_target)

            self.train_epoch_args.train_epoch_fn(self.train_epoch_args.config, self,
                                                 self._model,
                                                 epoch=self.search_epoch_count,
                                                 optimizers=optimizers,
                                                 lr_schedulers=lr_schedulers)

            compressed_model_accuracy = self.train_epoch_args.eval_fn(self._model, self.train_epoch_args.config)
            self.accuracy_bugdet = compressed_model_accuracy - self.minimal_tolerable_accuracy

            self.search_epoch_count += 1
            log_epoch_count += 1
            self.train_epoch_args.config.log_epoch = log_epoch_count

        self.train_epoch_args.on_training_end_fn()

    def run_initial_training_phase(self, optimizers, lr_schedulers):
        for epoch in range(self.initial_training_phase_epochs):
            self.train_epoch_args.config.log_epoch = epoch
            self.train_epoch_args.train_epoch_fn(self.train_epoch_args.config, self,
                                                 self._model,
                                                 epoch=epoch,
                                                 optimizers=optimizers,
                                                 lr_schedulers=lr_schedulers)

            compressed_model_accuracy = self.train_epoch_args.eval_fn(self._model, self.train_epoch_args.config)
        self.accuracy_bugdet = compressed_model_accuracy - self.minimal_tolerable_accuracy
        nncf_logger.info('Accuracy budget value after training is {}'.format(self.accuracy_bugdet))

    def disable_compression_schedulers(self):
        self._scheduler = CompositeCompressionScheduler()
        for child_controller in self.child_ctrls:
            child_controller._scheduler = PTStubCompressionScheduler()
            child_controller._scheduler.pruning_target = 0.0
            self._scheduler.add(child_controller._scheduler)

    def update_target_compression_level(self, compression_level_change_sign):
        if self.compression_level_target is None:
            self.compression_level_target = self.get_compression_level() + \
                compression_level_change_sign * self.compression_level_step
            self.disable_compression_schedulers()
            return True
        elif self.search_epoch_count > self.patience_epochs:
            if self.compression_increasing is not None and (self.compression_increasing and compression_level_change_sign > 0) or \
                (not self.compression_increasing and compression_level_change_sign < 0):
                self.compression_level_step *= self.step_reduction_factor
            self.compression_level_target += compression_level_change_sign * self.compression_level_step
            self.reset_search_epoch_count()
            self.compression_increasing = (compression_level_change_sign > 0)
            return True
        return False

    def reset_search_epoch_count(self):
        self.search_epoch_count = 0

    def set_compression_level(self, compression_level):
        controller_type = self.accuracy_aware_controller.__class__
        self.compression_level_setter_map[controller_type](self.accuracy_aware_controller, compression_level)

    def get_compression_level(self):
        controller_type = self.accuracy_aware_controller.__class__
        return self.compression_level_getter_map[controller_type](self.accuracy_aware_controller)

    @staticmethod
    def get_sparsity_level_from_controller(controller):
        return controller.sparsity_rate_for_model

    @staticmethod
    def get_pruning_level_from_controller(controller):
        if controller.prune_flops:
            return 1 - controller.current_flops / controller.full_flops
        return controller.pruning_rate

    @staticmethod
    def set_sparsity_level_to_controller(controller, sparsity_level):
        controller.set_sparsity_level(sparsity_level)

    @staticmethod
    def set_pruning_level_to_controller(controller, pruning_level):
        controller.frozen = False
        controller.set_pruning_rate(pruning_level)

    @staticmethod
    def determine_compression_variable_controller(controller_list):
        for controller in controller_list:
            if controller.__class__ in ACCURACY_AWARE_CONTROLLER_TYPES:
                return controller
        return None
