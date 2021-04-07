"""
 Copyright (c) 2021 Intel Corporation
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
from abc import ABC, abstractmethod
from copy import copy
from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.composite_compression import CompositeCompressionAlgorithmController
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionLevel
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.registry import Registry


ModelType = TypeVar('ModelType')
ACCURACY_AWARE_CONTROLLERS = Registry('accuracy_aware_controllers')


class TrainingRunner(ABC):

    @abstractmethod
    def train_epoch(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def validate(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def dump_checkpoint(self, model: ModelType, compression_controller: CompressionAlgorithmController):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def reset_training(self):
        pass

    @abstractmethod
    def retrieve_original_accuracy(self, model):
        pass


class StubCompressionScheduler(CompressionScheduler):

    def compression_level(self) -> CompressionLevel:
        return CompressionLevel.FULL


# pylint: disable=E1101
def run_accuracy_aware_compressed_training(model,
                                           compression_controller: CompressionAlgorithmController,
                                           runner: TrainingRunner):

    runner.retrieve_original_accuracy(model)

    accuracy_aware_controller = determine_compression_variable_controller(compression_controller)
    if accuracy_aware_controller is None:
        raise RuntimeError('No compression algorithm supported by the accuracy-aware training '
                           'runner was specified in the config')

    run_initial_training_phase(model, accuracy_aware_controller, runner)
    # runner.compression_rate_target = accuracy_aware_controller.compression_rate

    runner.tensorboard_writer.add_scalar('compression/accuracy_aware/target_compression_rate',
                                         accuracy_aware_controller.compression_rate,
                                         runner.cumulative_epoch_count)
    runner.update_training_history(compression_rate=accuracy_aware_controller.compression_rate,
                                   best_metric_value=runner.best_val_metric_value)

    runner.validate_every_n_epochs = 1
    while runner.compression_rate_step >= runner.minimal_compression_rate_step and \
        runner.cumulative_epoch_count < runner.maximal_total_epochs:

        if runner.compression_rate_target is not None:
            runner.update_training_history(compression_rate=copy(runner.compression_rate_target),
                                           best_metric_value=copy(runner.best_val_metric_value))

        was_compression_rate_changed = update_target_compression_rate(accuracy_aware_controller, runner)
        nncf_logger.info('Current target compression rate value: '
                         '{comp_rate:.3f}'.format(comp_rate=runner.compression_rate_target))
        nncf_logger.info('Current accuracy budget value: {acc_budget:.3f}'.format(acc_budget=runner.accuracy_bugdet))
        nncf_logger.info('Current compression rate step value: '
                         '{comp_step:.3f}'.format(comp_step=runner.compression_rate_step))

        if was_compression_rate_changed:
            if runner.compression_rate_target < runner.minimal_compression_rate:
                raise RuntimeError('Cannot produce a compressed model with a specified '
                                   'minimal tolerable accuracy')
            if runner.compression_rate_target > runner.maximal_compression_rate:
                nncf_logger.info('Reached maximal possible compression rate '
                                 '{max_rate}'.format(max_rate=runner.maximal_compression_rate))
                return model

            runner.reset_training()
            accuracy_aware_controller.compression_rate = runner.compression_rate_target
            runner.tensorboard_writer.add_scalar('compression/accuracy_aware/target_compression_rate',
                                                 runner.compression_rate_target, runner.cumulative_epoch_count)
            runner.tensorboard_writer.add_scalar('compression/accuracy_aware/compression_rate_step',
                                                 runner.compression_rate_step, runner.cumulative_epoch_count)

        compressed_model_accuracy = runner.train_epoch(model, accuracy_aware_controller)
        runner.accuracy_bugdet = compressed_model_accuracy - runner.minimal_tolerable_accuracy
        runner.tensorboard_writer.add_scalar('val/accuracy_aware/accuracy_bugdet', runner.accuracy_bugdet,
                                             runner.cumulative_epoch_count)

    runner.load_best_checkpoint(model)

    return model


def run_initial_training_phase(model, accuracy_aware_controller, runner):
    runner.configure_optimizers()
    for _ in range(runner.initial_training_phase_epochs):
        runner.train_epoch(model, accuracy_aware_controller)
    compressed_model_accuracy = runner.validate(model, accuracy_aware_controller)
    runner.accuracy_bugdet = compressed_model_accuracy - runner.minimal_tolerable_accuracy
    runner.tensorboard_writer.add_scalar('val/accuracy_aware/accuracy_bugdet', runner.accuracy_bugdet,
                                         runner.cumulative_epoch_count)
    nncf_logger.info('Accuracy budget value after training is {}'.format(runner.accuracy_bugdet))


def determine_compression_variable_controller(compression_controller):
    accuracy_aware_controllers = ACCURACY_AWARE_CONTROLLERS.registry_dict.values()
    if not isinstance(compression_controller, CompositeCompressionAlgorithmController):
        for ctrl_type in accuracy_aware_controllers:
            if isinstance(compression_controller, ctrl_type):
                return compression_controller
        return None
    for controller in compression_controller.child_ctrls:
        for ctrl_type in accuracy_aware_controllers:
            if isinstance(controller, ctrl_type):
                return controller
    return None


def update_target_compression_rate(accuracy_aware_controller, runner):
    current_compression_rate = accuracy_aware_controller.compression_rate
    best_accuracy_budget = runner.best_val_metric_value - runner.minimal_tolerable_accuracy
    if runner.compression_rate_target is None:
        runner.compression_rate_target = current_compression_rate + \
            determine_compression_rate_step_value(runner, current_compression_rate)
        runner.was_compression_increased_on_prev_step = np.sign(best_accuracy_budget)
        accuracy_aware_controller.disable_scheduler()
        return True
    if runner.training_epoch_count >= runner.patience_epochs:
        runner.compression_rate_target += determine_compression_rate_step_value(runner, current_compression_rate)
        runner.was_compression_increased_on_prev_step = np.sign(best_accuracy_budget)
        return True
    return False


def determine_compression_rate_step_value(runner, current_compression_rate,
                                          stepping_mode='interpolate', **kwargs):
    compression_step_updaters = {
        'uniform_decrease': uniform_decrease_compression_step_update,
        'interpolate': partial(interpolate_compression_step_update,
                               current_compression_rate=current_compression_rate),
    }
    return compression_step_updaters[stepping_mode](runner, **kwargs)


def uniform_decrease_compression_step_update(runner):
    best_accuracy_budget_sign = np.sign(runner.best_val_metric_value - runner.minimal_tolerable_accuracy)
    if runner.was_compression_increased_on_prev_step != best_accuracy_budget_sign:
        runner.compression_rate_step *= runner.step_reduction_factor
    return best_accuracy_budget_sign * runner.compression_rate_step


def interpolate_compression_step_update(runner,
                                        current_compression_rate,
                                        num_curve_pts=1000,
                                        full_compression_factor=10):
    training_history = runner.compressed_training_history
    nncf_logger.info('Compressed training history: {}'.format(training_history))
    training_history[0.0] = runner.maximal_accuracy_drop
    training_history[1.0] = -full_compression_factor * runner.maximal_accuracy_drop
    compression_rates, evaluated_acc_budgets = list(training_history.keys()), list(training_history.values())
    interp_kind = 'linear' if len(compression_rates) < 4 else 'cubic'
    acc_budget_vs_comp_rate_curve = interp1d(compression_rates, evaluated_acc_budgets,
                                             kind=interp_kind)
    rate_interval = np.linspace(0.0, 1.0, num=num_curve_pts, endpoint=True)
    acc_budget_values = acc_budget_vs_comp_rate_curve(rate_interval)
    target_compression_rate = rate_interval[np.argmin(np.abs(acc_budget_values))]
    nncf_logger.info('Predicted compression rate {}, '
                     'current compression rate {}'.format(target_compression_rate,
                                                          current_compression_rate))
    if runner.compression_rate_target is None:
        runner.compression_rate_step = np.abs(target_compression_rate - current_compression_rate)
        return target_compression_rate - current_compression_rate
    runner.compression_rate_step = np.abs(target_compression_rate - runner.compression_rate_target)
    return target_compression_rate - runner.compression_rate_target
