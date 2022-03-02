"""
 Copyright (c) 2022 Intel Corporation
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
from abc import ABC
from abc import abstractmethod
from copy import copy
from functools import partial
from typing import TypeVar

import numpy as np
from scipy.interpolate import interp1d

from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionStage
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.registry import Registry
from nncf.config.config import NNCFConfig
from nncf.config.extractors import extract_accuracy_aware_training_params
from nncf.common.accuracy_aware_training.runner import EarlyExitTrainingRunnerCreator
from nncf.common.accuracy_aware_training.runner import AdaptiveCompressionLevelTrainingRunnerCreator

ModelType = TypeVar('ModelType')
ADAPTIVE_COMPRESSION_CONTROLLERS = Registry('adaptive_compression_controllers')


class TrainingLoop(ABC):
    """
    The training loop object is instantiated by the user, the training process
    is launched via the `run` method.
    """

    @abstractmethod
    def run(self, model: ModelType, train_epoch_fn, validate_fn, configure_optimizers_fn=None,
            dump_checkpoint_fn=None, tensorboard_writer=None, log_dir=None):
        """
        Implements the custom logic to run a training loop for model fine-tuning
        by using the provided `train_epoch_fn`, `validate_fn` and `configure_optimizers_fn` methods.
        The passed methods are registered in the `TrainingRunner` instance and the training logic
        is implemented by calling the corresponding `TrainingRunner` methods

        :param model: The model instance before fine-tuning
        :param train_epoch_fn: a method to fine-tune the model for a single epoch
        (to be called inside the `train_epoch` of the TrainingRunner)
        :param validate_fn: a method to evaluate the model on the validation dataset
        (to be called inside the `train_epoch` of the TrainingRunner)
        :param dump_checkpoint_fn: a method to dump a checkpoint
        :param configure_optimizers_fn: a method to instantiate an optimizer and a learning
        rate scheduler (to be called inside the `configure_optimizers` of the TrainingRunner)
        :return: The fine-tuned model
        """


class EarlyExitCompressionTrainingLoop(TrainingLoop):
    """
    Adaptive compression training loop allows an accuracy-aware training process
    to reach the maximal accuracy drop
    (the maximal allowed accuracy degradation criterion is satisfied).
    """

    def __init__(self,
                 nncf_config: NNCFConfig,
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed=True, verbose=True,
                 dump_checkpoints=True):
        super().__init__()
        accuracy_aware_training_params = extract_accuracy_aware_training_params(nncf_config)
        runner_factory = EarlyExitTrainingRunnerCreator(accuracy_aware_training_params,
                                                        compression_controller,
                                                        lr_updates_needed, verbose,
                                                        dump_checkpoints)
        self.runner = runner_factory.create_training_loop()
        self.compression_controller = compression_controller

    def run(self, model, train_epoch_fn, validate_fn, configure_optimizers_fn=None,
            dump_checkpoint_fn=None, tensorboard_writer=None, log_dir=None):
        self.runner.initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn,
                                                 dump_checkpoint_fn, tensorboard_writer, log_dir)

        return self._run_early_exit_training_loop(model)

    def _run_early_exit_training_loop(self, model):
        self.runner.retrieve_uncompressed_model_accuracy(model)
        uncompressed_model_accuracy = self.runner.uncompressed_model_accuracy
        self.runner.calculate_minimal_tolerable_accuracy(uncompressed_model_accuracy)

        self.runner.configure_optimizers()

        compressed_model_accuracy = self.runner.validate(model)
        accuracy_budget = self._calculate_accuracy_budget(self.runner.minimal_tolerable_accuracy,
                                                          compressed_model_accuracy)
        accuracy_drop = self._calculate_accuracy_drop(uncompressed_model_accuracy, compressed_model_accuracy)
        rel_accuracy_drop = self._calculate_rel_accuracy_drop(uncompressed_model_accuracy,
                                                              compressed_model_accuracy)

        if self._accuracy_criterion_satisfied(accuracy_budget, self.compression_controller):
            self.runner.dump_statistics(model, self.compression_controller)
            nncf_logger.info('The accuracy criteria is reached after the initialization step.')
            self.print_accuracy_statistics(uncompressed_model_accuracy, compressed_model_accuracy,
                                           accuracy_drop, rel_accuracy_drop, accuracy_budget)
            return model

        nncf_logger.info('Results of the initialization step:')
        self.print_accuracy_statistics(uncompressed_model_accuracy, compressed_model_accuracy,
                                       accuracy_drop, rel_accuracy_drop, accuracy_budget)

        for epoch in range(1, self.runner.maximal_total_epochs + 1):
            self.runner.train_epoch(model, self.compression_controller)
            compressed_model_accuracy = self.runner.validate(model)
            self.runner.dump_statistics(model, self.compression_controller)
            accuracy_budget = self._calculate_accuracy_budget(self.runner.minimal_tolerable_accuracy,
                                                              compressed_model_accuracy)
            accuracy_drop = self._calculate_accuracy_drop(uncompressed_model_accuracy,
                                                          compressed_model_accuracy)
            rel_accuracy_drop = self._calculate_rel_accuracy_drop(uncompressed_model_accuracy,
                                                                  compressed_model_accuracy)
            if self._accuracy_criterion_satisfied(accuracy_budget, self.compression_controller):
                nncf_logger.info('The accuracy criteria is reached after {} epoch.'.format(epoch))
                self.print_accuracy_statistics(uncompressed_model_accuracy, compressed_model_accuracy,
                                               accuracy_drop, rel_accuracy_drop, accuracy_budget)
                break

            nncf_logger.info('Results of {} epoch:'.format(epoch))
            self.print_accuracy_statistics(uncompressed_model_accuracy, compressed_model_accuracy,
                                           accuracy_drop, rel_accuracy_drop, accuracy_budget)

            self.runner.update_learning_rate()

        self.runner.load_best_checkpoint(model)
        return model

    @staticmethod
    def print_accuracy_statistics(uncompressed_model_accuracy,
                                  compressed_model_accuracy,
                                  accuracy_drop,
                                  rel_accuracy_drop,
                                  accuracy_budget):
        nncf_logger.info('Original model accuracy: {:.4f}'.format(uncompressed_model_accuracy))
        nncf_logger.info('Compressed model accuracy: {:.4f}'.format(compressed_model_accuracy))
        nncf_logger.info('Absolute accuracy drop: {:.4f}'.format(accuracy_drop))
        nncf_logger.info('Relative accuracy drop: {:.2f}%'.format(rel_accuracy_drop))
        nncf_logger.info('Accuracy budget: {:.4f}'.format(accuracy_budget))

    @staticmethod
    def _calculate_accuracy_drop(uncompressed_model_accuracy, compressed_model_accuracy):
        return uncompressed_model_accuracy - compressed_model_accuracy

    @staticmethod
    def _calculate_accuracy_budget(minimal_tolerable_accuracy, compressed_model_accuracy):
        return compressed_model_accuracy - minimal_tolerable_accuracy

    @staticmethod
    def _calculate_rel_accuracy_drop(uncompressed_model_accuracy, compressed_model_accuracy):
        try:
            rel_accuracy_drop = 100 * (1.0 - compressed_model_accuracy / uncompressed_model_accuracy)
        except ZeroDivisionError:
            rel_accuracy_drop = 0

        return rel_accuracy_drop

    @staticmethod
    def _accuracy_criterion_satisfied(accuracy_budget, compression_controller):
        return accuracy_budget >= 0 and \
               compression_controller.compression_stage() == CompressionStage.FULLY_COMPRESSED


class AdaptiveCompressionTrainingLoop(TrainingLoop):
    """
    Adaptive compression training loop allows an accuracy-aware training process whereby
    the compression rate is automatically varied during training to reach the maximal
    possible compression rate with a positive accuracy budget
    (the maximal allowed accuracy degradation criterion is satisfied).
    """

    def __init__(self,
                 nncf_config: NNCFConfig,
                 compression_controller: CompressionAlgorithmController,
                 lr_updates_needed=True, verbose=True,
                 minimal_compression_rate=0.05,
                 maximal_compression_rate=0.95,
                 dump_checkpoints=True):
        self.adaptive_controller = self._get_adaptive_compression_ctrl(compression_controller)
        if self.adaptive_controller is None:
            raise RuntimeError('No compression algorithm supported by the accuracy-aware training '
                               'runner was specified in the config')

        accuracy_aware_training_params = extract_accuracy_aware_training_params(nncf_config)
        runner_factory = AdaptiveCompressionLevelTrainingRunnerCreator(accuracy_aware_training_params,
                                                                       self.adaptive_controller,
                                                                       lr_updates_needed, verbose,
                                                                       minimal_compression_rate,
                                                                       maximal_compression_rate,
                                                                       dump_checkpoints)
        self.runner = runner_factory.create_training_loop()

    def _get_adaptive_compression_ctrl(self, compression_controller):
        def _adaptive_compression_controllers():
            def remove_registry_prefix(algo_name):
                for prefix in ('pt_', 'tf_'):
                    if algo_name.startswith(prefix):
                        return algo_name[len(prefix):]
                raise RuntimeError('Compression algorithm names in the adaptive controllers '
                                   'registry should be prefixed with "pt_" or "tf_" depending on the '
                                   'backend framework')

            return {remove_registry_prefix(algo_name): controller_cls for algo_name, controller_cls in
                    ADAPTIVE_COMPRESSION_CONTROLLERS.registry_dict.items()}

        adaptive_compression_controllers = _adaptive_compression_controllers()

        if isinstance(compression_controller, CompositeCompressionAlgorithmController):
            for controller in compression_controller.child_ctrls:
                for ctrl_type in adaptive_compression_controllers.values():
                    if isinstance(controller, ctrl_type):
                        return controller
        elif isinstance(compression_controller, CompressionAlgorithmController):
            if compression_controller.name in adaptive_compression_controllers:
                return compression_controller

        raise RuntimeError('No compression algorithm that supports adaptive compression '
                           'accuracy-aware training was specified')

    def run(self, model, train_epoch_fn, validate_fn, configure_optimizers_fn=None,
            dump_checkpoint_fn=None, tensorboard_writer=None, log_dir=None):
        self.runner.initialize_training_loop_fns(train_epoch_fn, validate_fn, configure_optimizers_fn,
                                                 dump_checkpoint_fn, tensorboard_writer, log_dir)
        self.runner.retrieve_uncompressed_model_accuracy(model)
        uncompressed_model_accuracy = self.runner.uncompressed_model_accuracy
        self.runner.calculate_minimal_tolerable_accuracy(uncompressed_model_accuracy)
        self._run_initial_training_phase(model, self.adaptive_controller, self.runner)
        self.runner.add_tensorboard_scalar('compression/accuracy_aware/target_compression_rate',
                                           self.adaptive_controller.compression_rate,
                                           self.runner.cumulative_epoch_count)
        self.runner.update_training_history(compression_rate=self.adaptive_controller.compression_rate,
                                            best_metric_value=self.runner.best_val_metric_value)

        while self.runner.compression_rate_step >= self.runner.minimal_compression_rate_step and \
                self.runner.cumulative_epoch_count < self.runner.maximal_total_epochs:

            if self.runner.compression_rate_target is not None:
                self.runner.update_training_history(compression_rate=copy(self.runner.compression_rate_target),
                                                    best_metric_value=copy(self.runner.best_val_metric_value))

            was_compression_rate_changed = self._update_target_compression_rate(self.adaptive_controller, self.runner)
            nncf_logger.info('Current target compression rate value: '
                             '{comp_rate:.3f}'.format(comp_rate=self.runner.compression_rate_target))
            nncf_logger.info('Current accuracy budget value: '
                             '{acc_budget:.3f}'.format(acc_budget=self.runner.accuracy_bugdet))
            nncf_logger.info('Current compression rate step value: '
                             '{comp_step:.3f}'.format(comp_step=self.runner.compression_rate_step))

            if was_compression_rate_changed:
                if self.runner.compression_rate_target < self.runner.minimal_compression_rate:
                    raise RuntimeError('Cannot produce a compressed model with a specified '
                                       'minimal tolerable accuracy')
                if self.runner.compression_rate_target > self.runner.maximal_compression_rate:
                    nncf_logger.info('Reached maximal possible compression rate '
                                     '{max_rate}'.format(max_rate=self.runner.maximal_compression_rate))
                    self.runner.dump_statistics(model, self.adaptive_controller)
                    return model

                self.runner.reset_training()
                self.adaptive_controller.compression_rate = self.runner.compression_rate_target

                # (alexander.suslov) workaround for compression statistics and disabling scheduler
                self.adaptive_controller.scheduler.current_pruning_level = self.runner.compression_rate_target
                self.adaptive_controller.scheduler.current_sparsity_level = self.runner.compression_rate_target
                self.adaptive_controller.scheduler.target_level = self.runner.compression_rate_target

                self.runner.add_tensorboard_scalar('compression/accuracy_aware/target_compression_rate',
                                                   self.runner.compression_rate_target,
                                                   self.runner.cumulative_epoch_count)
                self.runner.add_tensorboard_scalar('compression/accuracy_aware/compression_rate_step',
                                                   self.runner.compression_rate_step,
                                                   self.runner.cumulative_epoch_count)

            self.runner.train_epoch(model, self.adaptive_controller)
            compressed_model_accuracy = self.runner.validate(model)
            self.runner.dump_statistics(model, self.adaptive_controller)
            self.runner.accuracy_bugdet = compressed_model_accuracy - self.runner.minimal_tolerable_accuracy
            self.runner.add_tensorboard_scalar('val/accuracy_aware/accuracy_bugdet', self.runner.accuracy_bugdet,
                                               self.runner.cumulative_epoch_count)
        self.runner.load_best_checkpoint(model)
        compressed_model_accuracy = self.runner.validate(model)
        possible_checkpoint_compression_rates = self.runner.get_compression_rates_with_positive_acc_budget()
        best_checkpoint_compression_rate = max(possible_checkpoint_compression_rates)
        nncf_logger.info('The final compressed model has {} compression rate with {} accuracy'.format(
            best_checkpoint_compression_rate, compressed_model_accuracy))
        return model

    @staticmethod
    def _run_initial_training_phase(model, accuracy_aware_controller, runner):
        runner.configure_optimizers()
        for _ in range(runner.initial_training_phase_epochs):
            runner.train_epoch(model, accuracy_aware_controller)
        compressed_model_accuracy = runner.validate(model)
        runner.accuracy_bugdet = compressed_model_accuracy - runner.minimal_tolerable_accuracy
        runner.add_tensorboard_scalar('val/accuracy_aware/accuracy_bugdet',
                                      runner.accuracy_bugdet, runner.cumulative_epoch_count)
        nncf_logger.info('Accuracy budget value after training is {}'.format(runner.accuracy_bugdet))

    def _update_target_compression_rate(self, accuracy_aware_controller, runner):
        current_compression_rate = accuracy_aware_controller.compression_rate
        best_accuracy_budget = runner.best_val_metric_value - runner.minimal_tolerable_accuracy
        if runner.compression_rate_target is None:
            runner.compression_rate_target = current_compression_rate + \
                                             self._determine_compression_rate_step_value(runner,
                                                                                         current_compression_rate)
            runner.was_compression_increased_on_prev_step = np.sign(best_accuracy_budget)
            accuracy_aware_controller.disable_scheduler()
            # TODO(kshpv) fix this incorrect work of disable_scheduler()
            accuracy_aware_controller.scheduler.target_level = runner.compression_rate_target
            return True
        if runner.training_epoch_count >= runner.patience_epochs:
            runner.compression_rate_target += self._determine_compression_rate_step_value(runner,
                                                                                          current_compression_rate)
            runner.was_compression_increased_on_prev_step = np.sign(best_accuracy_budget)
            return True
        return False

    def _determine_compression_rate_step_value(self, runner, current_compression_rate,
                                               stepping_mode='uniform_decrease', **kwargs):
        compression_step_updaters = {
            'uniform_decrease': self._uniform_decrease_compression_step_update,
            'interpolate': partial(self._interpolate_compression_step_update,
                                   current_compression_rate=current_compression_rate),
        }
        return compression_step_updaters[stepping_mode](runner, **kwargs)

    @staticmethod
    def _uniform_decrease_compression_step_update(runner):
        best_accuracy_budget_sign = np.sign(runner.best_val_metric_value - runner.minimal_tolerable_accuracy)
        if runner.was_compression_increased_on_prev_step is not None and \
                runner.was_compression_increased_on_prev_step != best_accuracy_budget_sign:
            runner.compression_rate_step *= runner.step_reduction_factor
        return best_accuracy_budget_sign * runner.compression_rate_step

    @staticmethod
    def _interpolate_compression_step_update(runner,
                                             current_compression_rate,
                                             num_curve_pts=1000,
                                             full_compression_factor=20,
                                             minimal_compression_rate=0.0,
                                             maximal_compression_rate=1.0):
        training_history = runner.compressed_training_history
        nncf_logger.info('Compressed training history: {}'.format(training_history))
        training_history[minimal_compression_rate] = runner.maximal_accuracy_drop
        training_history[maximal_compression_rate] = -full_compression_factor * runner.maximal_accuracy_drop
        compression_rates, evaluated_acc_budgets = list(training_history.keys()), list(training_history.values())
        interp_kind = 'linear' if len(compression_rates) < 4 else 'cubic'
        acc_budget_vs_comp_rate_curve = interp1d(compression_rates, evaluated_acc_budgets,
                                                 kind=interp_kind)
        rate_interval = np.linspace(minimal_compression_rate, maximal_compression_rate,
                                    num=num_curve_pts, endpoint=True)
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


class AccuracyAwareTrainingMode:
    EARLY_EXIT = 'early_exit'
    ADAPTIVE_COMPRESSION_LEVEL = 'adaptive_compression_level'


def create_accuracy_aware_training_loop(nncf_config: NNCFConfig,
                                        compression_ctrl: CompressionAlgorithmController,
                                        **additional_runner_args) -> TrainingLoop:
    """
    Creates an accuracy aware training loop corresponding to NNCFConfig and CompressionAlgorithmController.
    :param: nncf_config: An instance of the NNCFConfig.
    :compression_ctrl: An instance of thr CompressionAlgorithmController.
    :return: Accuracy aware training loop.
    """
    accuracy_aware_training_params = extract_accuracy_aware_training_params(nncf_config)
    accuracy_aware_training_mode = accuracy_aware_training_params.get('mode')
    if accuracy_aware_training_mode == AccuracyAwareTrainingMode.EARLY_EXIT:
        return EarlyExitCompressionTrainingLoop(nncf_config, compression_ctrl, **additional_runner_args)
    if accuracy_aware_training_mode == AccuracyAwareTrainingMode.ADAPTIVE_COMPRESSION_LEVEL:
        return AdaptiveCompressionTrainingLoop(nncf_config, compression_ctrl, **additional_runner_args)
    raise RuntimeError('Incorrect accuracy aware mode in the config file')
