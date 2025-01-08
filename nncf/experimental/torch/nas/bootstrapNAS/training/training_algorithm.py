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
from pathlib import Path
from shutil import copyfile
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import torch

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import CompressionStage
from nncf.common.logging import nncf_logger
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_controller import ElasticityController
from nncf.experimental.torch.nas.bootstrapNAS.training.base_training import BNASTrainingController
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import is_main_process

TModel = TypeVar("TModel")
OptimizerType = TypeVar("OptimizerType")
LRSchedulerType = TypeVar("LRSchedulerType")
TensorboardWriterType = TypeVar("TensorboardWriterType")
DataLoaderType = TypeVar("DataLoaderType")
TrainEpochFnType = Callable[
    [
        DataLoaderType,
        TModel,
        CompressionAlgorithmController,
        int,
        OptimizerType,
    ],
    None,
]
ValFnType = Callable[[TModel, DataLoaderType], Tuple[float, float, float]]


class EBTrainAlgoStateNames:
    MODEL_STATE = "model_state"
    EPOCH = "epoch"
    SUPERNET_ACC1 = "acc1"
    SUPERNET_BEST_ACC1 = "best_acc1"
    MIN_SUBNET_ACC1 = "min_subnet_acc1"
    MIN_SUBNET_BEST_ACC1 = "min_subnet_best_acc1"
    OPTIMIZER = "optimizer"
    TRAINING_ALGO_STATE = "training_algo_state"


class EpochBasedTrainingAlgorithm:
    """
    Algorithm for training supernet by using a train function for a single epoch. In contrast, there can be step-based
    algorithm that uses a train function for a single training step.
    """

    _state_names = EBTrainAlgoStateNames

    def __init__(
        self,
        nncf_network: NNCFNetwork,
        training_ctrl: BNASTrainingController,
        checkpoint: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the training algorithm

        :param nncf_network: it's supposed to be a model with elasticity ops,
        controlled by elasticity ctrl, which is owned by training ctrl
        :param training_ctrl: controller of the training algorithm (by default ProgressiveShrinkingController)
        :param checkpoint: data to restore state of the training algorithm
        """
        self._model = nncf_network
        self._training_ctrl = training_ctrl
        self._start_epoch = 0
        self._supernet_best_acc1 = 0
        self._min_subnet_best_acc1 = 0
        self._optimizer = None
        self._optimizer_state = None
        if checkpoint is not None:
            resuming_model_state_dict = checkpoint[self._state_names.MODEL_STATE]
            load_state(self._model, resuming_model_state_dict, is_resume=True)
            self._optimizer_state = checkpoint[self._state_names.OPTIMIZER]
            self._start_epoch = checkpoint[self._state_names.EPOCH]
            self._supernet_best_acc1 = checkpoint[self._state_names.SUPERNET_BEST_ACC1]
            self._min_subnet_best_acc1 = checkpoint[self._state_names.MIN_SUBNET_BEST_ACC1]

    @property
    def elasticity_ctrl(self) -> ElasticityController:
        """
        :return: elasticity controller
        """
        return self._training_ctrl.elasticity_controller

    def run(
        self,
        train_epoch_fn: TrainEpochFnType,
        train_loader: DataLoaderType,
        val_fn: ValFnType,
        val_loader: DataLoaderType,
        optimizer: OptimizerType,
        checkpoint_save_dir: str,
        tensorboard_writer: Optional[TensorboardWriterType] = None,
        train_iters: Optional[float] = None,
    ) -> Tuple[NNCFNetwork, ElasticityController]:
        """
        Implements a training loop for supernet training.

        :param train_epoch_fn: a method to fine-tune the model for a single epoch
        :param train_loader: data loader for training
        :param lr_scheduler: scheduler for learning rate
        :param val_fn: a method to evaluate the model on the validation dataset
        :param val_loader: data loader for validation
        :param optimizer: a training optimizer
        :param checkpoint_save_dir: path to a directory for saving checkpoints
        :param tensorboard_writer: The tensorboard object to be used for logging.
        :return: the fine-tuned model and elasticity controller
        """

        if train_iters is None:
            train_iters = len(train_loader)
        self._training_ctrl.set_training_lr_scheduler_args(optimizer, train_iters)  # len(train_loader))

        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)

        self._optimizer = optimizer

        log_validation_info = True
        if tensorboard_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tensorboard_writer = SummaryWriter(checkpoint_save_dir)
                # log compression config to tensorboard
            except ModuleNotFoundError:
                nncf_logger.warning(
                    "Tensorboard installation not found! Install tensorboard Python package "
                    "in order for BootstrapNAS tensorboard data to be dumped"
                )

        total_num_epochs = self._training_ctrl.get_total_num_epochs()

        best_compression_stage = CompressionStage.UNCOMPRESSED
        for epoch in range(self._start_epoch, total_num_epochs):
            self._training_ctrl.scheduler.epoch_step()

            # TODO(nlyalyus): support DDP (ticket 56849)
            # if config.distributed:
            #     train_sampler.set_epoch(epoch)

            # train for one epoch
            train_epoch_fn(train_loader, self._model, self._training_ctrl, epoch, optimizer)

            compression_stage = self._training_ctrl.compression_stage()

            self._training_ctrl.multi_elasticity_handler.activate_minimum_subnet()
            min_subnet_acc1, acc5, loss = self._validate_subnet(val_fn, val_loader)
            if log_validation_info:
                nncf_logger.info(
                    f"* Acc@1 {min_subnet_acc1:.3f} Acc@5 {acc5:.3f} "
                    f"for Minimal SubNet={self._training_ctrl.multi_elasticity_handler.get_active_config()}"
                )
            if is_main_process() and log_validation_info:
                tensorboard_writer.add_scalar("val/min_subnet_loss", loss, len(val_loader) * epoch)
                tensorboard_writer.add_scalar("val/min_subnet_top1", min_subnet_acc1, len(val_loader) * epoch)
                tensorboard_writer.add_scalar("val/min_subnet_top5", acc5, len(val_loader) * epoch)
            min_subnet_best_acc1 = self._define_best_accuracy(
                min_subnet_acc1, self._min_subnet_best_acc1, compression_stage, best_compression_stage
            )

            self._training_ctrl.multi_elasticity_handler.activate_supernet()
            supernet_acc1, acc5, loss = self._validate_subnet(val_fn, val_loader)
            if log_validation_info:
                nncf_logger.info(
                    f"* Acc@1 {supernet_acc1:.3f} Acc@5 {acc5:.3f} "
                    f"of SuperNet={self._training_ctrl.multi_elasticity_handler.get_active_config()}"
                )

            if is_main_process() and log_validation_info:
                tensorboard_writer.add_scalar("val/supernet_loss", loss, len(val_loader) * epoch)
                tensorboard_writer.add_scalar("val/supernet_top1", supernet_acc1, len(val_loader) * epoch)
                tensorboard_writer.add_scalar("val/supernet_top5", acc5, len(val_loader) * epoch)
            supernet_best_acc1 = self._define_best_accuracy(
                supernet_acc1, self._supernet_best_acc1, compression_stage, best_compression_stage
            )

            best_compression_stage = max(compression_stage, best_compression_stage)

            checkpoint_path = Path(checkpoint_save_dir, "supernet_last.pth")
            checkpoint = {
                self._state_names.EPOCH: epoch + 1,
                self._state_names.MODEL_STATE: self._model.state_dict(),
                self._state_names.SUPERNET_BEST_ACC1: supernet_best_acc1,
                self._state_names.SUPERNET_ACC1: supernet_acc1,
                self._state_names.MIN_SUBNET_BEST_ACC1: min_subnet_best_acc1,
                self._state_names.MIN_SUBNET_ACC1: min_subnet_acc1,
                self._state_names.OPTIMIZER: optimizer.state_dict(),
                self._state_names.TRAINING_ALGO_STATE: self._training_ctrl.get_compression_state(),
            }
            torch.save(checkpoint, checkpoint_path)

            if compression_stage == CompressionStage.FULLY_COMPRESSED and supernet_best_acc1 == supernet_acc1:
                best_path = Path(checkpoint_save_dir) / "supernet_best.pth"
                copyfile(checkpoint_path, best_path)

            # Backup elasticity state and model weight to directly restore from it in a separate search sample
            elasticity_path = Path(checkpoint_save_dir) / "last_elasticity.pth"
            elasticity_state = self._training_ctrl.elasticity_controller.get_compression_state()
            model_path = Path(checkpoint_save_dir) / "last_model_weights.pth"
            model_state = self._model.state_dict()
            torch.save(elasticity_state, elasticity_path)
            torch.save(model_state, model_path)

        return self._model, self.elasticity_ctrl

    @classmethod
    def from_config(cls, nncf_network: NNCFNetwork, nncf_config: NNCFConfig) -> "EpochBasedTrainingAlgorithm":
        """
        Creates the training algorithm from a config by a given empty NNCFNetwork.

        :param nncf_network: empty NNCFNetwork
        :param nncf_config: parameters of the training algorithm
        :return: the training algorithm
        """
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        training_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )
        return EpochBasedTrainingAlgorithm(model, training_ctrl)

    @classmethod
    def from_checkpoint(
        cls, nncf_network: NNCFNetwork, bn_adapt_args: BNAdaptationInitArgs, resuming_checkpoint_path: str
    ) -> "EpochBasedTrainingAlgorithm":
        """
        Resumes the training algorithm from a checkpoint by a given empty NNCFNetwork and BN adaptation arguments only,
        config is not involved.

        :param nncf_network: empty NNCFNetwork
        :param bn_adapt_args: arguments for batchnorm statistics adaptation algorithm
        :param resuming_checkpoint_path: path to the resuming checkpoint
        :return: the training algorithm
        """
        if not Path(resuming_checkpoint_path).is_file():
            raise FileNotFoundError("no checkpoint found at '{}'".format(resuming_checkpoint_path))
        nncf_logger.info(f"=> loading checkpoint '{resuming_checkpoint_path}'")
        checkpoint = torch.load(resuming_checkpoint_path, map_location="cpu")

        training_state = checkpoint[cls._state_names.TRAINING_ALGO_STATE]
        nncf_config = NNCFConfig()
        nncf_config.register_extra_structs([bn_adapt_args])
        model, training_ctrl = resume_compression_from_state(nncf_network, training_state, nncf_config)
        return EpochBasedTrainingAlgorithm(model, training_ctrl, checkpoint)

    @staticmethod
    def _define_best_accuracy(
        acc1: float, best_acc1: float, compression_stage: CompressionStage, best_compression_stage: CompressionStage
    ) -> float:
        """
        The best accuracy value should be considered not only by value, but also per each stage.
        Currently, there are two stages in NAS algo only: PARTIALLY_COMPRESSED and FULLY_COMPRESSED.
        It's aligned with Compression Algorithm stages, but ideally number of stages should correspond to number of
        scheduler stages (usually more than two).
        The last means that scheduler is activated the last stage.
        """
        is_best_by_accuracy = acc1 > best_acc1 and compression_stage == best_compression_stage
        is_best = is_best_by_accuracy or compression_stage > best_compression_stage
        if is_best:
            best_acc1 = acc1
        return best_acc1

    def _validate_subnet(self, val_fn: ValFnType, val_loader: DataLoaderType):
        self._training_ctrl.prepare_for_validation()
        return val_fn(self._model, val_loader)
