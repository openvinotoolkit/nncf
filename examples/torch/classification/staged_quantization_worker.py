"""
 Copyright (c) 2019-2022 Intel Corporation
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

import copy
import os.path as osp
import time

import torch
from torch.backends import cudnn
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.models import InceptionOutputs

from examples.torch.classification.main import AverageMeter
from examples.torch.classification.main import accuracy
from examples.torch.classification.main import create_data_loaders
from examples.torch.classification.main import create_datasets
from examples.torch.classification.main import get_lr
from examples.torch.classification.main import inception_criterion_fn
from examples.torch.classification.main import validate
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import ExecutionMode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import set_seed
from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.model_loader import MODEL_STATE_ATTR
from examples.torch.common.model_loader import extract_model_and_compression_states
from examples.torch.common.model_loader import load_model
from examples.torch.common.model_loader import load_resuming_checkpoint
from examples.torch.common.utils import SafeMLFLow
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import get_name
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import log_common_mlflow_params
from examples.torch.common.utils import make_additional_checkpoints
from examples.torch.common.utils import print_args
from nncf.api.compression import CompressionStage
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.torch.binarization.algo import BinarizationController
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.initialization import register_default_init_args
from nncf.torch import create_compressed_model
from nncf.torch.quantization.algo import QuantizationController
from nncf.torch.utils import is_main_process


class KDLossCalculator:
    def __init__(self, original_model, temperature=1.0):
        self.original_model = original_model
        self.original_model.eval()
        self.temperature = temperature

    def loss(self, inputs, quantized_network_outputs):
        T = self.temperature
        with torch.no_grad():
            ref_output = self.original_model(inputs).detach()
        kd_loss = -(nn.functional.log_softmax(quantized_network_outputs / T, dim=1) *
                    nn.functional.softmax(ref_output / T, dim=1)).mean() * (T * T * quantized_network_outputs.shape[1])
        return kd_loss


def get_quantization_optimizer(params_to_optimize, quantization_config):
    params = quantization_config.get("params", {})
    base_lr = params.get("base_lr", 1e-3)
    base_wd = params.get("base_wd", 1e-5)
    return torch.optim.Adam(params_to_optimize,
                            lr=base_lr,
                            weight_decay=base_wd)


class PolyLRDropScheduler:
    def __init__(self, optimizer, quantization_config):
        params = quantization_config.get('params', {})
        self.base_lr = params.get("base_lr", 1e-3)
        self.lr_poly_drop_start_epoch = params.get('lr_poly_drop_start_epoch', None)
        self.lr_poly_drop_duration_epochs = params.get('lr_poly_drop_duration_epochs', 30)
        self.disable_wd_start_epoch = params.get('disable_wd_start_epoch', None)
        self.optimizer = optimizer
        self.last_epoch = 0

    def step(self, epoch_fraction):
        epoch_float = self.last_epoch + epoch_fraction
        if self.lr_poly_drop_start_epoch is not None:
            start = self.lr_poly_drop_start_epoch
            finish = self.lr_poly_drop_start_epoch + self.lr_poly_drop_duration_epochs
            if start <= epoch_float < finish:
                lr = self.base_lr * pow(float(finish - epoch_float) / float(self.lr_poly_drop_duration_epochs), 2.1)
                for group in self.optimizer.param_groups:
                    group['lr'] = lr

        if self.disable_wd_start_epoch is not None:
            if epoch_float > self.disable_wd_start_epoch:
                for group in self.optimizer.param_groups:
                    group['weight_decay'] = 0.0

    def epoch_step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
        self.last_epoch += 1

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def state_dict(self):
        return self.__dict__


# pylint:disable=too-many-branches
# pylint:disable=too-many-statements
def staged_quantization_main_worker(current_gpu, config):
    configure_device(current_gpu, config)
    config.mlflow = SafeMLFLow(config)

    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    set_seed(config)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    model_name = config['model']
    is_inception = 'inception' in model_name
    train_criterion_fn = inception_criterion_fn if is_inception else default_criterion_fn

    train_loader = train_sampler = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config

    pretrained = is_pretrained_model_requested(config)
    is_export_only = 'export' in config.mode and ('train' not in config.mode and 'test' not in config.mode)

    if is_export_only:
        assert pretrained or (resuming_checkpoint_path is not None)
    else:
        # Data loading code
        train_dataset, val_dataset = create_datasets(config)
        train_loader, train_sampler, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)

        def autoq_eval_fn(model, eval_loader):
            _, top5, _ = validate(eval_loader, model, criterion, config)
            return top5

        nncf_config = register_default_init_args(
            nncf_config, init_loader, criterion=criterion, criterion_fn=train_criterion_fn,
            autoq_eval_fn=autoq_eval_fn, val_loader=val_loader, device=config.device)

    # create model
    model_name = config['model']
    model = load_model(model_name,
                       pretrained=pretrained,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'),
                       weights_path=config.get('weights'))
    original_model = copy.deepcopy(model)

    model.to(config.device)

    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        resuming_checkpoint = load_resuming_checkpoint(resuming_checkpoint_path)
    model_state_dict, compression_state = extract_model_and_compression_states(resuming_checkpoint)
    compression_ctrl, model = create_compressed_model(model, nncf_config, compression_state)
    if model_state_dict is not None:
        load_state(model, model_state_dict, is_resume=True)

    if not isinstance(compression_ctrl, (BinarizationController, QuantizationController)):
        raise RuntimeError(
            "The stage quantization sample worker may only be run with the binarization and quantization algorithms!")

    model, _ = prepare_model_for_execution(model, config)
    original_model.to(config.device)

    if config.distributed:
        compression_ctrl.distributed()

    params_to_optimize = model.parameters()

    compression_config = config['compression']
    quantization_config = compression_config if isinstance(compression_config, dict) else compression_config[0]
    optimizer = get_quantization_optimizer(params_to_optimize, quantization_config)
    optimizer_scheduler = PolyLRDropScheduler(optimizer, quantization_config)
    kd_loss_calculator = KDLossCalculator(original_model)

    best_acc1 = 0
    # optionally resume from a checkpoint
    if resuming_checkpoint is not None and config.to_onnx is None:
        best_acc1 = resuming_checkpoint['best_acc1']
        if 'train' in config.mode:
            kd_loss_calculator.original_model.load_state_dict(resuming_checkpoint['original_model_state_dict'])
            config.start_epoch = resuming_checkpoint['epoch']
            optimizer.load_state_dict(resuming_checkpoint['optimizer'])
            optimizer_scheduler.load_state_dict(resuming_checkpoint['optimizer_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch: {}, best_acc1: {:.3f})"
                        .format(resuming_checkpoint_path, resuming_checkpoint['epoch'], best_acc1))
        else:
            logger.info("=> loaded checkpoint '{}'".format(resuming_checkpoint_path))

    log_common_mlflow_params(config)

    if is_export_only:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    if is_main_process():
        statistics = compression_ctrl.statistics()
        logger.info(statistics.to_str())

    if 'train' in config.mode:
        batch_multiplier = (quantization_config.get("params", {})).get("batch_multiplier", 1)
        train_staged(config, compression_ctrl, model, criterion, train_criterion_fn, optimizer_scheduler, model_name,
                     optimizer,
                     train_loader, train_sampler, val_loader, kd_loss_calculator, batch_multiplier, best_acc1)

    if 'test' in config.mode:
        validate(val_loader, model, criterion, config)

    if 'export' in config.mode:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))



def train_staged(config, compression_ctrl, model, criterion, criterion_fn, optimizer_scheduler, model_name, optimizer,
                 train_loader, train_sampler, val_loader, kd_loss_calculator, batch_multiplier, best_acc1=0):
    best_compression_stage = CompressionStage.UNCOMPRESSED
    for epoch in range(config.start_epoch, config.epochs):
        # update compression scheduler state at the start of the epoch
        compression_ctrl.scheduler.epoch_step()

        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch_staged(train_loader, batch_multiplier, model, criterion, criterion_fn, optimizer,
                           optimizer_scheduler, kd_loss_calculator, compression_ctrl, epoch, config)

        # compute compression algo statistics
        statistics = compression_ctrl.statistics()

        acc1 = best_acc1
        if epoch % config.test_every_n_epochs == 0:
            # evaluate on validation set
            # pylint: disable=E1123
            acc1, _, _ = validate(val_loader, model, criterion, config, epoch=epoch)

        compression_stage = compression_ctrl.compression_stage()
        # remember best acc@1, considering compression stage. If current acc@1 less then the best acc@1, checkpoint
        # still can be best if current compression stage is larger than the best one. Compression stages in ascending
        # order: UNCOMPRESSED, PARTIALLY_COMPRESSED, FULLY_COMPRESSED.
        is_best_by_accuracy = acc1 > best_acc1 and compression_stage == best_compression_stage
        is_best = is_best_by_accuracy or compression_stage > best_compression_stage
        best_acc1 = max(acc1, best_acc1)
        best_compression_stage = max(compression_stage, best_compression_stage)

        # statistics (e.g. portion of the enabled quantizers) is related to the finished epoch,
        # hence printing should happen before epoch_step, which may inform about state of the next epoch (e.g. next
        # portion of enabled quantizers)
        if is_main_process():
            logger.info(statistics.to_str())

        optimizer_scheduler.epoch_step()

        if is_main_process():
            checkpoint_path = osp.join(config.checkpoint_save_dir, get_name(config) + '_last.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'arch': model_name,
                MODEL_STATE_ATTR: model.state_dict(),
                COMPRESSION_STATE_ATTR: compression_ctrl.get_compression_state(),
                'original_model_state_dict': kd_loss_calculator.original_model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'optimizer_scheduler': optimizer_scheduler.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)
            make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)

            for key, value in prepare_for_tensorboard(statistics).items():
                config.mlflow.safe_call('log_metric', 'compression/statistics/{0}'.format(key), value, epoch)
                config.tb.add_scalar("compression/statistics/{0}".format(key), value, len(train_loader) * epoch)


def train_epoch_staged(train_loader, batch_multiplier, model, criterion, criterion_fn, optimizer,
                       optimizer_scheduler: PolyLRDropScheduler, kd_loss_calculator: KDLossCalculator,
                       compression_ctrl, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses_meter = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    compression_scheduler = compression_ctrl.scheduler

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        compression_scheduler.step()
        # measure data loading time
        data_time.update(time.time() - end)

        input_ = input_.to(config.device)
        target = target.to(config.device)

        output = model(input_)
        criterion_loss = criterion_fn(output, target, criterion)

        if isinstance(output, InceptionOutputs):
            output = output.logits

        # compute KD loss
        kd_loss = kd_loss_calculator.loss(input_, output)
        loss = criterion_loss + kd_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        comp_loss_val = kd_loss.item()
        kd_losses_meter.update(comp_loss_val, input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        if i % batch_multiplier == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()

        optimizer_scheduler.step(float(i) / len(train_loader))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logger.info(
                '{rank}: '
                'Epoch: [{0}][{1}/{2}] '
                'Lr: {3:.3} '
                'Wd: {4:.3} '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'CE_loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) '
                'KD_loss: {kd_loss.val:.4f} ({kd_loss.avg:.4f}) '
                'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), get_lr(optimizer), get_wd(optimizer), batch_time=batch_time,
                    data_time=data_time, ce_loss=criterion_losses, kd_loss=kd_losses_meter,
                    loss=losses, top1=top1, top5=top5,
                    rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                ))

        if is_main_process():
            global_step = len(train_loader) * epoch
            config.tb.add_scalar("train/learning_rate", get_lr(optimizer), i + global_step)
            config.tb.add_scalar("train/criterion_loss", criterion_losses.avg, i + global_step)
            config.tb.add_scalar("train/kd_loss", kd_losses_meter.avg, i + global_step)
            config.tb.add_scalar("train/loss", losses.avg, i + global_step)
            config.tb.add_scalar("train/top1", top1.avg, i + global_step)
            config.tb.add_scalar("train/top5", top5.avg, i + global_step)

            statistics = compression_ctrl.statistics(quickly_collected_only=True)
            for stat_name, stat_value in prepare_for_tensorboard(statistics).items():
                config.tb.add_scalar('train/statistics/{}'.format(stat_name), stat_value, i + global_step)


def get_wd(optimizer):
    return optimizer.param_groups[0]['weight_decay']
