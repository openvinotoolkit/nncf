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
import os.path as osp
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from functools import partial
from shutil import copyfile
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import InceptionOutputs

from examples.common.argparser import get_common_argument_parser
from examples.common.example_logger import logger
from examples.common.execution import ExecutionMode, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.sample_config import SampleConfig, create_sample_config
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, make_additional_checkpoints, get_name, is_staged_quantization, print_statistics, \
    is_pretrained_model_requested, log_common_mlflow_params, SafeMLFLow, MockDataset, configure_device
from examples.common.utils import write_metrics
from nncf import create_compressed_model
from nncf.compression_method_api import CompressionLevel
from nncf.dynamic_graph.graph_builder import create_input_infos
from nncf.initialization import register_default_init_args, default_criterion_fn
from nncf.utils import safe_thread_call, is_main_process
from examples.classification.common import set_seed, load_resuming_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = create_sample_config(args, parser)

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config)
    copyfile(args.config, osp.join(config.log_dir, 'config.json'))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    config.execution_mode = get_execution_mode(config)

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    if not is_staged_quantization(config):
        start_worker(main_worker, config)
    else:
        from examples.classification.staged_quantization_worker import staged_quantization_main_worker
        start_worker(staged_quantization_main_worker, config)


def inception_criterion_fn(model_outputs: Any, target: Any, criterion: _Loss) -> torch.Tensor:
    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
    output, aux_outputs = model_outputs
    loss1 = criterion(output, target)
    loss2 = criterion(aux_outputs, target)
    return loss1 + 0.4 * loss2


# pylint:disable=too-many-branches
def main_worker(current_gpu, config: SampleConfig):
    configure_device(current_gpu, config)
    config.mlflow = SafeMLFLow(config)
    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    set_seed(config)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    model_name = config['model']
    train_criterion_fn = inception_criterion_fn if 'inception' in model_name else default_criterion_fn

    train_loader = train_sampler = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)

    if config.to_onnx is not None:
        assert pretrained or (resuming_checkpoint_path is not None)
    else:
        # Data loading code
        train_dataset, val_dataset = create_datasets(config)
        train_loader, train_sampler, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)

        def autoq_eval_fn(model, eval_loader):
            _, top5 = validate(eval_loader, model, criterion, config)
            return top5

        nncf_config = register_default_init_args(
            nncf_config, init_loader, criterion, train_criterion_fn,
            autoq_eval_fn, val_loader, config.device)

    # create model
    model = load_model(model_name,
                       pretrained=pretrained,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'),
                       weights_path=config.get('weights'))

    model.to(config.device)

    resuming_model_sd, resuming_checkpoint = load_resuming_checkpoint(resuming_checkpoint_path)
    compression_ctrl, model = create_compressed_model(model, nncf_config, resuming_state_dict=resuming_model_sd)

    if config.to_onnx:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        compression_ctrl.distributed()

    # define optimizer
    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    best_acc1 = 0
    # optionally resume from a checkpoint
    if resuming_checkpoint_path is not None:
        if config.mode.lower() == 'train' and config.to_onnx is None:
            config.start_epoch = resuming_checkpoint['epoch']
            best_acc1 = resuming_checkpoint['best_acc1']
            compression_ctrl.scheduler.load_state_dict(resuming_checkpoint['scheduler'])
            optimizer.load_state_dict(resuming_checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch: {}, best_acc1: {:.3f})"
                        .format(resuming_checkpoint_path, resuming_checkpoint['epoch'], best_acc1))
        else:
            logger.info("=> loaded checkpoint '{}'".format(resuming_checkpoint_path))

    log_common_mlflow_params(config)

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    if is_main_process():
        print_statistics(compression_ctrl.statistics())

    if config.mode.lower() == 'test':
        validate(val_loader, model, criterion, config)

    if config.mode.lower() == 'train':
        train(config, compression_ctrl, model, criterion, train_criterion_fn, lr_scheduler, model_name, optimizer,
              train_loader, train_sampler, val_loader, best_acc1)


def train(config, compression_ctrl, model, criterion, criterion_fn, lr_scheduler, model_name, optimizer,
          train_loader, train_sampler, val_loader, best_acc1=0):
    best_compression_level = CompressionLevel.NONE
    for epoch in range(config.start_epoch, config.epochs):
        # update compression scheduler state at the begin of the epoch
        compression_ctrl.scheduler.epoch_step()

        config.cur_epoch = epoch
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, criterion, criterion_fn, optimizer, compression_ctrl, epoch, config)

        # Learning rate scheduling should be applied after optimizer’s update
        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_acc1)

        # compute compression algo statistics
        stats = compression_ctrl.statistics()

        acc1 = best_acc1
        if epoch % config.test_every_n_epochs == 0:
            # evaluate on validation set
            acc1, _ = validate(val_loader, model, criterion, config)

        compression_level = compression_ctrl.compression_level()
        # remember best acc@1, considering compression level. If current acc@1 less then the best acc@1, checkpoint
        # still can be best if current compression level is bigger then best one. Compression levels in ascending
        # order: NONE, PARTIAL, FULL.
        is_best_by_accuracy = acc1 > best_acc1 and compression_level == best_compression_level
        is_best = is_best_by_accuracy or compression_level > best_compression_level
        if is_best:
            best_acc1 = acc1
        config.mlflow.safe_call('log_metric', "best_acc1", best_acc1)
        best_compression_level = max(compression_level, best_compression_level)
        acc = best_acc1 / 100
        if config.metrics_dump is not None:
            write_metrics(acc, config.metrics_dump)
        if is_main_process():
            print_statistics(stats)

            checkpoint_path = osp.join(config.checkpoint_save_dir, get_name(config) + '_last.pth')
            checkpoint = {
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'compression_level': compression_level,
                'acc1': acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': compression_ctrl.scheduler.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)
            make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)

            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    config.mlflow.safe_call('log_metric', 'compression/statistics/{0}'.format(key), value, epoch)
                    config.tb.add_scalar("compression/statistics/{0}".format(key), value, len(train_loader) * epoch)


def get_dataset(dataset_config, config, transform, is_train):
    if dataset_config == 'imagenet':
        prefix = 'train' if is_train else 'val'
        return datasets.ImageFolder(osp.join(config.dataset_dir, prefix), transform)
    if dataset_config == 'mock_32x32':
        # For testing purposes
        return MockDataset(img_size=(32, 32), transform=transform)
    return create_cifar(config, dataset_config, is_train, transform)


def create_cifar(config, dataset_config, is_train, transform):
    create_cifar_fn = None
    if dataset_config == 'cifar100':
        create_cifar_fn = partial(CIFAR100, config.dataset_dir, train=is_train, transform=transform)
    if dataset_config == 'cifar10':
        create_cifar_fn = partial(CIFAR10, config.dataset_dir, train=is_train, transform=transform)
    if create_cifar_fn:
        return safe_thread_call(partial(create_cifar_fn, download=True), partial(create_cifar_fn, download=False))
    return None


def create_datasets(config):
    dataset_config = config.dataset if config.dataset is not None else 'imagenet'
    dataset_config = dataset_config.lower()
    assert dataset_config in ['imagenet', 'cifar100', 'cifar10', 'mock_32x32'], "Unknown dataset option"

    if dataset_config == 'imagenet':
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    elif dataset_config == 'cifar100':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                         std=(0.2023, 0.1994, 0.2010))
    elif dataset_config in ['cifar10', 'mock_32x32']:
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                         std=(0.5, 0.5, 0.5))

    input_info_list = create_input_infos(config)
    image_size = input_info_list[0].shape[-1]
    size = int(image_size / 0.875)
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = get_dataset(dataset_config, config, val_transform, is_train=False)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = get_dataset(dataset_config, config, train_transforms, is_train=True)

    return train_dataset, val_dataset


def create_data_loaders(config, train_dataset, val_dataset):
    pin_memory = config.execution_mode != ExecutionMode.CPU_ONLY

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    batch_size = int(config.batch_size)
    workers = int(config.workers)
    if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        batch_size //= config.ngpus_per_node
        workers //= config.ngpus_per_node

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
        sampler=val_sampler, drop_last=True)

    train_sampler = None
    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def create_train_data_loader(batch_size_):
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size_, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True)

    train_loader = create_train_data_loader(batch_size)

    if config.batch_size_init:
        init_loader = create_train_data_loader(config.batch_size_init)
    else:
        init_loader = deepcopy(train_loader)
    if config.distributed:
        init_loader.num_workers = 0  # PyTorch multiprocessing dataloader issue WA

    return train_loader, train_sampler, val_loader, init_loader


def train_epoch(train_loader, model, criterion, criterion_fn, optimizer, compression_ctrl, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    compression_losses = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    compression_scheduler = compression_ctrl.scheduler

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        compression_scheduler.step()

        input_ = input_.to(config.device)
        target = target.to(config.device)

        # compute output
        output = model(input_)
        criterion_loss = criterion_fn(output, target, criterion)

        # compute compression loss
        compression_loss = compression_ctrl.loss()
        loss = criterion_loss + compression_loss

        if isinstance(output, InceptionOutputs):
            output = output.logits
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        comp_loss_val = compression_loss.item() if isinstance(compression_loss, torch.Tensor) else compression_loss
        compression_losses.update(comp_loss_val, input_.size(0))
        criterion_losses.update(criterion_loss.item(), input_.size(0))
        top1.update(acc1, input_.size(0))
        top5.update(acc5, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logger.info(
                '{rank}: '
                'Epoch: [{0}][{1}/{2}] '
                'Lr: {3:.3} '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                'CE_loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) '
                'CR_loss: {cr_loss.val:.4f} ({cr_loss.avg:.4f}) '
                'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), get_lr(optimizer), batch_time=batch_time,
                    data_time=data_time, ce_loss=criterion_losses, cr_loss=compression_losses,
                    loss=losses, top1=top1, top5=top5,
                    rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                ))

        if is_main_process():
            global_step = len(train_loader) * epoch
            config.tb.add_scalar("train/learning_rate", get_lr(optimizer), i + global_step)
            config.tb.add_scalar("train/criterion_loss", criterion_losses.avg, i + global_step)
            config.tb.add_scalar("train/compression_loss", compression_losses.avg, i + global_step)
            config.tb.add_scalar("train/loss", losses.avg, i + global_step)
            config.tb.add_scalar("train/top1", top1.avg, i + global_step)
            config.tb.add_scalar("train/top5", top5.avg, i + global_step)

            for stat_name, stat_value in compression_ctrl.statistics(quickly_collected_only=True).items():
                if isinstance(stat_value, (int, float)):
                    config.tb.add_scalar('train/statistics/{}'.format(stat_name), stat_value, i + global_step)


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.to(config.device)
            target = target.to(config.device)

            # compute output
            output = model(input_)
            loss = default_criterion_fn(output, target, criterion)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss, input_.size(0))
            top1.update(acc1, input_.size(0))
            top5.update(acc5, input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                logger.info(
                    '{rank}'
                    'Test: [{0}/{1}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                    'Acc@1: {top1.val:.3f} ({top1.avg:.3f}) '
                    'Acc@5: {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5,
                        rank='{}:'.format(config.rank) if config.multiprocessing_distributed else ''
                    ))

        if is_main_process():
            config.tb.add_scalar("val/loss", losses.avg, len(val_loader) * config.get('cur_epoch', 0))
            config.tb.add_scalar("val/top1", top1.avg, len(val_loader) * config.get('cur_epoch', 0))
            config.tb.add_scalar("val/top5", top5.avg, len(val_loader) * config.get('cur_epoch', 0))
            config.mlflow.safe_call('log_metric', "val/loss", float(losses.avg), config.get('cur_epoch', 0))
            config.mlflow.safe_call('log_metric', "val/top1", float(top1.avg), config.get('cur_epoch', 0))
            config.mlflow.safe_call('log_metric', "val/top5", float(top5.avg), config.get('cur_epoch', 0))

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'.format(top1=top1, top5=top5))

        acc = top1.avg / 100
        if config.metrics_dump is not None:
            write_metrics(acc, config.metrics_dump)

    return top1.avg, top5.avg


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    main(sys.argv[1:])
