# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path as osp
import sys
import time
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import Any, Tuple

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.models import InceptionOutputs

import nncf
from examples.common.paths import configure_paths
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.common.argparser import get_common_argument_parser
from examples.torch.common.argparser import parse_args
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import ExecutionMode
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import set_seed
from examples.torch.common.execution import start_worker
from examples.torch.common.export import export_model_after_ptq
from examples.torch.common.model_loader import MODEL_STATE_ATTR
from examples.torch.common.model_loader import extract_model_and_compression_states
from examples.torch.common.model_loader import load_model
from examples.torch.common.model_loader import load_resuming_checkpoint
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import MockDataset
from examples.torch.common.utils import NullContextManager
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import create_code_snapshot
from examples.torch.common.utils import get_run_name
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import is_staged_quantization
from examples.torch.common.utils import make_additional_checkpoints
from examples.torch.common.utils import print_args
from examples.torch.common.utils import write_metrics
from nncf.api.compression import CompressionStage
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.config.utils import is_accuracy_aware_training
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.initialization import register_default_init_args
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.structures import ExecutionParameters
from nncf.torch.utils import is_main_process
from nncf.torch.utils import safe_thread_call

model_names = sorted(
    name for name, val in models.__dict__.items() if name.islower() and not name.startswith("__") and callable(val)
)


def broadcast_initialized_parameters(quantized_model: NNCFNetwork):
    for module in quantized_model.modules():
        if isinstance(module, BaseQuantizer):
            module.broadcast_initialized_params()


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument("--dataset", help="Dataset to use.", choices=["imagenet", "cifar100", "cifar10"], default=None)
    parser.add_argument(
        "--test-every-n-epochs", default=1, type=int, help="Enables running validation every given number of epochs"
    )
    parser.add_argument(
        "--mixed-precision",
        dest="mixed_precision",
        help="Enables torch.cuda.amp autocasting during training and validation steps",
        action="store_true",
    )
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parse_args(parser, argv)
    config = create_sample_config(args, parser)

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config, get_run_name(config))
    copyfile(args.config, osp.join(config.log_dir, "config.json"))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    config.execution_mode = get_execution_mode(config)

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    if not is_staged_quantization(config):
        start_worker(main_worker, config)
    else:
        from examples.torch.classification.staged_quantization_worker import staged_quantization_main_worker

        start_worker(staged_quantization_main_worker, config)


def inception_criterion_fn(model_outputs: Any, target: Any, criterion: _Loss) -> torch.Tensor:
    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
    output, aux_outputs = model_outputs
    loss1 = criterion(output, target)
    loss2 = criterion(aux_outputs, target)
    return loss1 + 0.4 * loss2


def main_worker(current_gpu, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_main_process():
        configure_logging(logger, config)
        print_args(config)
    else:
        config.tb = None

    set_seed(config)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    model_name = config["model"]
    train_criterion_fn = inception_criterion_fn if "inception" in model_name else default_criterion_fn

    train_loader = train_sampler = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)
    is_export_only = "export" in config.mode and ("train" not in config.mode and "test" not in config.mode)

    # Data loading code
    train_dataset, val_dataset = create_datasets(config)
    train_loader, train_sampler, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)

    def train_steps_fn(loader, model, optimizer, compression_ctrl, train_steps):
        train_epoch(
            loader,
            model,
            criterion,
            train_criterion_fn,
            optimizer,
            compression_ctrl,
            0,
            config,
            train_iters=train_steps,
            log_training_info=False,
        )

    def validate_model_fn(model, eval_loader):
        top1, top5, loss = validate(eval_loader, model, criterion, config, log_validation_info=False)
        return top1, top5, loss

    def model_eval_fn(model):
        top1, _, _ = validate(val_loader, model, criterion, config)
        return top1

    execution_params = ExecutionParameters(config.cpu_only, config.current_gpu)

    nncf_config = register_default_init_args(
        nncf_config,
        init_loader,
        criterion=criterion,
        criterion_fn=train_criterion_fn,
        train_steps_fn=train_steps_fn,
        validate_fn=lambda *x: validate_model_fn(*x)[::2],
        autoq_eval_fn=lambda *x: validate_model_fn(*x)[1],
        val_loader=val_loader,
        model_eval_fn=model_eval_fn,
        device=config.device,
        execution_parameters=execution_params,
    )

    # create model
    model = load_model(
        model_name,
        pretrained=pretrained,
        num_classes=config.get("num_classes", 1000),
        model_params=config.get("model_params"),
        weights_path=config.get("weights"),
    )

    model.to(config.device)

    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        resuming_checkpoint = load_resuming_checkpoint(resuming_checkpoint_path)
    # Compression state loading is not possible yet for PTQ + training.
    # model_state_dict, compression_state = extract_model_and_compression_states(resuming_checkpoint)
    model_state_dict, _ = extract_model_and_compression_states(resuming_checkpoint)

    #####################PTQ##############################

    def transform_fn(data_item: Tuple[torch.Tensor, int], device: torch.device) -> torch.Tensor:
        images, _ = data_item
        return images.to(device)

    quantize_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )
    quantize_init_args: QuantizationRangeInitArgs = nncf_config.get_extra_struct(QuantizationRangeInitArgs)
    calibration_dataset = nncf.Dataset(quantize_data_loader, partial(transform_fn, device=quantize_init_args.device))

    model = nncf.quantize(model, calibration_dataset)
    # validate(val_loader, model, criterion, config)

    ######################################################
    if model_state_dict is not None:
        current_state_dict = model.state_dict()
        updated_weights = {}
        for k, v in model_state_dict.items():
            if k.startswith("_nncf"):
                current_shape = current_state_dict[k].shape
                updated_weights[k] = v.reshape(current_shape)
        model_state_dict.update(updated_weights)
        load_state(model, model_state_dict, is_resume=True)

    if is_export_only:
        export_model_after_ptq(model, config)
        return

    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        broadcast_initialized_parameters(model)

    # define optimizer
    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    best_acc1 = 0
    # optionally resume from a checkpoint
    if resuming_checkpoint_path is not None:
        if "train" in config.mode:
            config.start_epoch = resuming_checkpoint["epoch"]
            best_acc1 = resuming_checkpoint["best_acc1"]
            optimizer.load_state_dict(resuming_checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch: {}, best_acc1: {:.3f})".format(
                    resuming_checkpoint_path, resuming_checkpoint["epoch"], best_acc1
                )
            )
        else:
            logger.info("=> loaded checkpoint '{}'".format(resuming_checkpoint_path))

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    if "train" in config.mode:
        if is_accuracy_aware_training(config):
            raise RuntimeError("Accuracy aware training is not supported for QAT after PTQ.")
        train(
            config,
            model,
            criterion,
            train_criterion_fn,
            lr_scheduler,
            model_name,
            optimizer,
            train_loader,
            train_sampler,
            val_loader,
            best_acc1,
        )

    if "test" in config.mode:
        val_model = model
        validate(val_loader, val_model, criterion, config)

    if "export" in config.mode:
        export_model_after_ptq(model, config)


def train(
    config,
    model,
    criterion,
    criterion_fn,
    lr_scheduler,
    model_name,
    optimizer,
    train_loader,
    train_sampler,
    val_loader,
    best_acc1=0,
):
    best_compression_stage = CompressionStage.UNCOMPRESSED

    for epoch in range(config.start_epoch, config.epochs):
        # update compression scheduler state at the begin of the epoch

        if config.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, criterion, criterion_fn, optimizer, epoch, config)

        # Learning rate scheduling should be applied after optimizer’s update
        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_acc1)

        # compute compression algo statistics

        acc1 = best_acc1
        if epoch % config.test_every_n_epochs == 0:
            # evaluate on validation set
            acc1, _, _ = validate(val_loader, model, criterion, config, epoch=epoch)

        compression_stage = CompressionStage.FULLY_COMPRESSED
        # remember best acc@1, considering compression stage. If current acc@1 less then the best acc@1, checkpoint
        # still can be best if current compression stage is larger than the best one. Compression stages in ascending
        # order: UNCOMPRESSED, PARTIALLY_COMPRESSED, FULLY_COMPRESSED.
        is_best_by_accuracy = acc1 > best_acc1 and compression_stage == best_compression_stage
        is_best = is_best_by_accuracy or compression_stage > best_compression_stage
        if is_best:
            best_acc1 = acc1
        best_compression_stage = max(compression_stage, best_compression_stage)
        if is_main_process():
            if config.metrics_dump is not None:
                acc = best_acc1 / 100
                write_metrics(acc, config.metrics_dump)

            checkpoint_path = osp.join(config.checkpoint_save_dir, get_run_name(config) + "_last.pth")
            checkpoint = {
                "epoch": epoch + 1,
                "arch": model_name,
                MODEL_STATE_ATTR: model.state_dict(),
                "best_acc1": best_acc1,
                "acc1": acc1,
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, checkpoint_path)
            make_additional_checkpoints(checkpoint_path, is_best, epoch + 1, config)


def get_dataset(dataset_config, config, transform, is_train):
    if dataset_config == "imagenet":
        prefix = "train" if is_train else "val"
        return datasets.ImageFolder(osp.join(config.dataset_dir, prefix), transform)
    # For testing purposes
    num_images = config.get("num_mock_images", 1000)
    if dataset_config == "mock_32x32":
        return MockDataset(img_size=(32, 32), transform=transform, num_images=num_images)
    if dataset_config == "mock_299x299":
        return MockDataset(img_size=(299, 299), transform=transform, num_images=num_images)
    return create_cifar(config, dataset_config, is_train, transform)


def create_cifar(config, dataset_config, is_train, transform):
    create_cifar_fn = None
    if dataset_config in ["cifar100", "cifar100_224x224"]:
        create_cifar_fn = partial(CIFAR100, config.dataset_dir, train=is_train, transform=transform)
    if dataset_config == "cifar10":
        create_cifar_fn = partial(CIFAR10, config.dataset_dir, train=is_train, transform=transform)
    if create_cifar_fn:
        return safe_thread_call(partial(create_cifar_fn, download=True), partial(create_cifar_fn, download=False))
    return None


def create_datasets(config):
    dataset_config = config.dataset if config.dataset is not None else "imagenet"
    dataset_config = dataset_config.lower()
    assert dataset_config in [
        "imagenet",
        "cifar100",
        "cifar10",
        "cifar100_224x224",
        "mock_32x32",
        "mock_299x299",
    ], "Unknown dataset option"

    if dataset_config == "imagenet":
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    elif dataset_config in ["cifar100", "cifar100_224x224"]:
        normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2761))
    elif dataset_config == "cifar10":
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    elif dataset_config in ["mock_32x32", "mock_299x299"]:
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    input_info = FillerInputInfo.from_nncf_config(config)
    image_size = input_info.elements[0].shape[-1]
    size = int(image_size / 0.875)
    if dataset_config in ["cifar10", "cifar100_224x224", "cifar100"]:
        list_val_transforms = [transforms.ToTensor(), normalize]
        if dataset_config == "cifar100_224x224":
            list_val_transforms.insert(0, transforms.Resize(image_size))
        val_transform = transforms.Compose(list_val_transforms)

        list_train_transforms = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        if dataset_config == "cifar100_224x224":
            list_train_transforms.insert(0, transforms.Resize(image_size))
        train_transforms = transforms.Compose(list_train_transforms)
    elif dataset_config in ["mock_32x32", "mock_299x299"]:
        val_transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        val_transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    val_dataset = get_dataset(dataset_config, config, val_transform, is_train=False)
    train_dataset = get_dataset(dataset_config, config, train_transforms, is_train=True)

    return train_dataset, val_dataset


def create_data_loaders(config, train_dataset, val_dataset):
    pin_memory = config.execution_mode != ExecutionMode.CPU_ONLY

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    batch_size = int(config.batch_size)
    workers = int(config.workers)
    batch_size_val = int(config.batch_size_val) if config.batch_size_val is not None else int(config.batch_size)
    if config.execution_mode == ExecutionMode.MULTIPROCESSING_DISTRIBUTED:
        batch_size //= config.ngpus_per_node
        batch_size_val //= config.ngpus_per_node
        workers //= config.ngpus_per_node

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        drop_last=False,
    )

    train_sampler = None
    if config.distributed:
        sampler_seed = 0 if config.seed is None else config.seed
        dist_sampler_shuffle = config.seed is None
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, seed=sampler_seed, shuffle=dist_sampler_shuffle
        )

    train_shuffle = train_sampler is None and config.seed is None

    def create_train_data_loader(batch_size_):
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size_,
            shuffle=train_shuffle,
            num_workers=workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            drop_last=True,
        )

    train_loader = create_train_data_loader(batch_size)

    if config.batch_size_init:
        init_loader = create_train_data_loader(config.batch_size_init)
    else:
        init_loader = deepcopy(train_loader)
    return train_loader, train_sampler, val_loader, init_loader


def train_epoch(
    train_loader,
    model,
    criterion,
    criterion_fn,
    optimizer,
    epoch,
    config,
    train_iters=None,
    log_training_info=True,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    compression_losses = AverageMeter()
    criterion_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if train_iters is None:
        train_iters = len(train_loader)

    casting = autocast if config.mixed_precision else NullContextManager

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_ = input_.to(config.device)
        target = target.to(config.device)

        # compute output
        with casting():
            output = model(input_)
            criterion_loss = criterion_fn(output, target, criterion)

            # compute compression loss
            loss = criterion_loss

        if isinstance(output, InceptionOutputs):
            output = output.logits
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
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

        if i % config.print_freq == 0 and log_training_info:
            logger.info(
                "{rank}: "
                "Epoch: [{0}][{1}/{2}] "
                "Lr: {3:.3} "
                "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Data: {data_time.val:.3f} ({data_time.avg:.3f}) "
                "CE_loss: {ce_loss.val:.4f} ({ce_loss.avg:.4f}) "
                "CR_loss: {cr_loss.val:.4f} ({cr_loss.avg:.4f}) "
                "Loss: {loss.val:.4f} ({loss.avg:.4f}) "
                "Acc@1: {top1.val:.3f} ({top1.avg:.3f}) "
                "Acc@5: {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    get_lr(optimizer),
                    batch_time=batch_time,
                    data_time=data_time,
                    ce_loss=criterion_losses,
                    cr_loss=compression_losses,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    rank="{}:".format(config.rank) if config.multiprocessing_distributed else "",
                )
            )

        if is_main_process() and log_training_info:
            global_step = train_iters * epoch
            config.tb.add_scalar("train/learning_rate", get_lr(optimizer), i + global_step)
            config.tb.add_scalar("train/criterion_loss", criterion_losses.val, i + global_step)
            config.tb.add_scalar("train/compression_loss", compression_losses.val, i + global_step)
            config.tb.add_scalar("train/loss", losses.val, i + global_step)
            config.tb.add_scalar("train/top1", top1.val, i + global_step)
            config.tb.add_scalar("train/top5", top5.val, i + global_step)

        if i >= train_iters:
            break


def validate(val_loader, model, criterion, config, epoch=0, log_validation_info=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    casting = autocast if config.mixed_precision else NullContextManager
    with torch.no_grad():
        end = time.time()
        for i, (input_, target) in enumerate(val_loader):
            input_ = input_.to(config.device)
            target = target.to(config.device)

            # compute output
            with casting():
                output = model(input_)
                loss = default_criterion_fn(output, target, criterion)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_.size(0))
            top1.update(acc1, input_.size(0))
            top5.update(acc5, input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0 and log_validation_info:
                logger.info(
                    "{rank}"
                    "Test: [{0}/{1}] "
                    "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    "Loss: {loss.val:.4f} ({loss.avg:.4f}) "
                    "Acc@1: {top1.val:.3f} ({top1.avg:.3f}) "
                    "Acc@5: {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                        rank="{}:".format(config.rank) if config.multiprocessing_distributed else "",
                    )
                )

        if is_main_process() and log_validation_info:
            config.tb.add_scalar("val/loss", losses.avg, len(val_loader) * epoch)
            config.tb.add_scalar("val/top1", top1.avg, len(val_loader) * epoch)
            config.tb.add_scalar("val/top5", top5.avg, len(val_loader) * epoch)

            logger.info(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n".format(top1=top1, top5=top5))

        if is_main_process() and config.metrics_dump is not None:
            acc = top1.avg / 100
            write_metrics(acc, config.metrics_dump)

    return top1.avg, top5.avg, losses.avg


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
            correct_k = correct[:k].reshape(-1).sum(0, keepdim=True)
            res.append(correct_k.float().mul_(100.0 / batch_size).item())
        return res


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


if __name__ == "__main__":
    main(sys.argv[1:])
