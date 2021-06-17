"""
 Copyright (c) 2020 Intel Corporation
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

import sys
import os.path as osp
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa

from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from nncf.tensorflow.helpers.model_manager import TFOriginalModelManager
from nncf.tensorflow.initialization import register_default_init_args

from examples.tensorflow.classification.datasets.builder import DatasetBuilder
from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.common.callbacks import get_callbacks
from examples.tensorflow.common.callbacks import get_progress_bar
from examples.tensorflow.common.distributed import get_distribution_strategy
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.model_loader import get_model
from examples.tensorflow.common.optimizer import build_optimizer
from examples.tensorflow.common.sample_config import create_sample_config
from examples.tensorflow.common.scheduler import build_scheduler
from examples.tensorflow.common.utils import configure_paths
from examples.tensorflow.common.utils import create_code_snapshot
from examples.tensorflow.common.utils import get_saving_parameters
from examples.tensorflow.common.utils import serialize_config
from examples.tensorflow.common.utils import write_metrics


def get_argument_parser():
    parser = get_common_argument_parser(precision=False,
                                        save_checkpoint_freq=False,
                                        print_freq=False)

    parser.add_argument(
        '--mode',
        '-m',
        nargs='+',
        choices=['train', 'test', 'export'],
        default='train',
        help='train: performs training and validation; test: tests the model; export: exports the model.'
    )
    parser.add_argument(
        '--dataset',
        help='Dataset to use.',
        choices=['imagenet', 'cifar100', 'cifar10'],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pretrained models from the tf.keras.applications",
        action="store_true",
    )
    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)
    config = create_sample_config(args, parser)
    configure_paths(config)
    return config


def get_dataset_builders(config, num_devices, one_hot=True):
    image_size = config.input_info.sample_size[-2]

    train_builder = DatasetBuilder(
        config,
        image_size=image_size,
        num_devices=num_devices,
        one_hot=one_hot,
        is_train=True)

    val_builder = DatasetBuilder(
        config,
        image_size=image_size,
        num_devices=num_devices,
        one_hot=one_hot,
        is_train=False)

    return [train_builder, val_builder]


def load_checkpoint(checkpoint, ckpt_path):
    logger.info('Load from checkpoint is enabled.')
    if tf.io.gfile.isdir(ckpt_path):
        path_to_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        logger.info('Latest checkpoint: {}'.format(path_to_checkpoint))
    else:
        path_to_checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + '.index') else None
        logger.info('Provided checkpoint: {}'.format(path_to_checkpoint))

    if not path_to_checkpoint:
        logger.info('No checkpoint detected.')
        return 0

    logger.info('Checkpoint file {} found and restoring from checkpoint'
                .format(path_to_checkpoint))

    status = checkpoint.restore(path_to_checkpoint)
    status.expect_partial()
    logger.info('Completed loading from checkpoint.')
    return None


def resume_from_checkpoint(checkpoint, ckpt_path, steps_per_epoch):
    if load_checkpoint(checkpoint, ckpt_path) == 0:
        return 0

    initial_step = checkpoint.model.optimizer.iterations.numpy()
    initial_epoch = initial_step // steps_per_epoch

    logger.info('Resuming from epoch %d', initial_epoch)
    return initial_epoch


def run(config):
    strategy = get_distribution_strategy(config)
    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    model_fn, model_params = get_model(config.model,
                                       input_shape=config.get('input_info', {}).get('sample_size', None),
                                       num_classes=config.get('num_classes', 1000),
                                       pretrained=config.get('pretrained', False),
                                       weights=config.get('weights', None))

    builders = get_dataset_builders(config, strategy.num_replicas_in_sync)
    datasets = [builder.build() for builder in builders]

    train_builder, validation_builder = builders
    train_dataset, validation_dataset = datasets

    nncf_config = config.nncf_config
    nncf_config = register_default_init_args(nncf_config=nncf_config,
                                             data_loader=train_dataset,
                                             batch_size=train_builder.global_batch_size)

    train_epochs = config.epochs
    train_steps = train_builder.steps_per_epoch
    validation_steps = validation_builder.steps_per_epoch

    resume_training = config.ckpt_path is not None

    with TFOriginalModelManager(model_fn, **model_params) as model:
        with strategy.scope():
            compression_ctrl, compress_model = create_compressed_model(model,
                                                                       nncf_config,
                                                                       should_init=not resume_training)
            compression_callbacks = create_compression_callbacks(compression_ctrl,
                                                                 log_dir=config.log_dir)

            scheduler = build_scheduler(
                config=config,
                steps_per_epoch=train_steps)
            optimizer = build_optimizer(
                config=config,
                scheduler=scheduler)

            loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5'),
                tfa.metrics.MeanMetricWrapper(loss_obj, name='ce_loss'),
                tfa.metrics.MeanMetricWrapper(compression_ctrl.loss, name='cr_loss')
            ]

            compress_model.add_loss(compression_ctrl.loss)
            compress_model.compile(optimizer=optimizer,
                                   loss=loss_obj,
                                   metrics=metrics,
                                   run_eagerly=config.get('eager_mode', False))

            compress_model.summary()

            checkpoint = tf.train.Checkpoint(model=compress_model, compression_ctrl=compression_ctrl)

            initial_epoch = 0
            if resume_training:
                initial_epoch = resume_from_checkpoint(checkpoint=checkpoint,
                                                       ckpt_path=config.ckpt_path,
                                                       steps_per_epoch=train_steps)

    callbacks = get_callbacks(
        include_tensorboard=True,
        track_lr=True,
        write_model_weights=False,
        initial_step=initial_epoch * train_steps,
        model_dir=config.log_dir,
        ckpt_dir=config.checkpoint_save_dir,
        checkpoint=checkpoint)

    callbacks.append(get_progress_bar(
        stateful_metrics=['loss'] + [metric.name for metric in metrics]))
    callbacks.extend(compression_callbacks)

    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': 1,
    }

    if 'train' in config.mode:
        logger.info('training...')
        compress_model.fit(
            train_dataset,
            epochs=train_epochs,
            steps_per_epoch=train_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            **validation_kwargs)

    logger.info('evaluation...')
    statistics = compression_ctrl.statistics()
    logger.info(statistics.to_str())
    results = compress_model.evaluate(
        validation_dataset,
        steps=validation_steps,
        callbacks=[get_progress_bar(
            stateful_metrics=['loss'] + [metric.name for metric in metrics])],
        verbose=1)

    if config.metrics_dump is not None:
        write_metrics(results[1], config.metrics_dump)

    if 'export' in config.mode:
        save_path, save_format = get_saving_parameters(config)
        compression_ctrl.export_model(save_path, save_format)
        logger.info('Saved to {}'.format(save_path))


def export(config):
    model, model_params = get_model(config.model,
                                    input_shape=config.get('input_info', {}).get('sample_size', None),
                                    num_classes=config.get('num_classes', 1000),
                                    pretrained=config.get('pretrained', False),
                                    weights=config.get('weights', None))
    model = model(**model_params)
    compression_ctrl, compress_model = create_compressed_model(model,
                                                               config.nncf_config,
                                                               should_init=False)

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5')
    ]
    loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    compress_model.compile(loss=loss_obj,
                           metrics=metrics)
    compress_model.summary()

    checkpoint = tf.train.Checkpoint(model=compress_model, compression_ctrl=compression_ctrl)

    if config.ckpt_path is not None:
        load_checkpoint(checkpoint=checkpoint,
                        ckpt_path=config.ckpt_path)

    save_path, save_format = get_saving_parameters(config)
    compression_ctrl.export_model(save_path, save_format)
    logger.info('Saved to {}'.format(save_path))


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)

    serialize_config(config, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[3]
    create_code_snapshot(nncf_root, osp.join(config.log_dir, 'snapshot.tar.gz'))
    if 'train' in config.mode or 'test' in config.mode:
        run(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
