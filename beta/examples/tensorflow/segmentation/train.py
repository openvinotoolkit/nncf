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

import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np

from beta.nncf import create_compressed_model
from beta.nncf.configs.config import Config
from beta.nncf.helpers.utils import print_statistics
from beta.nncf.tensorflow.helpers.model_manager import TFOriginalModelManager

from beta.examples.tensorflow.common.logger import logger
from beta.examples.tensorflow.common.argparser import get_common_argument_parser
from beta.examples.tensorflow.common.distributed import get_distribution_strategy
from beta.examples.tensorflow.common.object_detection.checkpoint_utils import get_variables
from beta.examples.tensorflow.common.object_detection.datasets.builder import COCODatasetBuilder
from beta.examples.tensorflow.common.optimizer import build_optimizer
from beta.examples.tensorflow.common.scheduler import build_scheduler
from beta.examples.tensorflow.common.utils import configure_paths
from beta.examples.tensorflow.common.utils import create_code_snapshot
from beta.examples.tensorflow.common.utils import serialize_config
from beta.examples.tensorflow.common.utils import SummaryWriter
from beta.examples.tensorflow.segmentation.models.model_selector import get_predefined_config
from beta.examples.tensorflow.segmentation.models.model_selector import get_model_builder


def get_argument_parser():
    parser = get_common_argument_parser(weights=False,
                                        precision=False,
                                        save_checkpoint_freq=False,
                                        export_args=False,
                                        print_freq=False,
                                        dataset_type=False)

    parser.add_argument('--backbone-checkpoint',
                        default=None,
                        type=str,
                        help='Path to backbone checkpoint.')

    parser.add_argument('--weights',
                        default=None,
                        type=str,
                        help='Path to pretrained weights in ckpt format.')

    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    sample_config = Config(
        {'dataset_type': 'tfrecords'}
    )

    config_from_json = Config.from_json(args.config)
    predefined_config = get_predefined_config(config_from_json.model)

    sample_config.update(predefined_config)
    sample_config.update(config_from_json)
    sample_config.update_from_args(args, parser)
    configure_paths(sample_config)

    return sample_config


def get_dataset_builders(config, strategy):
    if config.dataset_type != 'tfrecords':
        raise RuntimeError('The train.py does not support TensorFlow Datasets (TFDS). '
                           'Please use TFRecords.')

    num_devices = strategy.num_replicas_in_sync if strategy else 1

    train_builder = COCODatasetBuilder(config=config,
                                       is_train=True,
                                       num_devices=num_devices)

    config_ = config.deepcopy()
    config_.batch_size = train_builder.batch_size
    calibration_builder = COCODatasetBuilder(config=config_,
                                             is_train=True,
                                             num_devices=1)

    return train_builder, calibration_builder


def load_checkpoint(checkpoint, ckpt_path):
    logger.info('Load from checkpoint is enabled')
    if tf.io.gfile.isdir(ckpt_path):
        path_to_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        logger.info('Latest checkpoint: {}'.format(path_to_checkpoint))
    else:
        path_to_checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + '.index') else None
        logger.info('Provided checkpoint: {}'.format(path_to_checkpoint))

    if not path_to_checkpoint:
        logger.info('No checkpoint detected')
        return 0

    logger.info('Checkpoint file {} found and restoring from checkpoint'.format(path_to_checkpoint))
    status = checkpoint.restore(path_to_checkpoint)
    status.expect_partial()
    logger.info('Completed loading from checkpoint')

    return None


def resume_from_checkpoint(checkpoint_manager, compression_ctrl, ckpt_path, steps_per_epoch):
    if load_checkpoint(checkpoint_manager.checkpoint, ckpt_path) == 0:
        return 0
    optimizer = checkpoint_manager.checkpoint.optimizer
    initial_step = optimizer.iterations.numpy()
    initial_epoch = initial_step // steps_per_epoch
    compression_ctrl.scheduler.load_state(initial_step, steps_per_epoch)
    logger.info('Resuming from epoch %d (global step %d)', initial_epoch, initial_step)
    return initial_epoch, initial_step


def create_train_step_fn(strategy, model, loss_fn, optimizer):
    """Creates a distributed training step"""

    def _train_step_fn(inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            all_losses = loss_fn(labels, outputs)
            losses = {}
            for k, v in all_losses.items():
                losses[k] = tf.reduce_mean(v)
            per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync

        grads = tape.gradient(per_replica_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    @tf.function
    def train_step(dataset_inputs):
        per_replica_losses = strategy.run(_train_step_fn, args=(dataset_inputs,))
        losses = tf.nest.map_structure(lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None),
                                       per_replica_losses)
        return losses

    return train_step


def train(train_step, train_dist_dataset, initial_epoch, initial_step,
          epochs, steps_per_epoch, checkpoint_manager, compression_ctrl, log_dir, optimizer):

    train_summary_writer = SummaryWriter(log_dir, 'train')
    compression_summary_writer = SummaryWriter(log_dir, 'compression')

    logger.info('Training started')
    for epoch in range(initial_epoch, epochs):
        logger.info('Epoch {}/{}'.format(epoch, epochs))
        compression_ctrl.scheduler.epoch_step(epoch)

        for step, x in enumerate(train_dist_dataset):
            if epoch == initial_epoch and step < initial_step % steps_per_epoch:
                continue

            checkpoint_manager.checkpoint.step.assign_add(1)

            if step == steps_per_epoch:
                save_path = checkpoint_manager.save()
                logger.info('Saved checkpoint for epoch={}: {}'.format(epoch, save_path))
                break

            compression_ctrl.scheduler.step()
            train_loss = train_step(x)
            train_metric_result = tf.nest.map_structure(lambda s: s.numpy().astype(float), train_loss)

            if np.isnan(train_metric_result['total_loss']):
                raise ValueError('total loss is NaN')

            train_metric_result.update({'learning_rate': optimizer.lr(optimizer.iterations).numpy()})

            train_summary_writer(metrics=train_metric_result, step=optimizer.iterations.numpy())

            if step % 100 == 0:
                logger.info('Step {}/{}'.format(step, steps_per_epoch))
                logger.info('Training metric = {}'.format(train_metric_result))

        statistics = compression_ctrl.statistics()
        print_statistics(statistics)
        statistics = {'compression/statistics/' + key: value
                      for key, value in statistics.items()
                      if isinstance(value, (int, float))}
        compression_summary_writer(metrics=statistics,
                                   step=optimizer.iterations.numpy())

    train_summary_writer.close()
    compression_summary_writer.close()


def run_train(config):
    strategy = get_distribution_strategy(config)

    # Create dataset
    builders = get_dataset_builders(config, strategy)
    datasets = [builder.build() for builder in builders]
    train_builder, _ = builders
    train_dataset, calibration_dataset = datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    # Training parameters
    epochs = config.epochs
    steps_per_epoch = train_builder.steps_per_epoch

    # We use `model_batch_size` to create input layer for model
    config.model_batch_size = train_builder.batch_size

    # Create model builder
    model_builder = get_model_builder(config)

    with TFOriginalModelManager(model_builder.build_model,
                                weights=config.get('weights', None),
                                is_training=True) as model:
        with strategy.scope():
            compression_ctrl, compress_model = create_compressed_model(model, config)

            scheduler = build_scheduler(
                config=config,
                epoch_size=train_builder.num_examples,
                batch_size=train_builder.global_batch_size,
                steps=steps_per_epoch)

            optimizer = build_optimizer(
                config=config,
                scheduler=scheduler)

            loss_fn = model_builder.build_loss_fn()

            variables = get_variables(compress_model)
            checkpoint = tf.train.Checkpoint(variables=variables, optimizer=optimizer, step=tf.Variable(0))
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_save_dir, max_to_keep=None)

            initial_epoch = initial_step = 0
            if config.ckpt_path:
                initial_epoch, initial_step = resume_from_checkpoint(checkpoint_manager,
                                                                     compression_ctrl,
                                                                     config.ckpt_path,
                                                                     steps_per_epoch)
            else:
                logger.info('Initialization...')
                compression_ctrl.initialize(dataset=calibration_dataset)

    train_step = create_train_step_fn(strategy, compress_model, loss_fn, optimizer)

    logger.info('Training...')
    train(train_step, train_dist_dataset, initial_epoch, initial_step,
          epochs, steps_per_epoch, checkpoint_manager, compression_ctrl, config.log_dir, optimizer)

    logger.info('Compression statistics')
    print_statistics(compression_ctrl.statistics())


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)

    serialize_config(config, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[2]
    create_code_snapshot(nncf_root, os.path.join(config.log_dir, "snapshot.tar.gz"))

    run_train(config)


if __name__ == '__main__':
    main(sys.argv[1:])
