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
import functools
import os
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np

from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop
from nncf.tensorflow import create_compressed_model
from nncf.tensorflow.helpers.model_manager import TFModelManager
from nncf.tensorflow.initialization import register_default_init_args
from nncf.common.utils.tensorboard import prepare_for_tensorboard
from nncf.config.utils import is_accuracy_aware_training
from nncf.config.structures import ModelEvaluationArgs
from nncf.tensorflow.utils.state import TFCompressionState
from nncf.tensorflow.utils.state import TFCompressionStateLoader

from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.common.distributed import get_distribution_strategy
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.object_detection.datasets.builder import COCODatasetBuilder
from examples.tensorflow.common.optimizer import build_optimizer
from examples.tensorflow.common.sample_config import create_sample_config
from examples.tensorflow.common.scheduler import build_scheduler
from examples.tensorflow.common.utils import SummaryWriter
from examples.tensorflow.common.utils import Timer
from examples.tensorflow.common.utils import print_args
from examples.tensorflow.common.utils import serialize_config
from examples.tensorflow.common.utils import serialize_cli_args
from examples.tensorflow.common.utils import create_code_snapshot
from examples.tensorflow.common.utils import configure_paths
from examples.tensorflow.common.utils import get_saving_parameters
from examples.tensorflow.common.utils import write_metrics
from examples.tensorflow.object_detection.models.model_selector import get_predefined_config
from examples.tensorflow.object_detection.models.model_selector import get_model_builder
from examples.tensorflow.common.utils import close_strategy_threadpool
from examples.tensorflow.common.utils import set_seed


def get_argument_parser():
    parser = get_common_argument_parser(precision=False,
                                        save_checkpoint_freq=False)

    parser.add_argument('--backbone-checkpoint',
                        default=None,
                        type=str,
                        help='Path to backbone checkpoint.')

    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config_from_json = create_sample_config(args, parser)
    predefined_config = get_predefined_config(config_from_json.model)

    predefined_config.update(config_from_json)
    configure_paths(predefined_config)

    return predefined_config


def get_dataset_builders(config, num_devices):
    train_builder = COCODatasetBuilder(config=config,
                                       is_train=True,
                                       num_devices=num_devices)

    val_builder = COCODatasetBuilder(config=config,
                                     is_train=False,
                                     num_devices=num_devices)

    return train_builder, val_builder


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


def resume_from_checkpoint(checkpoint_manager, ckpt_path, steps_per_epoch):
    if load_checkpoint(checkpoint_manager.checkpoint, ckpt_path) == 0:
        return 0
    optimizer = checkpoint_manager.checkpoint.optimizer
    initial_step = optimizer.iterations.numpy()
    initial_epoch = initial_step // steps_per_epoch

    logger.info('Resuming from epoch %d (global step %d)', initial_epoch, initial_step)
    return initial_epoch, initial_step


def load_compression_state(ckpt_path: str):
    checkpoint = tf.train.Checkpoint(compression_state=TFCompressionStateLoader())
    load_checkpoint(checkpoint, ckpt_path)
    return checkpoint.compression_state.state


def create_test_step_fn(strategy, model, predict_post_process_fn):
    """Creates a distributed test step"""

    def _test_step_fn(inputs):
        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        labels, prediction_outputs = predict_post_process_fn(labels, model_outputs)

        return labels, prediction_outputs

    @tf.function
    def test_step(dataset_inputs):
        labels, outputs = strategy.run(_test_step_fn, args=(dataset_inputs,))
        outputs = tf.nest.map_structure(strategy.experimental_local_results, outputs)
        labels = tf.nest.map_structure(strategy.experimental_local_results, labels)

        return labels, outputs

    return test_step


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


def train_epoch(train_step, compression_ctrl, epoch, initial_epoch, steps_per_epoch, optimizer, checkpoint_manager,
                train_dist_dataset, train_summary_writer, initial_step, print_freq, timer):
    compression_ctrl.scheduler.epoch_step(epoch)

    for step, x in enumerate(train_dist_dataset):
        if epoch == initial_epoch and step < initial_step % steps_per_epoch:
            continue
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

        if step % print_freq == 0:
            time = timer.toc(average=False)
            logger.info('Step: {}/{} Time: {:.3f} sec'.format(step, steps_per_epoch, time))
            logger.info('Training metric = {}'.format(train_metric_result))
            timer.tic()


def train(train_step, test_step, eval_metric, train_dist_dataset, test_dist_dataset, initial_epoch, initial_step,
    epochs, steps_per_epoch, checkpoint_manager, compression_ctrl, log_dir, optimizer, num_test_batches, print_freq):

    train_summary_writer = SummaryWriter(log_dir, 'train')
    validation_summary_writer = SummaryWriter(log_dir, 'validation')
    compression_summary_writer = SummaryWriter(log_dir, 'compression')

    timer = Timer()
    timer.tic()

    statistics = compression_ctrl.statistics()
    logger.info(statistics.to_str())

    logger.info('Training...')
    for epoch in range(initial_epoch, epochs):
        logger.info('Epoch: {}/{}'.format(epoch, epochs))

        train_epoch(train_step, compression_ctrl, epoch, initial_epoch, steps_per_epoch,
                    optimizer, checkpoint_manager, train_dist_dataset, train_summary_writer,
                    initial_step, print_freq, timer)

        test_metric_result = evaluate(test_step, eval_metric, test_dist_dataset, num_test_batches, print_freq)
        validation_summary_writer(metrics=test_metric_result, step=optimizer.iterations.numpy())
        eval_metric.reset_states()
        logger.info('Validation metric = {}'.format(test_metric_result))

        statistics = compression_ctrl.statistics()
        logger.info(statistics.to_str())
        statistics = {
            f'compression/statistics/{name}': value for name, value in prepare_for_tensorboard(statistics).items()
        }
        compression_summary_writer(metrics=statistics,
                                   step=optimizer.iterations.numpy())

    train_summary_writer.close()
    validation_summary_writer.close()
    compression_summary_writer.close()


def evaluate(test_step, metric, test_dist_dataset, num_batches, print_freq):
    """Runs evaluation steps and aggregate metrics"""
    timer = Timer()
    timer.tic()

    logger.info('Testing...')
    for batch_idx, x in enumerate(test_dist_dataset):
        labels, outputs = test_step(x)
        metric.update_state(labels, outputs)

        if batch_idx % print_freq == 0:
            time = timer.toc(average=False)
            logger.info('Predict for batch: {}/{} Time: {:.3f} sec'.format(batch_idx, num_batches, time))
            timer.tic()

    logger.info('Total time: {:.3f} sec'.format(timer.total_time))

    timer.reset()

    logger.info('Evaluating predictions...')
    timer.tic()
    result = metric.result()
    timer.toc(average=False)
    logger.info('Total time: {:.3f} sec'.format(timer.total_time))

    return result


def model_eval_fn(model, strategy, model_builder, test_dist_dataset, num_test_batches, config):
    test_step = create_test_step_fn(strategy, model, model_builder.post_processing)
    metric_result = evaluate(test_step, model_builder.eval_metrics(), test_dist_dataset,
                             num_test_batches, config.print_freq)
    return metric_result['AP']


def run(config):
    strategy = get_distribution_strategy(config)
    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    set_seed(config)

    # Create dataset
    train_builder, test_builder = get_dataset_builders(config, strategy.num_replicas_in_sync)
    train_dataset = train_builder.build()
    test_dataset = test_builder.build()
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # Training parameters
    epochs = config.epochs
    steps_per_epoch = train_builder.steps_per_epoch
    num_test_batches = test_builder.steps_per_epoch

    # Create model builder
    model_builder = get_model_builder(config)

    # Register additional parameters in the NNCFConfig for initialization
    # the compressed model during building
    nncf_config = config.nncf_config
    nncf_config = register_default_init_args(nncf_config=nncf_config,
                                             data_loader=train_dataset,
                                             batch_size=train_builder.global_batch_size)

    resume_training = config.ckpt_path is not None

    compression_state = None
    if resume_training:
        compression_state = load_compression_state(config.ckpt_path)

    with TFModelManager(model_builder.build_model,
                        config.nncf_config,
                        weights=config.get('weights', None)) as model:
        with strategy.scope():
            config.nncf_config.register_extra_structs(
                [ModelEvaluationArgs(eval_fn=functools.partial(model_eval_fn,
                                                               strategy=strategy,
                                                               model_builder=model_builder,
                                                               test_dist_dataset=test_dist_dataset,
                                                               num_test_batches=num_test_batches,
                                                               config=config))])
            compression_ctrl, compress_model = create_compressed_model(model, nncf_config, compression_state)
            scheduler = build_scheduler(
                config=config,
                steps_per_epoch=steps_per_epoch)

            optimizer = build_optimizer(
                config=config,
                scheduler=scheduler)

            eval_metric = model_builder.eval_metrics()
            loss_fn = model_builder.build_loss_fn(compress_model, compression_ctrl.loss)
            predict_post_process_fn = model_builder.post_processing

            checkpoint = tf.train.Checkpoint(model=compress_model,
                                             optimizer=optimizer,
                                             compression_state=TFCompressionState(compression_ctrl))
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_save_dir, max_to_keep=None)

            initial_epoch = initial_step = 0
            if resume_training:
                initial_epoch, initial_step = resume_from_checkpoint(checkpoint_manager,
                                                                     config.ckpt_path,
                                                                     steps_per_epoch)

    train_step = create_train_step_fn(strategy, compress_model, loss_fn, optimizer)
    test_step = create_test_step_fn(strategy, compress_model, predict_post_process_fn)

    if 'train' in config.mode:
        if is_accuracy_aware_training(config):
            train_summary_writer = SummaryWriter(config.log_dir, 'train')
            timer = Timer()
            timer.tic()

            def train_epoch_fn(compression_ctrl, model, epoch, **kwargs):
                train_step = create_train_step_fn(strategy, model, loss_fn, optimizer)
                train_epoch(train_step, compression_ctrl, epoch, initial_epoch, steps_per_epoch,
                            optimizer, checkpoint_manager, train_dist_dataset, train_summary_writer,
                            initial_step, config.print_freq, timer)

            def validate_fn(model, **kwargs):
                test_step = create_test_step_fn(strategy, model, predict_post_process_fn)
                metric_result = evaluate(test_step, eval_metric, test_dist_dataset,
                                         num_test_batches, config.print_freq)
                return metric_result['AP']

            acc_aware_training_loop = create_accuracy_aware_training_loop(nncf_config, compression_ctrl)
            compress_model = acc_aware_training_loop.run(compress_model,
                                                         train_epoch_fn=train_epoch_fn,
                                                         validate_fn=validate_fn,
                                                         tensorboard_writer=SummaryWriter(config.log_dir,
                                                                                          'accuracy_aware_training'),
                                                         log_dir=config.log_dir)
        else:
            train(train_step, test_step, eval_metric, train_dist_dataset, test_dist_dataset,
                  initial_epoch, initial_step, epochs, steps_per_epoch, checkpoint_manager,
                  compression_ctrl, config.log_dir, optimizer, num_test_batches, config.print_freq)

    statistics = compression_ctrl.statistics()
    logger.info(statistics.to_str())
    metric_result = evaluate(test_step, eval_metric, test_dist_dataset, num_test_batches, config.print_freq)
    logger.info('Validation metric = {}'.format(metric_result))

    if config.metrics_dump is not None:
        write_metrics(metric_result['AP'], config.metrics_dump)

    if 'export' in config.mode:
        save_path, save_format = get_saving_parameters(config)
        compression_ctrl.export_model(save_path, save_format)
        logger.info("Saved to {}".format(save_path))

    close_strategy_threadpool(strategy)


def export(config):
    model_builder = get_model_builder(config)
    model = model_builder.build_model(weights=config.get('weights', None))

    compression_state = None
    if config.ckpt_path:
        compression_state = load_compression_state(config.ckpt_path)

    compression_ctrl, compress_model = create_compressed_model(model, config.nncf_config, compression_state)

    if config.ckpt_path:
        checkpoint = tf.train.Checkpoint(model=compress_model,
                                         compression_state=TFCompressionState(compression_ctrl))
        load_checkpoint(checkpoint, config.ckpt_path)

    save_path, save_format = get_saving_parameters(config)
    compression_ctrl.export_model(save_path, save_format)
    logger.info("Saved to {}".format(save_path))


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)
    print_args(config)

    serialize_config(config.nncf_config, config.log_dir)
    serialize_cli_args(parser, argv, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[3]
    create_code_snapshot(nncf_root, os.path.join(config.log_dir, "snapshot.tar.gz"))

    if 'train' in config.mode or 'test' in config.mode:
        run(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
