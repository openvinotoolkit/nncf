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

from nncf.experimental.tensorflow.nncf_network import NNCFNetwork

import sys
import os.path as osp
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa

from nncf.config.utils import is_accuracy_aware_training
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow.helpers.model_manager import TFOriginalModelManager
from nncf.tensorflow.initialization import register_default_init_args
from nncf.tensorflow.utils.state import TFCompressionState

from examples.tensorflow.common.callbacks import get_callbacks
from examples.tensorflow.common.callbacks import get_progress_bar
from examples.tensorflow.common.distributed import get_distribution_strategy
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.model_loader import get_model
from examples.tensorflow.common.optimizer import build_optimizer
from examples.tensorflow.common.scheduler import build_scheduler
from examples.tensorflow.common.utils import create_code_snapshot
from examples.tensorflow.common.utils import get_saving_parameters
from examples.tensorflow.common.utils import print_args
from examples.tensorflow.common.utils import serialize_config
from examples.tensorflow.common.utils import serialize_cli_args
from examples.tensorflow.common.utils import write_metrics
from examples.tensorflow.common.utils import SummaryWriter

from examples.tensorflow.classification.main import get_argument_parser
from examples.tensorflow.classification.main import get_config_from_argv
from examples.tensorflow.classification.main import get_dataset_builders
from examples.tensorflow.classification.main import get_num_classes

from examples.tensorflow.classification.main import resume_from_checkpoint
from examples.tensorflow.classification.main import load_compression_state


def get_input_signature(config):
    input_info = config.get('input_info', {})
    if isinstance(input_info, dict):
        sample_size = input_info.get('sample_size', None)
    else:
        sample_size = input_info[0].get('sample_size', None) if input_info else None
    if not sample_size:
        raise RuntimeError('sample_size must be provided in configuration file')
    shape = [None] + list(sample_size[1:])
    return tf.TensorSpec(shape=shape, dtype=tf.float32)


def run(config):
    strategy = get_distribution_strategy(config)
    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    model_fn, model_params = get_model(config.model,
                                       input_shape=config.get('input_info', {}).get('sample_size', None),
                                       num_classes=config.get('num_classes', get_num_classes(config.dataset)),
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

    if is_accuracy_aware_training(config):
        with TFOriginalModelManager(model_fn, **model_params) as model:
            model.compile(metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc@1')])
            results = model.evaluate(
                validation_dataset,
                steps=validation_steps,
                return_dict=True)
            uncompressed_model_accuracy = 100 * results['acc@1']

    compression_state = None
    if resume_training:
        compression_state = load_compression_state(config.ckpt_path)

    with strategy.scope():
        model = NNCFNetwork(model_fn(**model_params), get_input_signature(nncf_config))

    compression_ctrl, compress_model = create_compressed_model(model, nncf_config, compression_state)

    with strategy.scope():
        compression_callbacks = create_compression_callbacks(compression_ctrl, log_dir=config.log_dir)

        scheduler = build_scheduler(
            config=config,
            steps_per_epoch=train_steps)
        optimizer = build_optimizer(
            config=config,
            scheduler=scheduler)

        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        compress_model.add_loss(compression_ctrl.loss)

        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5'),
            tfa.metrics.MeanMetricWrapper(loss_obj, name='ce_loss'),
            tfa.metrics.MeanMetricWrapper(compression_ctrl.loss, name='cr_loss')
        ]

        compress_model.compile(optimizer=optimizer,
                               loss=loss_obj,
                               metrics=metrics,
                               run_eagerly=config.get('eager_mode', False))

        checkpoint = tf.train.Checkpoint(model=compress_model,
                                         compression_state=TFCompressionState(compression_ctrl))

        initial_epoch = 0
        if resume_training:
            initial_epoch = resume_from_checkpoint(checkpoint=checkpoint,
                                                   ckpt_path=config.ckpt_path,
                                                   steps_per_epoch=train_steps)

    callbacks = get_callbacks(
        include_tensorboard=True,
        track_lr=True,
        profile_batch=0,
        initial_step=initial_epoch * train_steps,
        log_dir=config.log_dir,
        ckpt_dir=config.checkpoint_save_dir,
        checkpoint=checkpoint)

    callbacks.append(get_progress_bar(
        stateful_metrics=['loss'] + [metric.name for metric in metrics]))
    callbacks.extend(compression_callbacks)

    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': config.test_every_n_epochs,
    }

    if 'train' in config.mode:
        if is_accuracy_aware_training(config):
            logger.info('starting an accuracy-aware training loop...')
            result_dict_to_val_metric_fn = lambda results: 100 * results['acc@1']
            compress_model.accuracy_aware_fit(train_dataset,
                                              compression_ctrl,
                                              nncf_config=config.nncf_config,
                                              callbacks=callbacks,
                                              initial_epoch=initial_epoch,
                                              steps_per_epoch=train_steps,
                                              tensorboard_writer=SummaryWriter(config.log_dir,
                                                                               'accuracy_aware_training'),
                                              log_dir=config.log_dir,
                                              uncompressed_model_accuracy=uncompressed_model_accuracy,
                                              result_dict_to_val_metric_fn=result_dict_to_val_metric_fn,
                                              **validation_kwargs)
        else:
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
    pass


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)
    print_args(config)

    serialize_config(config.nncf_config, config.log_dir)
    serialize_cli_args(parser, argv, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[3]
    create_code_snapshot(nncf_root, osp.join(config.log_dir, 'snapshot.tar.gz'))
    if 'train' in config.mode or 'test' in config.mode:
        run(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
