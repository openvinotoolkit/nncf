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

import sys

import tensorflow as tf

from examples.tensorflow.common.utils import close_strategy_threadpool
from nncf.tensorflow import create_compressed_model
from nncf.tensorflow import register_default_init_args
from nncf.tensorflow.helpers.model_manager import TFModelManager
from nncf.tensorflow.utils.state import TFCompressionState
from nncf.tensorflow.utils.state import TFCompressionStateLoader

from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.common.distributed import get_distribution_strategy
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.object_detection.datasets.builder import COCODatasetBuilder
from examples.tensorflow.common.object_detection.checkpoint_utils import get_variables
from examples.tensorflow.common.sample_config import create_sample_config
from examples.tensorflow.common.sample_config import SampleConfig
from examples.tensorflow.common.utils import configure_paths
from examples.tensorflow.common.utils import get_saving_parameters
from examples.tensorflow.common.utils import print_args
from examples.tensorflow.common.utils import SummaryWriter
from examples.tensorflow.common.utils import write_metrics
from examples.tensorflow.common.utils import Timer
from examples.tensorflow.segmentation.models.model_selector import get_predefined_config
from examples.tensorflow.segmentation.models.model_selector import get_model_builder


def get_argument_parser():
    parser = get_common_argument_parser(mode=False,
                                        weights=False,
                                        epochs=False,
                                        precision=False,
                                        save_checkpoint_freq=False,
                                        to_h5=False,
                                        dataset_type=False)

    parser.add_argument(
                     '--mode',
                     '-m',
                     nargs='+',
                     choices=['train', 'test', 'export'],
                     default='train',
                     help='train: performs validation of a checkpoint that was saved during training '
                          '(use --checkpoint-save-dir to specify a path to the train-time checkpoint directory) ;'
                          ' test: tests the model checkpoint (use --resume to specify the checkpoint file itself);'
                          ' export: exports the model.')

    parser.add_argument(
        '--eval-timeout',
        default=None,
        type=int,
        help='The maximum number of seconds to wait between checkpoints. '
             'If left as None, then the process will wait indefinitely.'
    )

    parser.add_argument(
        '--weights',
        default=None,
        type=str,
        help='Path to pretrained weights in ckpt format.'
    )

    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    sample_config = SampleConfig(
        {'dataset_type': 'tfrecords'}
    )

    config_from_json = create_sample_config(args, parser)
    predefined_config = get_predefined_config(config_from_json.model)

    sample_config.update(predefined_config)
    sample_config.update(config_from_json)
    configure_paths(sample_config)

    return sample_config


def get_dataset_builders(config, num_devices):
    val_builder = COCODatasetBuilder(config=config,
                                     is_train=False,
                                     num_devices=num_devices)
    config_ = config.deepcopy()
    config_.batch_size = val_builder.batch_size
    calibration_builder = COCODatasetBuilder(config=config_,
                                             is_train=True,
                                             num_devices=1)
    return val_builder, calibration_builder


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


def load_compression_state(ckpt_path: str):
    checkpoint = tf.train.Checkpoint(compression_state=TFCompressionStateLoader())
    load_checkpoint(checkpoint, ckpt_path)
    return checkpoint.compression_state.state


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


def restore_compressed_model(config, strategy, model_builder, ckpt_path = None):
    compression_state = None
    if ckpt_path:
        compression_state = load_compression_state(ckpt_path)

    with TFModelManager(model_builder.build_model,
                        config.nncf_config,
                        weights=config.get('weights', None),
                        is_training=False) as model:
        with strategy.scope():
            compression_ctrl, compress_model = create_compressed_model(model,
                                                                       config.nncf_config,
                                                                       compression_state)

            variables = get_variables(compress_model)
            checkpoint = tf.train.Checkpoint(variables=variables,
                                             compression_state=TFCompressionState(compression_ctrl),
                                             step=tf.Variable(0))
            if ckpt_path:
                load_checkpoint(checkpoint, config.ckpt_path)

    return compression_ctrl, compress_model, checkpoint


def run_evaluation(config, eval_timeout=None):
    """Runs evaluation on checkpoint save directory"""
    strategy = get_distribution_strategy(config)
    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    validation_builder, calibration_builder = get_dataset_builders(config, strategy.num_replicas_in_sync)
    calibration_dataset = calibration_builder.build()
    val_dataset = validation_builder.build()
    num_batches = validation_builder.steps_per_epoch
    test_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    config.nncf_config = register_default_init_args(nncf_config=config.nncf_config,
                                                    data_loader=calibration_dataset,
                                                    batch_size=validation_builder.global_batch_size)

    # We use `model_batch_size` to create input layer for model
    config.model_batch_size = validation_builder.batch_size

    model_builder = get_model_builder(config)
    eval_metric = model_builder.eval_metrics()
    predict_post_process_fn = model_builder.post_processing

    if 'test' in config.mode:
        compression_ctrl, compress_model, _ = restore_compressed_model(config, strategy, model_builder,
                                                                       config.ckpt_path)
        test_step = create_test_step_fn(strategy, compress_model, predict_post_process_fn)

        statistics = compression_ctrl.statistics()
        logger.info(statistics.to_str())
        metric_result = evaluate(test_step, eval_metric, test_dist_dataset, num_batches, config.print_freq)
        eval_metric.reset_states()
        logger.info('Test metric = {}'.format(metric_result))

        if 'export' in config.mode:
            save_path, save_format = get_saving_parameters(config)
            compression_ctrl.export_model(save_path, save_format)
            logger.info("Saved to {}".format(save_path))

    elif 'train' in config.mode:
        validation_summary_writer = SummaryWriter(config.log_dir, 'validation')

        is_first_checkpoint = True
        for checkpoint_path in tf.train.checkpoints_iterator(config.checkpoint_save_dir, config.eval_timeout):
            if is_first_checkpoint:
                is_first_checkpoint = False
                _, compress_model, checkpoint = restore_compressed_model(config, strategy, model_builder,
                                                                         checkpoint_path)
                test_step = create_test_step_fn(strategy, compress_model, predict_post_process_fn)
            else:
                checkpoint.restore(checkpoint_path).expect_partial()

            logger.info('Checkpoint file {} found and restoring from checkpoint'.format(checkpoint_path))
            logger.info('Checkpoint step: {}'.format(checkpoint.step.numpy()))
            metric_result = evaluate(test_step, eval_metric, test_dist_dataset, num_batches, config.print_freq)

            current_step = checkpoint.step.numpy()
            validation_summary_writer(metrics=metric_result, step=current_step)

            eval_metric.reset_states()
            logger.info('Validation metric = {}'.format(metric_result))

        validation_summary_writer.close()

    if config.metrics_dump is not None:
        write_metrics(metric_result['AP'], config.metrics_dump)

    close_strategy_threadpool(strategy)


def export(config):
    model_builder = get_model_builder(config)

    strategy = tf.distribute.get_strategy()
    compression_ctrl, _, _ = restore_compressed_model(config, strategy, model_builder, config.ckpt_path)

    save_path, save_format = get_saving_parameters(config)
    compression_ctrl.export_model(save_path, save_format)
    logger.info("Saved to {}".format(save_path))


def main(argv):
    tf.get_logger().setLevel('INFO')
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)
    print_args(config)

    if config.dataset_type != 'tfrecords':
        raise RuntimeError('The train.py does not support TensorFlow Datasets (TFDS). '
                           'Please use TFRecords.')

    if 'train' in config.mode or 'test' in config.mode:
        run_evaluation(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
