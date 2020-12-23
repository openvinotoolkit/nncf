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
import tensorflow as tf

from nncf import create_compressed_model
from nncf.configs.config import Config
from nncf.tensorflow.helpers.model_manager import TFOriginalModelManager
from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.segmentation.models.model_selector import get_predefined_config, get_model_builder
from examples.tensorflow.common.object_detection.datasets.builder import COCODatasetBuilder
from examples.tensorflow.common.distributed import get_distribution_strategy, get_strategy_scope
from examples.tensorflow.common.utils import configure_paths, get_saving_parameters
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.utils import SummaryWriter


def get_argument_parser():
    parser = get_common_argument_parser()

    parser.add_argument(
        '--mode',
        '-m',
        nargs='+',
        choices=['train', 'test', 'export'],
        default='train',
        help='train: performs validation during training; test: tests the model; export: exports the model.'
    )

    parser.add_argument(
        '--eval-timeout',
        default=None,
        type=int,
        help='The maximum number of seconds to wait between checkpoints. '
             'If left as None, then the process will wait indefinitely.'
    )

    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config_from_json = Config.from_json(args.config)
    predefined_config = get_predefined_config(config_from_json.model)

    predefined_config.update(config_from_json)
    predefined_config.update_from_args(args, parser)
    configure_paths(predefined_config)

    return predefined_config


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


def evaluate(test_step, metric, test_dist_dataset):
    """Runs evaluation steps and aggregate metrics"""
    for x in test_dist_dataset:
        labels, outputs = test_step(x)
        metric.update_state(labels, outputs)

    return metric.result()


def create_test_step_fn(strategy, model, predict_post_process_fn):
    """Creates a distributed test step"""

    def _test_step_fn(inputs):
        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        if predict_post_process_fn:
            labels, prediction_outputs = predict_post_process_fn(labels, model_outputs)

        return labels, prediction_outputs

    @tf.function
    def test_step(dataset_inputs):
        labels, outputs = strategy.run(_test_step_fn, args=(dataset_inputs,))
        outputs = tf.nest.map_structure(strategy.experimental_local_results, outputs)
        labels = tf.nest.map_structure(strategy.experimental_local_results, labels)

        return labels, outputs

    return test_step


def run_evaluation(config, eval_timeout=None):
    """Runs evaluation on checkpoint save directory"""

    strategy = get_distribution_strategy(config)
    strategy_scope = get_strategy_scope(strategy)

    num_devices = strategy.num_replicas_in_sync if strategy else 1
    dataset_builder = COCODatasetBuilder(config=config, is_train=False, num_devices=num_devices)
    dataset = dataset_builder.build()
    test_dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # We use `model_batch_size` to create input layer for model
    config.model_batch_size = dataset_builder.batch_size

    model_builder = get_model_builder(config)

    with TFOriginalModelManager(model_builder.build_model,
                                weights=config.get('weights', None),
                                is_training=False) as model:
        with strategy_scope:
            compression_ctrl, compress_model = create_compressed_model(model, config)
            checkpoint = tf.train.Checkpoint(model=compress_model, step=tf.Variable(0))
            eval_metric = model_builder.eval_metrics()
            predict_post_process_fn = model_builder.post_processing

    test_step = create_test_step_fn(strategy, compress_model, predict_post_process_fn)

    if 'test' in config.mode:
        if config.ckpt_path:
            load_checkpoint(checkpoint, config.ckpt_path)

        metric_result = evaluate(test_step, eval_metric, test_dist_dataset)
        eval_metric.reset_states()
        logger.info('Test metric = {}'.format(metric_result))

        if 'export' in config.mode:
            save_path, save_format = get_saving_parameters(config)
            compression_ctrl.export_model(save_path, save_format)
            logger.info("Saved to {}".format(save_path))

    elif 'train' in config.mode:
        validation_summary_writer = SummaryWriter(config.log_dir, 'validation')
        checkpoint_dir = config.checkpoint_save_dir
        eval_timeout = config.eval_timeout

        for checkpoint_path in tf.train.checkpoints_iterator(checkpoint_dir, timeout=eval_timeout):
            status = checkpoint.restore(checkpoint_path)
            status.expect_partial()
            logger.info('Checkpoint file {} found and restoring from checkpoint'.format(checkpoint_path))
            logger.info('Checkpoint step: {}'.format(checkpoint.step.numpy()))
            logger.info('Evaluation...')
            metric_result = evaluate(test_step, eval_metric, test_dist_dataset)

            current_step = checkpoint.step.numpy()
            validation_summary_writer(metrics=metric_result, step=current_step)

            eval_metric.reset_states()
            logger.info('Validation metric = {}'.format(metric_result))

        validation_summary_writer.close()


def main(argv):
    tf.get_logger().setLevel('INFO')
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)
    run_evaluation(config)


if __name__ == '__main__':
    main(sys.argv[1:])
