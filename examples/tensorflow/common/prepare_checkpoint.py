"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.tensorflow.helpers.model_creation import create_compressed_model
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.sample_config import create_sample_config
from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.object_detection.models.model_selector import get_predefined_config
from examples.tensorflow.object_detection.models.model_selector import get_model_builder


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config_from_json = create_sample_config(args, parser)
    predefined_config = get_predefined_config(config_from_json.model)

    predefined_config.update(config_from_json)

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


def checkpoint_saver(config):
    """
    Load checkpoint and re-save it without optimizer (memory footprint is reduced)
    """
    model_builder = get_model_builder(config)
    model = model_builder.build_model()

    compression_ctrl, compress_model = create_compressed_model(model, config.nncf_config)

    checkpoint = tf.train.Checkpoint(model=compress_model, compression_ctrl=compression_ctrl)
    load_checkpoint(checkpoint, config.ckpt_path)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_save_dir, max_to_keep=None)
    save_path = checkpoint_manager.save()
    logger.info('Saved checkpoint: {}'.format(save_path))


def main(argv):
    parser = get_common_argument_parser(metrics_dump=False,
                                        weights=False,
                                        execution_args=False,
                                        batch_size=False,
                                        epochs=False,
                                        precision=False,
                                        dataset_dir=False,
                                        dataset_type=False,
                                        log_dir=False,
                                        save_checkpoint_freq=False,
                                        export_args=False,
                                        print_freq=False)

    config = get_config_from_argv(argv, parser)

    checkpoint_saver(config)


if __name__ == '__main__':
    main(sys.argv[1:])
