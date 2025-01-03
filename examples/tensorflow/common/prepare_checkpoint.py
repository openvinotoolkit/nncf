# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import tensorflow as tf

import nncf
from examples.common.sample_config import create_sample_config
from examples.tensorflow.common.argparser import get_common_argument_parser
from examples.tensorflow.common.logger import logger
from examples.tensorflow.common.object_detection.checkpoint_utils import get_variables
from examples.tensorflow.object_detection.models.model_selector import get_model_builder as get_model_od_builder
from examples.tensorflow.object_detection.models.model_selector import get_predefined_config as get_predefined_od_config
from examples.tensorflow.segmentation.models.model_selector import get_model_builder as get_model_seg_builder
from examples.tensorflow.segmentation.models.model_selector import get_predefined_config as get_predefined_seg_config
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from nncf.tensorflow.utils.state import TFCompressionState
from nncf.tensorflow.utils.state import TFCompressionStateLoader


class ModelType:
    object_detection = "object_detection"
    segmentation = "segmentation"


def get_config_and_model_type_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config_from_json = create_sample_config(args, parser)
    if args.model_type == ModelType.object_detection:
        predefined_config = get_predefined_od_config(config_from_json.model)
    elif args.model_type == ModelType.segmentation:
        predefined_config = get_predefined_seg_config(config_from_json.model)
    else:
        raise nncf.ValidationError("Wrong model type specified")

    predefined_config.update(config_from_json)
    return predefined_config, args.model_type


def load_checkpoint(checkpoint, ckpt_path):
    logger.info("Load from checkpoint is enabled")
    if tf.io.gfile.isdir(ckpt_path):
        path_to_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        logger.info("Latest checkpoint: {}".format(path_to_checkpoint))
    else:
        path_to_checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + ".index") else None
        logger.info("Provided checkpoint: {}".format(path_to_checkpoint))

    if not path_to_checkpoint:
        logger.info("No checkpoint detected")
        return 0

    logger.info("Checkpoint file {} found and restoring from checkpoint".format(path_to_checkpoint))
    status = checkpoint.restore(path_to_checkpoint)
    status.expect_partial()
    logger.info("Completed loading from checkpoint")

    return None


def load_compression_state(ckpt_path: str):
    checkpoint = tf.train.Checkpoint(compression_state=TFCompressionStateLoader())
    load_checkpoint(checkpoint, ckpt_path)
    return checkpoint.compression_state.state


def od_checkpoint_saver(config):
    """
    Load object detection checkpoint and re-save it without optimizer (memory footprint is reduced).
    """
    model_builder = get_model_od_builder(config)
    model = model_builder.build_model()

    compression_state = load_compression_state(config.ckpt_path)
    compression_ctrl, compress_model = create_compressed_model(model, config.nncf_config, compression_state)

    checkpoint = tf.train.Checkpoint(model=compress_model, compression_state=TFCompressionState(compression_ctrl))
    load_and_save_checkpoint(checkpoint, config)


def seg_checkpoint_saver(config):
    """
    Load segmentation checkpoint and re-save it without optimizer (memory footprint is reduced).
    """
    model_builder = get_model_seg_builder(config)
    model = model_builder.build_model()

    compression_state = load_compression_state(config.ckpt_path)
    compression_ctrl, compress_model = create_compressed_model(model, config.nncf_config, compression_state)

    variables = get_variables(compress_model)
    checkpoint = tf.train.Checkpoint(
        variables=variables, compression_state=TFCompressionState(compression_ctrl), step=tf.Variable(0)
    )
    load_and_save_checkpoint(checkpoint, config)


def load_and_save_checkpoint(checkpoint, config):
    """
    Load checkpoint and re-save it.
    """
    load_checkpoint(checkpoint, config.ckpt_path)
    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_save_dir, max_to_keep=None)
    save_path = checkpoint_manager.save()
    logger.info("Saved checkpoint: {}".format(save_path))


def main(argv):
    parser = get_common_argument_parser(
        metrics_dump=False,
        resume_args=False,
        execution_args=False,
        epochs=False,
        precision=False,
        dataset_dir=False,
        dataset_type=False,
        log_dir=False,
        save_checkpoint_freq=False,
        export_args=False,
        print_freq=False,
    )
    parser.add_argument(
        "--model-type",
        choices=[ModelType.object_detection, ModelType.segmentation],
        help="Type of the model which checkpoint is being provided.",
        required=True,
    )

    parser.add_argument(
        "--resume",
        metavar="PATH",
        type=str,
        default=None,
        dest="ckpt_path",
        help="Specifies the path to the checkpoint which should be optimized.",
        required=True,
    )

    config, model_type = get_config_and_model_type_from_argv(argv, parser)

    if model_type == ModelType.object_detection:
        od_checkpoint_saver(config)
    if model_type == ModelType.segmentation:
        seg_checkpoint_saver(config)


if __name__ == "__main__":
    main(sys.argv[1:])
