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

from nncf.configs.config import CustomArgumentParser


def get_common_argument_parser(**flags):
    """Defines command-line arguments, and parses them.

    """
    parser = CustomArgumentParser()

    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with task/model-specific parameters.',
        required=True)

    model_init_mode = parser.add_mutually_exclusive_group()

    if flags.get('resume', True):
        model_init_mode.add_argument(
            '--resume',
            metavar='PATH',
            type=str,
            default=None,
            dest='ckpt_path',
            help='Specifies the path to the checkpoint to resume training, test or export model '
                'from the defined checkpoint or folder with checkpoints to resume training, test '
                'or export from the last checkpoint.')

    if flags.get('weights', True):
        model_init_mode.add_argument(
            '--weights',
            default=None,
            type=str,
            help='Path to pretrained weights in H5 format.')

    if flags.get('checkpoint_save_dir', True):
        parser.add_argument(
            '--checkpoint-save-dir',
            metavar='PATH',
            type=str,
            default=None,
            help='Specifies the directory for the trained model checkpoints to be saved to.')

    execution_type = parser.add_mutually_exclusive_group()

    if flags.get('gpu_id', True):
        execution_type.add_argument(
            '--gpu-id',
            type=int,
            metavar='N',
            help='The ID of the GPU training will be performed on, without any parallelization.')

    if flags.get('cpu_only', True):
        execution_type.add_argument(
            '--cpu-only',
            action='store_true',
            help='Specifies that the computation should be performed using CPU only.')

    # Hyperparameters
    if flags.get('batch_size', True):
        parser.add_argument(
            '--batch-size',
            '-b',
            type=int,
            default=10,
            metavar='N',
            help='Global batch size. It will be split equally between multiple GPUs in the distributed mode. '
                'Default: 10')

    if flags.get('epochs', True):
        parser.add_argument(
            '--epochs',
            type=int,
            default=300,
            help='Number of training epochs. Default: 300')

    if flags.get('precision', True):
        parser.add_argument(
            '--precision',
            type=str,
            default='float32',
            help='Precision to use {bfloat16, float32}. '
                'Default: float32')

    # Dataset params
    if flags.get('dataset_dir', True):
        parser.add_argument(
            '--data',
            dest='dataset_dir',
            type=str,
            help='Path to the root directory of the selected dataset.')

    if flags.get('dataset_type', True):
        parser.add_argument(
            '--dataset-type',
            choices=['tfds', 'tfrecords'],
            default='tfds',
            help='Dataset type.')

    # Storage settings
    if flags.get('log_dir', True):
        parser.add_argument(
            '--log-dir',
            type=str,
            default='runs',
            help='The directory where models and TensorboardX summaries '
                'are saved. Default: runs')

    if flags.get('save_checkpoint_freq', True):
        parser.add_argument(
            '--save-checkpoint-freq',
            default=5,
            type=int,
            help='Checkpoint save frequency (epochs). Default: 5')

    if flags.get('export_args', True):
        export_format = parser.add_mutually_exclusive_group()

        if flags.get('to_frozen_graph', True):
            export_format.add_argument(
                '--to-frozen-graph',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the Frozen Graph by given path.')

        if flags.get('to_saved_model', True):
            export_format.add_argument(
                '--to-saved-model',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the TensorFlow SavedModel format by given path.')

        if flags.get('to_h5', True):
            export_format.add_argument(
                '--to-h5',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the Keras H5 format by given path.')

    # Display
    if flags.get('print_freq', True):
        parser.add_argument(
            '-p',
            '--print-freq',
            default=10,
            type=int,
            metavar='N',
            help='Print frequency (batch iterations). '
                'Default: 10)')

    return parser
