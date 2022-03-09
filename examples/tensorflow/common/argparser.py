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

from examples.tensorflow.common.sample_config import CustomArgumentParser


def get_common_argument_parser(**flags):
    """Defines command-line arguments, and parses them.

    """
    parser = CustomArgumentParser()

    add_argument(parser=parser,
                 condition=flags.get('mode', True),
                 parameters=argument_parameters(
                     '--mode',
                     '-m',
                     nargs='+',
                     choices=['train', 'test', 'export'],
                     default='train',
                     help='train: performs training and validation; test: tests the model; export: exports the model.'
                 )
                 )

    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with task/model-specific parameters.',
        required=True)

    add_argument(
        parser=parser,
        condition=flags.get('metrics_dump', True),
        parameters=argument_parameters(
            '--metrics-dump',
            type=str,
            help='Name of metrics collecting .json file'))

    if flags.get('resume_args', True):
        model_init_mode = parser.add_mutually_exclusive_group()

        add_argument(
            parser=model_init_mode,
            condition=flags.get('resume', True),
            parameters=argument_parameters(
                '--resume',
                metavar='PATH',
                type=str,
                default=None,
                dest='ckpt_path',
                help='Specifies the path to the checkpoint to resume training, test or export model '
                    'from the defined checkpoint or folder with checkpoints to resume training, test '
                    'or export from the last checkpoint.'))

        add_argument(
            parser=model_init_mode,
            condition=flags.get('weights', True),
            parameters=argument_parameters(
                '--weights',
                default=None,
                type=str,
                help='Path to pretrained weights in H5 format.'))

    add_argument(
        parser=parser,
        condition=flags.get('checkpoint_save_dir', True),
        parameters=argument_parameters(
            '--checkpoint-save-dir',
            metavar='PATH',
            type=str,
            default=None,
            help='Specifies the directory for the trained model checkpoints to be saved to.'))

    if flags.get('execution_args', True):
        execution_type = parser.add_mutually_exclusive_group()

        add_argument(
            parser=execution_type,
            condition=flags.get('gpu_id', True),
            parameters=argument_parameters(
                '--gpu-id',
                type=int,
                metavar='N',
                help='The ID of the GPU training will be performed on, without any parallelization.'))

        add_argument(
            parser=execution_type,
            condition=flags.get('cpu_only', True),
            parameters=argument_parameters(
                '--cpu-only',
                action='store_true',
                help='Specifies that the computation should be performed using CPU only.'))

    # Hyperparameters
    add_argument(
        parser=parser,
        condition=flags.get('batch_size', True),
        parameters=argument_parameters(
            '--batch-size',
            '-b',
            type=int,
            default=10,
            metavar='N',
            help='Global batch size. It will be split equally between multiple GPUs '
                 'in the distributed mode. Default: 10'))

    add_argument(
        parser=parser,
        condition=flags.get('epochs', True),
        parameters=argument_parameters(
            '--epochs',
            type=int,
            default=300,
            help='Number of training epochs. Default: 300'))

    add_argument(
        parser=parser,
        condition=flags.get('precision', True),
        parameters=argument_parameters(
            '--precision',
            type=str,
            default='float32',
            help='Precision to use {bfloat16, float32}. Default: float32'))

    # Dataset params
    add_argument(
        parser=parser,
        condition=flags.get('dataset_dir', True),
        parameters=argument_parameters(
            '--data',
            dest='dataset_dir',
            type=str,
            help='Path to the root directory of the selected dataset.'))

    add_argument(
        parser=parser,
        condition=flags.get('dataset_type', True),
        parameters=argument_parameters(
            '--dataset-type',
            choices=['tfds', 'tfrecords'],
            default='tfds',
            help='Dataset type.'))

    # Storage settings
    add_argument(
        parser=parser,
        condition=flags.get('log_dir', True),
        parameters=argument_parameters(
            '--log-dir',
            type=str,
            default='runs',
            help='The directory where models and TensorboardX summaries are saved. '
                 'Default: runs'))

    add_argument(
        parser=parser,
        condition=flags.get('save_checkpoint_freq', True),
        parameters=argument_parameters(
            '--save-checkpoint-freq',
            default=5,
            type=int,
            help='Checkpoint save frequency (epochs). Default: 5'))

    if flags.get('export_args', True):
        export_format = parser.add_mutually_exclusive_group()

        add_argument(
            parser=export_format,
            condition=flags.get('to_frozen_graph', True),
            parameters=argument_parameters(
                '--to-frozen-graph',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the Frozen Graph by given path.'))

        add_argument(
            parser=export_format,
            condition=flags.get('to_saved_model', True),
            parameters=argument_parameters(
                '--to-saved-model',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the TensorFlow SavedModel format '
                     'by given path.'))

        add_argument(
            parser=export_format,
            condition=flags.get('to_h5', True),
            parameters=argument_parameters(
                '--to-h5',
                type=str,
                metavar='PATH',
                default=None,
                help='Export the compressed model to the Keras H5 format by given path.'))

    # Display
    add_argument(
        parser=parser,
        condition=flags.get('print_freq', True),
        parameters=argument_parameters(
            '-p',
            '--print-freq',
            default=10,
            type=int,
            metavar='N',
            help='Print frequency (batch iterations). Default: 10)'))

    parser.add_argument(
        "--disable-compression",
        help="Disable compression",
        action="store_true",
    )

    parser.add_argument(
        '--seed', default=None, type=int,
        help='Specific seed for initializing pseudo-random number generators.')

    return parser


def argument_parameters(*args, **kwargs):
    return (args, kwargs)


def add_argument(parser, condition, parameters):
    if condition:
        parser.add_argument(*parameters[0], **parameters[1])
