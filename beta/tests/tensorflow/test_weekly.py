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

import json
import os
import sys
import tempfile

import time # for Command
import signal # for Command
import subprocess # for Command
import threading

import pytest
from pytest import approx

from beta.tests.conftest import PROJECT_ROOT

# sample
# ├── dataset
# │   ├── path
# │   ├── batch
# │   ├── configs
# │   │     ├─── config_filename
# │   │     │       ├── expected_accuracy
# │   │     │       ├── absolute_tolerance_train
# │   │     │       ├── absolute_tolerance_eval
# │   │     │       ├── execution_arg
# │   │     │       ├── weights
GLOBAL_CONFIG = {
    'classification': {
            'imagenet': {
                    'configs': {
                        'quantization/inception_v3_imagenet_int8.json': {
                            'expected_accuracy': 78.41,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity_quantization/inception_v3_imagenet_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 77.52,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity/inception_v3_imagenet_magnitude_sparsity.json': {
                            'expected_accuracy': 77.87,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'quantization/mobilenet_v2_imagenet_int8.json': {
                            'expected_accuracy': 71.96,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity_quantization/mobilenet_v2_imagenet_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 72.17,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity/mobilenet_v2_imagenet_magnitude_sparsity.json': {
                            'expected_accuracy': 72.36,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity/mobilenet_v2_hub_imagenet_magnitude_sparsity.json': {
                            'expected_accuracy': 71.83,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'quantization/mobilenet_v3_small_imagenet_int8.json': {
                            'expected_accuracy': 67.51,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity_quantization/mobilenet_v3_small_imagenet_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 67.81,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'quantization/mobilenet_v3_large_imagenet_int8.json': {
                            'expected_accuracy': 75.13,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity_quantization/mobilenet_v3_large_imagenet_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 74.94,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'quantization/resnet50_imagenet_int8.json': {
                            'expected_accuracy': 75.04,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity_quantization/resnet50_imagenet_magnitude_sparsity_int8.json': {
                            'expected_accuracy': 74.46,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'sparsity/resnet50_imagenet_magnitude_sparsity.json': {
                            'expected_accuracy': 75.00,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                        'pruning/resnet50_imagenet_pruning_geometric_median.json': {
                            'expected_accuracy': 74.98,
                            'absolute_tolerance_train': 0.5,
                            'absolute_tolerance_eval': 0.5
                        },
                    }
                }
        },
    'object_detection': {
            'coco2017': {
                'configs': {
                    'quantization/retinanet_coco_int8.json': {
                        'expected_accuracy': 33.3,
                        'absolute_tolerance_train': 0.5,
                        'absolute_tolerance_eval': 0.5
                    },
                    'sparsity/retinanet_coco_magnitude_sparsity.json': {
                        'expected_accuracy': 33.13,
                        'absolute_tolerance_train': 0.5,
                        'absolute_tolerance_eval': 0.5
                    },
                    'pruning/retinanet_coco_pruning.json': {
                        'expected_accuracy': 32.70,
                        'absolute_tolerance_train': 0.5,
                        'absolute_tolerance_eval': 0.5
                    },
                    'quantization/yolo_v4_coco_int8.json': {
                        'expected_accuracy': 46.20,
                        'absolute_tolerance_train': 0.5,
                        'absolute_tolerance_eval': 0.5
                    },
                    'sparsity/yolo_v4_coco_magnitude_sparsity.json': {
                        'expected_accuracy': 46.54,
                        'absolute_tolerance_train': 0.5,
                        'absolute_tolerance_eval': 0.5
                    },
                }
            }
        },
    'segmentation': {
        'coco2017': {
            'configs': {
                'quantization/mask_rcnn_coco_int8.json': {
                    'expected_accuracy': 37.25,
                    'absolute_tolerance_train': 0.5,
                    'absolute_tolerance_eval': 0.5
                },
                'sparsity/mask_rcnn_coco_magnitude_sparsity.json': {
                    'expected_accuracy': 36.93,
                    'absolute_tolerance_train': 0.5,
                    'absolute_tolerance_eval': 0.5
                },
            }
        }
    }
}

EXAMPLES_DIR = PROJECT_ROOT.joinpath('examples', 'tensorflow')


def get_cli_dict_args(args): # move to nncf/tests/test_helpers.py
    cli_args = dict()
    for key, val in args.items():
        cli_key = '--{}'.format(str(key))
        cli_args[cli_key] = None
        if val is not None:
            cli_args[cli_key] = str(val)
    return cli_args

# move to nncf/tests/test_helpers.py
class Command:
    def __init__(self, cmd, path=None):
        self.cmd = cmd
        self.process = None
        self.exec_time = -1
        self.output = []  # store output here
        self.kwargs = {}
        self.timeout = False
        self.path = path

        # set system/version dependent "start_new_session" analogs
        if sys.platform == "win32":
            self.kwargs.update(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        elif sys.version_info < (3, 2):  # assume posix
            self.kwargs.update(preexec_fn=os.setsid)
        else:  # Python 3.2+ and Unix
            self.kwargs.update(start_new_session=True)

    def kill_process_tree(self, pid):
        try:
            if sys.platform != "win32":
                os.killpg(pid, signal.SIGKILL)
            else:
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)])
        except OSError as err:
            print(err)

    def run(self, timeout=3600, assert_returncode_zero=True):
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()  # See runs_subprocess_in_precommit for more info on why this is needed

        def target():
            start_time = time.time()
            self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
                                            bufsize=1, cwd=self.path, **self.kwargs)
            self.timeout = False

            self.output = []
            for line in self.process.stdout:
                line = line.decode('utf-8')
                self.output.append(line)
                sys.stdout.write(line)

            sys.stdout.flush()
            self.process.stdout.close()

            self.process.wait()
            self.exec_time = time.time() - start_time

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            try:
                print("Error: process taking too long to complete--terminating" + ", [ " + self.cmd + " ]")
                self.kill_process_tree(self.process.pid)
                self.exec_time = timeout
                self.timeout = True
                thread.join()
            except OSError as e:
                print(self.process.pid, "Exception when try to kill task by PID, " + e.strerror)
                raise
        returncode = self.process.wait()
        print("Process returncode = " + str(returncode))
        if assert_returncode_zero:
            assert returncode == 0, "Process exited with a non-zero exit code {}; output:{}".format(
                returncode,
                "".join(self.output))
        return returncode

    def get_execution_time(self):
        return self.exec_time


def create_command_line(args, sample_type):
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] += ':' + str(PROJECT_ROOT)
    else:
        env['PYTHONPATH'] = str(PROJECT_ROOT)
    if sample_type == 'segmentation':
        main_py = 'train.py' if args['--mode'] == 'train' else 'evaluation.py'
    else:
        main_py = 'main.py'
    executable = EXAMPLES_DIR.joinpath(sample_type, main_py).as_posix()

    cli_args = " ".join(key if (val is None or val is True) else "{} {}".format(key, val) for key, val in args.items())
    return "{python_exe} {main_py} {args}".format(
         main_py=executable, args=cli_args, python_exe=sys.executable
    )


CONFIG_PARAMS = []
for sample_type_ in GLOBAL_CONFIG:
    datasets = GLOBAL_CONFIG[sample_type_]
    for dataset_name_ in datasets:
        dataset_path = datasets[dataset_name_].get('path', os.path.join(tempfile.gettempdir(), dataset_name_))
        batch_size = datasets[dataset_name_].get('batch', None)
        configs = datasets[dataset_name_].get('configs', {})
        for config_name in configs:
            config_params = configs[config_name]
            execution_args = config_params.get('execution_arg', [''])
            expected_accuracy_ = config_params.get('expected_accuracy', 100)
            absolute_tolerance_train_ = config_params.get('absolute_tolerance_train', 1)
            absolute_tolerance_eval_ = config_params.get('absolute_tolerance_eval', 0.5)
            weights_path_ = config_params.get('weights', None)
            epochs = config_params.get('epochs', None)
            if weights_path_:
                weights_path_ = os.path.join(sample_type_, dataset_name_, weights_path_)
            for execution_arg_ in execution_args:
                config_path_ = EXAMPLES_DIR.joinpath(sample_type_, 'configs', config_name)
                jconfig = json.load(config_path_.open())
                args_ = {
                    'data': dataset_path,
                    'weights': weights_path_,
                    'config': str(config_path_)
                }

                test_config_ = {
                    'sample_type': sample_type_,
                    'expected_accuracy': expected_accuracy_,
                    'absolute_tolerance_train': absolute_tolerance_train_,
                    'absolute_tolerance_eval': absolute_tolerance_eval_,
                    'checkpoint_name': config_name.split('/')[-1]
                }
                CONFIG_PARAMS.append(tuple([test_config_, args_, execution_arg_, dataset_name_]))


def get_config_name(config_path):
    base = os.path.basename(config_path)
    return os.path.splitext(base)[0]


@pytest.fixture(scope='module', params=CONFIG_PARAMS,
                ids=['-'.join([p[0]['sample_type'], get_config_name(p[1]['config'])]) for p in CONFIG_PARAMS])
def _params(request, tmp_path_factory, dataset_dir, weekly_tests):
    if not weekly_tests:
        pytest.skip('For weekly testing use --run-weekly-tests option.')
    test_config, args, execution_arg, _ = request.param
    if args['weights']:
        if not os.path.exists(args['weights']):
            raise FileExistsError('Weights file does not exist: {}'.format(args['weights']))
    else:
        del args['weights']
    if execution_arg:
        args[execution_arg] = None
    checkpoint_save_dir = str(tmp_path_factory.mktemp('models'))
    checkpoint_save_dir = os.path.join(checkpoint_save_dir, execution_arg.replace('-', '_'))
    args['checkpoint-save-dir'] = checkpoint_save_dir
    if dataset_dir:
        args['data'] = dataset_dir
    return {
        'test_config': test_config,
        'args': args,
    }


@pytest.mark.dependency(name="train")
def test_compression_train(_params, tmp_path):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['mode'] = 'train'
    args['log-dir'] = tmp_path
    args['metrics-dump'] = os.path.join(args['checkpoint-save-dir'], 'metrics.json')

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type']))
    runner.run(timeout=threading.TIMEOUT_MAX)

    assert os.path.exists(args['metrics-dump'])
    with open(args['metrics-dump']) as metric_file: # test_sota_checkpoint.TestSotaCheckpoints.read_metric
        metrics = json.load(metric_file)
        actual_acc = metrics['Accuracy']

    ref_acc = tc['expected_accuracy']
    better_accuracy_tolerance = 3
    tolerance = tc['absolute_tolerance_train'] if actual_acc < ref_acc else better_accuracy_tolerance
    assert actual_acc == approx(ref_acc, abs=tolerance)


# Not sure that we need this test
@pytest.mark.dependency(depends=["train"])
def test_compression_eval_trained(_params, tmp_path):
    p = _params
    args = p['args']
    tc = p['test_config']

    args['mode'] = 'test'
    args['log-dir'] = tmp_path
    args['resume'] = args['checkpoint-save-dir']
    args['metrics-dump'] = os.path.join(args['checkpoint-save-dir'], 'metrics.json')
    if 'weights' in args:
        del args['weights']

    runner = Command(create_command_line(get_cli_dict_args(args), tc['sample_type']))
    runner.run(timeout=threading.TIMEOUT_MAX)

    assert os.path.exists(args['metrics-dump'])
    with open(args['metrics-dump']) as metric_file: # test_sota_checkpoint.TestSotaCheckpoints.read_metric
        metrics = json.load(metric_file)
        actual_acc = metrics['Accuracy']

    ref_acc = tc['expected_accuracy']
    assert actual_acc == approx(ref_acc, abs=tc['absolute_tolerance_eval'])
