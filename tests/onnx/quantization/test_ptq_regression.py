"""
Copyright (c) 2023 Intel Corporation

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

import pytest

import subprocess
import sys
from tests.shared.paths import PROJECT_ROOT
from tests.shared.paths import EXAMPLES_DIR


def test_mobilenet_v2_regression(verbose: bool = False):
    script_path = EXAMPLES_DIR / 'post_training_quantization' / 'onnx' / 'mobilenet_v2' / 'main.py'
    command = f'{sys.executable} {script_path}'
    print(f"Run command: {command}")
    with subprocess.Popen(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          cwd=PROJECT_ROOT) as result:
        command_output, _ = result.communicate()
        command_output = command_output.decode("utf-8")
        accuracy_drop = command_output.splitlines()[-3].split(' ')[-1]
        assert 100 * float(accuracy_drop) < 0.3
        if result.returncode != 0:
            print(command_output)
            pytest.fail()
            return
        if verbose:
            print(command_output)
