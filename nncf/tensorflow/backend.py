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

# Workaround to handle the common part for Torch and TensorFlow backends.
import importlib
import os
import sys


def import_module_from_path(module_name, path):
    module_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

backend = import_module_from_path(
    'nncf.common.utils.backend',
    os.path.join(source_root, 'nncf', 'common', 'utils', 'backend.py')
)

backend.__nncf_backend__ = 'TensorFlow'
