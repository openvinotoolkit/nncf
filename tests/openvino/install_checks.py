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

import pkgutil
import re
import sys

import nncf

EXCLUDED_MODULES_PATTERNS = (
    'nncf\\.onnx.*',
    'nncf\\.tensorflow.*',
    'nncf\\.torch.*',
    'nncf\\.experimental\\.onnx.*',
    'nncf\\.experimental\\.tensorflow.*',
    'nncf\\.experimental\\.torch.*',
    '.*?onnx_[^\\.]*',
    '.*?torch_[^\\.]*',
    '.*?tf_[^\\.]*',
    )


def excluded_module(name):
    for pattern in EXCLUDED_MODULES_PATTERNS:
        if re.fullmatch(pattern, name):
            return True
    return False


def onerror(name):
    if not excluded_module(name):
        raise RuntimeError(f'Could not import {name}')


for loader, module_name, is_pkg in pkgutil.walk_packages(nncf.__path__,
                                                         nncf.__name__ + '.',
                                                         onerror):
    if module_name in sys.modules or excluded_module(module_name):
        continue
    if not is_pkg:
        loader.find_module(module_name).load_module(module_name)
