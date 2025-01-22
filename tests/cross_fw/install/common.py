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

import pkgutil
import re
import sys
from functools import partial

import nncf


def excluded_module(name, excluded_modules_patterns):
    for pattern in excluded_modules_patterns:
        if re.fullmatch(pattern, name):
            return True
    return False


def onerror(name, excluded_modules_patterns):
    if not excluded_module(name, excluded_modules_patterns):
        raise nncf.InternalError(f"Could not import {name}")


def load_nncf_modules(excluded_modules_patterns, verbose=False):
    onerror_partial = partial(onerror, excluded_modules_patterns=excluded_modules_patterns)
    for loader, module_name, _ in pkgutil.walk_packages(nncf.__path__, nncf.__name__ + ".", onerror_partial):
        if module_name in sys.modules or excluded_module(module_name, excluded_modules_patterns):
            if verbose:
                print(f"Module {module_name} ------ SKIPPED")
            continue
        loader.find_module(module_name).load_module(module_name)
        if verbose:
            print(f"Module {module_name} ------ LOADED")
