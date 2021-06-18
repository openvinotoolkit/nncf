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

import os

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

preferable_framework = os.getenv('NNCF_BACKEND')
if preferable_framework is not None:
    preferable_framework = preferable_framework.lower()
    if preferable_framework == 'torch':
        if torch is None:
            raise RuntimeError('Preferable backend set to torch, but PyTorch is not installed')
        __nncf_backend__ = 'Torch'
    elif preferable_framework == 'tf':
        if tf is None:
            raise RuntimeError('Preferable backend set to tf, but TensorFlow is not installed')
        __nncf_backend__ = 'TensorFlow'
    else:
        raise RuntimeError('Unrecognized preferable framework. NNCF_BACKEND must be "torch" or "tf"')
elif torch and tf:
    raise RuntimeError('PyTorch and TensorFlow have been found. Please set NNCF_BACKEND to "torch" or "tf"')
elif torch:
    __nncf_backend__ = 'Torch'
elif tf:
    __nncf_backend__ = 'TensorFlow'
else:
    raise RuntimeError('None of PyTorch or TensorFlow have been found. Please, install one of the frameworks')
