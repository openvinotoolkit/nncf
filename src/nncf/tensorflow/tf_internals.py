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


from keras import backend as backend
from keras import layers as layers
from keras.layers import Rescaling as Rescaling
from packaging import version as version
from tensorflow.python.eager import context as eager_context  # noqa: F401

from nncf.tensorflow import tensorflow_version

if tensorflow_version < version.parse("2.13"):
    from keras import engine as keras_engine  # noqa: F401
    from keras.applications import imagenet_utils as imagenet_utils
    from keras.engine.keras_tensor import KerasTensor as KerasTensor
    from keras.utils.control_flow_util import smart_cond as smart_cond
else:
    from keras.src import engine as keras_engine  # noqa: F401
    from keras.src.applications import imagenet_utils as imagenet_utils  # noqa: E501
    from keras.src.engine.keras_tensor import KerasTensor as KerasTensor  # noqa: E501
    from keras.src.utils.control_flow_util import smart_cond as smart_cond  # noqa: E501
