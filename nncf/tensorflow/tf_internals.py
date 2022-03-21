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

from packaging import version

from nncf.tensorflow import tensorflow_version

from tensorflow.python.eager import context as eager_context

if version.parse(tensorflow_version) < version.parse('2.6'):
    from tensorflow.python.keras import engine as keras_engine
    from tensorflow.python.keras.utils.control_flow_util import smart_cond
else:
    from keras import engine as keras_engine
    from keras.utils.control_flow_util import smart_cond
