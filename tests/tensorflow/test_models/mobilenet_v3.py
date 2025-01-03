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

from examples.tensorflow.common.models import MobileNetV3Large as mobilenet_v3_large
from examples.tensorflow.common.models import MobileNetV3Small as mobilenet_v3_small


def MobileNetV3Small(input_shape=None):
    return mobilenet_v3_small(input_shape)


def MobileNetV3Large(input_shape=None):
    return mobilenet_v3_large(input_shape)
