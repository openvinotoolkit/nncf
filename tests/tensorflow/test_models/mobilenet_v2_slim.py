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

from examples.tensorflow.common.models import mobilenet_v2_100_224


def HubMobileNetV2(input_shape=None):
    # Note: this will download the model to a temporary directory, which must either exist and contain
    # the model, or not exist at all.  On Windows there are sometimes situations when the temporary directory
    # for the model is not deleted completely, leaving the empty temporary directory, which leads to an error when
    # the model is next downloaded.
    return mobilenet_v2_100_224(input_shape)
