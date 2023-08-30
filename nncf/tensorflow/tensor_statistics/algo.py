# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable
import tensorflow as tf

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.tensorflow.tensor import TFNNCFTensor


def get_collection_hook(collector: TensorStatisticCollectorBase) -> Callable[[tf.Tensor], tf.Tensor]:
    # The hook closure function should be instantiated and returned in a separate function - inlining this in
    # "for" loops iterating over `collector` leads to unexpected behaviour of closure binding to wrong collector
    # instance

    def hook(x: tf.Tensor) -> tf.Tensor:
        collector.register_input(TFNNCFTensor(x))
        return x

    return hook