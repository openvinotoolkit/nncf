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

import pytest
import tensorflow as tf
import tensorflow_hub as hub

from nncf import NNCFConfig
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test

# TODO(achurkin): enable after 120296 ticked is fixed
# from nncf.experimental.tensorflow.patch_tf import patch_tf_operations
# patch_tf_operations()


@pytest.mark.skip(reason="ticket 120296")
def test_keras_layer_model():
    nncf_config = NNCFConfig(
        {
            "model": "Model",
            "input_info": [{"sample_size": [1, 224, 224, 3]}],
            "compression": {"algorithm": "experimental_quantization"},
        }
    )

    model = tf.keras.Sequential(
        [hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5")]
    )

    with pytest.raises(ValueError):
        create_compressed_model_and_algo_for_test(model, nncf_config, force_no_init=True)
