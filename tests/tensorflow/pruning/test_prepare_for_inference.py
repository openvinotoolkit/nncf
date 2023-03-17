"""
 Copyright (c) 2023 Intel Corporation
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

import pytest
import tensorflow as tf

from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.pruning.helpers import get_concat_test_model


@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_prepare_for_inference(enable_quantization):
    input_shape = (1, 8, 8, 3)
    model = get_concat_test_model(input_shape)

    config = get_empty_config(input_sample_sizes=input_shape)
    config.update(
        {"compression": [{"algorithm": "filter_pruning", "pruning_init": 0.5, "params": {"prune_first_conv": True}}]}
    )
    if enable_quantization:
        config["compression"].append(
            {
                "algorithm": "quantization",
                "preset": "mixed",
                "initializer": {
                    "batchnorm_adaptation": {
                        "num_bn_adaptation_samples": 0,
                    }
                },
            }
        )

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    input_tensor = tf.ones(input_shape)
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.prepare_for_inference()
    x_tf = inference_model(input_tensor)

    TFTensorListComparator.check_equal(x_nncf, x_tf)

@pytest.mark.parametrize("do_copy", (True, False))
@pytest.mark.parametrize("enable_quantization", (True, False), ids=("with_quantization", "no_quantization"))
def test_do_copy(do_copy, enable_quantization):
    input_shape = (1, 8, 8, 3)
    model = get_concat_test_model(input_shape)

    config = get_empty_config(input_sample_sizes=input_shape)
    config.update(
        {"compression": [{"algorithm": "filter_pruning", "pruning_init": 0.5, "params": {"prune_first_conv": True}}]}
    )
    if enable_quantization:
        config["compression"].append({"algorithm": "quantization", "preset": "mixed"})

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    inference_model = compression_ctrl.prepare_for_inference(do_copy=do_copy)

    # Transform model for pruning creates copy of the model in both cases
    assert id(inference_model) != id(compression_model)
