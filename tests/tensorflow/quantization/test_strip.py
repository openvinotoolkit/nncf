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

import pytest
import tensorflow as tf

import nncf
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.tensorflow.graph.utils import get_nncf_operations
from tests.tensorflow.accuracy_aware_training.test_keras_api import get_const_target_mock_regression_dataset
from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_two_conv_test_model
from tests.tensorflow.quantization.utils import get_basic_quantization_config


def test_strip():
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    input_tensor = tf.ones([1, 4, 4, 1])
    x_nncf = compressed_model(input_tensor)

    inference_model = compression_ctrl.strip()
    x_tf = inference_model(input_tensor)

    TFTensorListComparator.check_equal(x_nncf, x_tf)


@pytest.mark.parametrize("do_copy", (True, False))
def test_do_copy(do_copy):
    model = get_basic_two_conv_test_model()
    config = get_basic_quantization_config()
    config["compression"] = {
        "algorithm": "quantization",
        "preset": "mixed",
    }
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    inference_model = compression_ctrl.strip(do_copy=do_copy)

    if do_copy:
        assert id(inference_model) != id(compression_model)
    else:
        assert id(inference_model) == id(compression_model)


@pytest.mark.parametrize("strip_model", (True, False, None))
def test_nncf_quantize_strip(strip_model):
    model = get_basic_two_conv_test_model()

    def transform_fn(data_item):
        return data_item[0]

    dataset = nncf.Dataset(get_const_target_mock_regression_dataset(img_size=4), transform_fn)

    if strip_model is not None:
        advanced_parameters = AdvancedQuantizationParameters()
        advanced_parameters.strip_model = strip_model
    else:
        advanced_parameters = None

    quantized_model = nncf.quantize(model, dataset, advanced_parameters=advanced_parameters)

    has_half_range = False
    for _, _, op in get_nncf_operations(quantized_model, ["conv2d_kernel_quantizer", "conv2d_1_kernel_quantizer"]):
        has_half_range = has_half_range or op.half_range

    if strip_model is None or strip_model is True:
        assert not has_half_range
    else:
        assert has_half_range
