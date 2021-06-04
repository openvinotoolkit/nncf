"""
 Copyright (c) 2020 Intel Corporation
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

import types

import tensorflow as tf

from beta.nncf.tensorflow.algorithm_selector import get_compression_algorithm_builder
from beta.nncf.tensorflow.api.composite_compression import TFCompositeCompressionAlgorithmBuilder
from beta.nncf.tensorflow.helpers.utils import get_built_model
from beta.nncf.tensorflow.accuracy_aware_training.keras_model_utils import accuracy_aware_fit
from nncf.common.structures import ModelEvaluationArgs
from nncf.common.utils.helpers import is_accuracy_aware_training


def create_compression_algorithm_builder(config):
    compression_config = config.get('compression', {})

    if isinstance(compression_config, dict):
        return get_compression_algorithm_builder(compression_config)(compression_config)
    if isinstance(compression_config, list):
        composite_builder = TFCompositeCompressionAlgorithmBuilder()
        for algo_config in compression_config:
            composite_builder.add(get_compression_algorithm_builder(algo_config)(algo_config))
        return composite_builder
    return None


def create_compressed_model(model, config):
    model = get_built_model(model, config)
    original_model_accuracy = None

    if is_accuracy_aware_training(config, compression_config_passed=True):
        if config.has_extra_struct(ModelEvaluationArgs):
            evaluation_args = config.get_extra_struct(ModelEvaluationArgs)
            original_model_accuracy = evaluation_args.eval_fn(model)

    builder = create_compression_algorithm_builder(config)
    if builder is None:
        return None, model
    compressed_model = builder.apply_to(model)
    compression_ctrl = builder.build_controller(compressed_model)
    compressed_model.original_model_accuracy = original_model_accuracy
    if isinstance(compressed_model, tf.keras.Model):
        compressed_model.accuracy_aware_fit = types.MethodType(accuracy_aware_fit, compressed_model)
    return compression_ctrl, compressed_model
