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

import types
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.config.structures import ModelEvaluationArgs
from nncf.config.utils import is_accuracy_aware_training
from nncf.config.utils import is_experimental_quantization
from nncf.tensorflow.accuracy_aware_training.keras_model_utils import accuracy_aware_fit
from nncf.tensorflow.api.compression import TFCompressionAlgorithmBuilder
from nncf.config.extractors import extract_algorithm_names
from nncf.tensorflow.algorithm_selector import NoCompressionAlgorithmBuilder
from nncf.tensorflow.algorithm_selector import get_compression_algorithm_builder
from nncf.tensorflow.api.composite_compression import TFCompositeCompressionAlgorithmBuilder
from nncf.tensorflow.helpers.utils import get_built_model


def create_compression_algorithm_builder(config: NNCFConfig,
                                         should_init: bool) -> TFCompressionAlgorithmBuilder:
    """
    Factory to create an instance of the compression algorithm builder
    by NNCFConfig.

    :param config: An instance of NNCFConfig that defines compression methods.
    :param should_init: The flag indicates that the generated compression builder
        will initialize (True) or not (False) the training parameters of the model
        during model building.
    :return: An instance of the `CompressionAlgorithmBuilder`
    """
    algo_names = extract_algorithm_names(config)
    number_compression_algorithms = len(algo_names)
    if number_compression_algorithms == 0:
        return NoCompressionAlgorithmBuilder(config, should_init)
    if number_compression_algorithms == 1:
        algo_name = next(iter(algo_names))
        return get_compression_algorithm_builder(algo_name)(config, should_init)

    return TFCompositeCompressionAlgorithmBuilder(config, should_init)


def create_compressed_model(model: tf.keras.Model,
                            config: NNCFConfig,
                            compression_state: Optional[Dict[str, Any]] = None) \
        -> Tuple[CompressionAlgorithmController, tf.keras.Model]:
    """
    The main function used to produce a model ready for compression fine-tuning
    from an original TensorFlow Keras model and a configuration object.

    :param model: The original model. Should have its parameters already loaded
        from a checkpoint or another source.
    :param config: A configuration object used to determine the exact compression
        modifications to be applied to the model.
    :param compression_state: compression state to unambiguously restore the compressed model.
        Includes builder and controller states. If it is specified, trainable parameter initialization will be skipped
        during building.
    :return: A tuple (compression_ctrl, compressed_model) where
        - compression_ctrl: The controller of the compression algorithm.
        - compressed_model: The model with additional modifications
            necessary to enable algorithm-specific compression during fine-tuning.
    """
    if is_experimental_quantization(config):
        from nncf.experimental.tensorflow.nncf_network import NNCFNetwork
        input_signature = get_input_signature(config)
        model = NNCFNetwork(model, input_signature)
        model.compute_output_signature(model.input_signature)

    model = get_built_model(model, config)
    original_model_accuracy = None

    if is_accuracy_aware_training(config):
        if config.has_extra_struct(ModelEvaluationArgs):
            evaluation_args = config.get_extra_struct(ModelEvaluationArgs)
            original_model_accuracy = evaluation_args.eval_fn(model)

    builder = create_compression_algorithm_builder(config, should_init=not compression_state)
    if compression_state:
        builder.load_state(compression_state[BaseController.BUILDER_STATE])
    compressed_model = builder.apply_to(model)
    compression_ctrl = builder.build_controller(compressed_model)
    compressed_model.original_model_accuracy = original_model_accuracy
    if isinstance(compressed_model, tf.keras.Model):
        compressed_model.accuracy_aware_fit = types.MethodType(accuracy_aware_fit, compressed_model)
    return compression_ctrl, compressed_model


def get_input_signature(config: NNCFConfig):
    input_info = config.get('input_info', {})
    samples_sizes = []

    if isinstance(input_info, dict):
        sample_size = input_info['sample_size']
        samples_sizes.append(sample_size)
    elif isinstance(input_info, list):
        for info in input_info:
            sample_size = info['sample_size']
            samples_sizes.append(sample_size)
    else:
        raise RuntimeError('sample_size must be provided in configuration file')

    input_signature = []
    for sample_size in samples_sizes:
        shape = [None] + list(sample_size[1:])
        input_signature.append(tf.TensorSpec(shape=shape, dtype=tf.float32))

    return input_signature if len(input_signature) > 1 else input_signature[0]
