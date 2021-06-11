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

import tensorflow as tf

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmBuilder
from nncf.config.extractors import extract_compression_algorithm_configs
from nncf.tensorflow.algorithm_selector import get_compression_algorithm_builder
from nncf.tensorflow.api.composite_compression import TFCompositeCompressionAlgorithmBuilder
from nncf.tensorflow.helpers.utils import get_built_model


def create_compression_algorithm_builder(config: NNCFConfig,
                                         should_init: bool) -> CompressionAlgorithmBuilder:
    """
    Factory to create an instance of the compression algorithm builder
    by NNCFConfig.

    :param config: An instance of NNCFConfig that defines compression methods.
    :param should_init: The flag indicates that the generated compression builder
        will initialize (True) or not (False) the training parameters of the model
        during model building.
    :return: An instance of the `CompressionAlgorithmBuilder`
    """
    compression_algorithm_configs = extract_compression_algorithm_configs(config)

    number_compression_algorithms = len(compression_algorithm_configs)
    if number_compression_algorithms == 1:
        algo_config = compression_algorithm_configs[0]
        return get_compression_algorithm_builder(algo_config)(algo_config, should_init)
    if number_compression_algorithms > 1:
        return TFCompositeCompressionAlgorithmBuilder(config, should_init)
    return None


def create_compressed_model(model: tf.keras.Model,
                            config: NNCFConfig,
                            should_init: bool = True) -> tf.keras.Model:
    """
    The main function used to produce a model ready for compression fine-tuning
    from an original TensorFlow Keras model and a configuration object.

    :param model: The original model. Should have its parameters already loaded
        from a checkpoint or another source.
    :param config: A configuration object used to determine the exact compression
        modifications to be applied to the model.
    :param should_init: If False, trainable parameter initialization will be
        skipped during building.
    :return: The model with additional modifications necessary to enable
        algorithm-specific compression during fine-tuning.
    """
    model = get_built_model(model, config)

    builder = create_compression_algorithm_builder(config, should_init)
    if builder is None:
        return None, model
    compressed_model = builder.apply_to(model)
    compression_ctrl = builder.build_controller(compressed_model)
    return compression_ctrl, compressed_model
