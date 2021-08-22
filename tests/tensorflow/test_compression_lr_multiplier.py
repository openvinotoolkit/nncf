"""
 Copyright (c) 2019-2020 Intel Corporation
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
import itertools
import copy
import numpy as np
import pytest
import tensorflow as tf
from typing import Callable, Dict, Generator, List, Optional, Tuple

from nncf import NNCFConfig
from nncf.tensorflow import register_default_init_args, create_compressed_model
from tests.common.test_compression_lr_multiplier import BaseCompressionLRMultiplierTester
from tests.common.test_compression_lr_multiplier import get_configs_building_params
from tests.common.test_compression_lr_multiplier import get_quantization_config
from tests.common.test_compression_lr_multiplier import get_rb_sparsity_config
# from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_lenet_model
from tests.tensorflow.helpers import TFTensorListComparator


def create_initialized_model_and_dataset(config: NNCFConfig) -> Tuple[tf.keras.Model, tf.data.Dataset]:
    tf.random.set_seed(42)
    np.random.seed(42)
    config = copy.deepcopy(config)

    def generate_dataset():
        while True:
            # pylint: disable = no-value-for-parameter
            yield np.random.randn(1, 32, 32, 1), tf.one_hot([np.random.randint(10)], 10)

    dataset = tf.data.Dataset.from_generator(
        generate_dataset, output_signature=(
            tf.TensorSpec(shape=(1, 32, 32, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 10), dtype=tf.float32)
        ))

    config = register_default_init_args(config, dataset, batch_size=1)
    _algo, model = create_compressed_model(get_lenet_model(), config)

    return model, dataset


@pytest.fixture(name='sample_size')
def sample_size_():
    return [1, 32, 32, 1]


@pytest.fixture(name='get_ref_model_and_dataset')
def get_ref_model_and_dataset_(ref_config: NNCFConfig) -> Callable[[], Tuple[tf.keras.Model, tf.data.Dataset]]:
    def f():
        return create_initialized_model_and_dataset(ref_config)
    return f


@pytest.fixture(name='get_target_model_and_dataset')
def get_target_model_and_dataset_(target_config: NNCFConfig) -> Callable[[], Tuple[tf.keras.Model, tf.data.Dataset]]:
    def f():
        return create_initialized_model_and_dataset(target_config)
    return f


class TestTFCompressionLRMultiplier(BaseCompressionLRMultiplierTester):
    ALGO_NAME_TO_PATH_MAP = {
        'quantization': 'nncf.tensorflow.quantization',
        'rb_sparsity': 'nncf.tensorflow.sparsity.rb',
        'binarization': 'nncf.tensorflow.binarization'
    }

    TensorListComparator = TFTensorListComparator

    @pytest.fixture(name='configs_building_params',
                    params=get_configs_building_params([
                        get_quantization_config, get_rb_sparsity_config
                    ]))
    def configs_building_params_(self, request) -> Dict:
        return request.param

    @classmethod
    def _perform_model_training_steps(cls, model: tf.keras.Model, train_data: tf.data.Dataset,
                                      num_steps: int = 1) -> Tuple[tf.keras.Model, Optional[tf.GradientTape]]:
        tf.random.set_seed(42)
        loss_obj = tf.keras.losses.mean_squared_error
        optimizer = tf.keras.optimizers.SGD(lr=0.1)

        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as grad_tape:
                predictions = model(inputs, training=True)
                loss = loss_obj(labels, predictions)

            grads = grad_tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return grads

        grads = None
        for input_batch, label_batch in itertools.islice(train_data, num_steps):
            grads = train_step(input_batch, label_batch)
            keys = map(lambda v: v.ref(), model.trainable_variables)
            grads = dict(zip(keys, grads))

        return model, grads

    @classmethod
    def _get_grads(cls, algo_to_params: Dict[str, List[tf.Tensor]], grads) -> Dict[str, List[tf.Tensor]]:
        res = {}
        for algo, params in algo_to_params.items():
            res[algo] = []
            if grads is not None:
                for param in params:
                    res[algo].append(grads[param.ref()])
        return res

    @classmethod
    def _get_layer_cls_and_params(cls, model: tf.keras.Model) -> Generator[Tuple[type, List[tf.Tensor]], None, None]:
        for layer in model.layers:
            params = layer.trainable_weights
            yield layer.__class__, params

    @classmethod
    def _get_params_and_grads_after_training_steps(cls, model: tf.keras.Model, dataset: tf.data.Dataset,
                                                   num_steps: int = 1) -> Tuple[Dict[str, List[tf.Tensor]],
                                                                                Dict[str, List[tf.Tensor]]]:
        model, grads = cls._perform_model_training_steps(model, dataset, num_steps)
        params = cls._get_params_grouped_by_algos(model)
        grads = cls._get_grads(params, grads)
        return params, grads
