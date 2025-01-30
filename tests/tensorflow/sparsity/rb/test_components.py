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

import contextlib
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import pytest
import tensorflow as tf

from nncf import NNCFConfig
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.tensorflow.helpers.callback_creation import create_compression_callbacks
from nncf.tensorflow.helpers.model_creation import create_compressed_model
from nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask
from nncf.tensorflow.sparsity.rb.loss import SparseLoss
from nncf.tensorflow.sparsity.rb.operation import RBSparsifyingWeight
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import get_basic_fc_test_model
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.helpers import get_op_by_cls
from tests.tensorflow.helpers import get_weight_by_name

CONF = Path(__file__).parent.parent.parent / "data" / "configs" / "sequential_model_cifar10_rb_sparsity.json"
MASKS_SEEDS_PATH = Path(__file__).parent / "output_seeds.txt"

TEST_MODELS = {
    "Dense": lambda: get_basic_fc_test_model(input_shape=(4,), out_shape=10),
    "Conv2D": lambda: get_basic_conv_test_model(
        input_shape=(4, 4, 1), out_channels=2, kernel_size=2, weight_init=-1.0, bias_init=-2.0, transpose=False
    ),
    "Conv2DTranspose": lambda: get_basic_conv_test_model(
        input_shape=(4, 4, 1), out_channels=1, kernel_size=2, weight_init=-1.0, bias_init=-2.0, transpose=True
    ),
}


def get_basic_rb_sparse_model(model_name, local=False, config=CONF, freeze=False):
    model = TEST_MODELS[model_name]()
    if isinstance(config, Path):
        config = NNCFConfig.from_json(config)
    if local:
        config.update({"params": {"sparsity_level_setting_mode": "local"}})
    compress_model, algo = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)
    if freeze:
        algo.freeze()
    return compress_model, algo, config


def calc_rb_mask_decorator(fn):
    def wrapper(*args, **kwargs):
        if tf.distribute.get_replica_context():
            thread_id = tf.distribute.get_replica_context().replica_id_in_sync_group
        else:
            thread_id = "dummy"

        tf.print(thread_id, args[1], output_stream=f"file://{MASKS_SEEDS_PATH}")
        mask = fn(*args, **kwargs)
        return mask

    return wrapper


@pytest.fixture(scope="module")
def mirrored_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        return tf.distribute.get_strategy()
    num_of_replicas = 3
    strategy = tf.distribute.MirroredStrategy([f"GPU:{i}" for i in range(num_of_replicas)])
    return strategy


@pytest.mark.parametrize("quantization", [False, True], ids=["without_quantization", "with_quantization"])
@patch("nncf.tensorflow.sparsity.rb.operation.calc_rb_binary_mask", new=calc_rb_mask_decorator(calc_rb_binary_mask))
def test_distributed_masks_are_equal(quantization, mirrored_strategy):
    # Clean output file
    with contextlib.suppress(OSError):
        os.remove(MASKS_SEEDS_PATH)

    # Fill file with seeds
    num_of_replicas = 3
    with mirrored_strategy.scope():
        config = NNCFConfig.from_json(CONF)
        if quantization:
            config.update({"compression": [config["compression"], {"algorithm": "quantization"}]})
        model = TEST_MODELS["Conv2D"]()
        compression_state_to_skip_init = {BaseCompressionAlgorithmController.BUILDER_STATE: {}}
        algo, model = create_compressed_model(model, config, compression_state_to_skip_init)
        model.add_loss(algo.loss)
        compression_callbacks = create_compression_callbacks(algo, log_tensorboard=False)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(5e-4),
            metrics=["accuracy"],
        )

        dataset_len_per_replica = 10
        dataset_len = dataset_len_per_replica * num_of_replicas

        dummy_x = tf.random.normal((dataset_len,) + model.input_shape[1:])
        dummy_y = tf.random.normal((dataset_len,) + model.output_shape[1:])
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(num_of_replicas).with_options(options)

        model.fit(dataset, epochs=1, validation_split=0, callbacks=[compression_callbacks])
    # Check seeds in file
    with open(MASKS_SEEDS_PATH, "r", encoding="utf8") as f:
        seeds = f.readlines()
    seeds_per_replica = defaultdict(list)
    for row in seeds:
        replica_id, *seed = row.split()
        seeds_per_replica[replica_id].append((int(seed[0][1:]), int(seed[1][:-1])))

    # Check seeds are equal for all replicas
    for key, other_key in combinations(seeds_per_replica, 2):
        assert seeds_per_replica[key] == seeds_per_replica[other_key]
    # Check seeds differs during training
    assert len(set(seeds_per_replica["0"])) > 1
    # Remove temporary file
    os.remove(MASKS_SEEDS_PATH)


@pytest.mark.parametrize("local_mode", [False], ids=["global_loss"])
@pytest.mark.parametrize("model_name", list(TEST_MODELS.keys()), ids=list(TEST_MODELS.keys()))
class TestSparseModules:
    def test_create_loss__with_defaults(self, model_name, local_mode):
        _, algo, _ = get_basic_rb_sparse_model(model_name, local_mode)
        loss = algo.loss
        assert not loss.disabled
        assert loss.target_sparsity_rate == 0
        assert loss.p == 0.05

    REF_LOSS_IF_NOT_FROZEN = {
        "Dense": tf.fill((10,), 4.0),
        "Conv2D": tf.fill((3, 3, 1), 4.0),
        "Conv2DTranspose": tf.reshape(
            tf.constant(
                [[1, 2, 2, 2, 1], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [2, 4, 4, 4, 2], [1, 2, 2, 2, 1]], dtype=tf.float32
            ),
            (1, 5, 5, 1),
        ),
    }

    @pytest.mark.parametrize(
        ("mask_value", "zero_mask"),
        ((None, False), (0.0, True), (0.3, False), (-0.3, True)),
        ids=("default", "zero", "positive", "negative"),
    )
    def test_can_forward_sparse_module__with_frozen_mask(self, model_name, local_mode, mask_value, zero_mask):
        model, _, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=True)
        sm = model.layers[1]
        # Set weights
        kernel = get_weight_by_name(sm, "kernel")
        kernel.assign(tf.ones_like(kernel))
        # Set bias
        bias = get_weight_by_name(sm, "bias")
        bias.assign(tf.zeros_like(bias))
        if mask_value is not None:
            # Set mask
            mask = get_weight_by_name(sm, "mask")
            mask.assign(tf.fill(mask.shape, mask_value))
        input_ = tf.ones((1,) + model.input_shape[1:])
        output_ = model(input_)
        if zero_mask:
            assert tf.reduce_all(output_ == 0)
        else:
            assert tf.reduce_all(output_ == self.REF_LOSS_IF_NOT_FROZEN[model_name])

    @pytest.mark.parametrize(("frozen", "raising"), ((True, True), (False, False)), ids=("frozen", "not_frozen"))
    def test_calc_loss(self, model_name, frozen, raising, local_mode):
        model, algo, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=frozen)
        op = list(list(model.layers[1].weights_attr_ops.values())[0].values())[0]
        trainable = model.layers[1].ops_weights[op.name]["trainable"]
        assert tf.equal(trainable, tf.constant(not frozen))
        cls = SparseLoss

        loss = cls(algo.loss._target_ops)
        try:
            assert loss() == 0
        except ZeroDivisionError:
            pytest.fail("Division by zero")
        except tf.errors.InvalidArgumentError:
            if not raising:
                pytest.fail("Exception is not expected")

    @pytest.mark.parametrize("frozen", (False, True), ids=("sparsify", "frozen"))
    class TestWithSparsify:
        def test_can_freeze_mask(self, model_name, local_mode, frozen):
            model, _, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=frozen)
            rb_op = get_op_by_cls(model.layers[1], RBSparsifyingWeight)
            weights = model.layers[1].get_operation_weights(rb_op.name)
            assert tf.equal(weights["trainable"], tf.constant(not frozen))
            rb_op.freeze(weights)
            assert not weights["trainable"]

        def test_disable_loss(self, model_name, local_mode, frozen):
            model, algo, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=frozen)
            rb_op = get_op_by_cls(model.layers[1], RBSparsifyingWeight)
            weights = model.layers[1].get_operation_weights(rb_op.name)
            trainable = weights["trainable"]
            assert tf.equal(trainable, tf.constant(not frozen))
            loss = algo.loss
            loss.disable()
            assert not trainable

        def test_check_gradient_existing(self, model_name, local_mode, frozen):
            model, algo, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=frozen)

            algo.loss.set_target_sparsity_loss(1.0)
            dataset_len = (1,)
            dummy_x = tf.random.normal(dataset_len + model.input_shape[1:])
            dummy_y = tf.random.normal(dataset_len + model.output_shape[1:])

            loss_ce = tf.keras.losses.CategoricalCrossentropy()

            with tf.GradientTape() as tape:
                output = model(dummy_x, training=True)
                loss = loss_ce(dummy_y, output)

            grads = tape.gradient(loss, model.trainable_weights)
            grads_weights_paris = list(zip(grads, model.trainable_weights))
            for grad, weight in grads_weights_paris:
                if "mask" not in weight.name:
                    assert grad is not None
                else:
                    assert grad is None if frozen else grad is not None

        def test_masks_gradients(self, model_name, local_mode, frozen):
            model, algo, _ = get_basic_rb_sparse_model(model_name, local_mode, freeze=frozen)

            algo.loss.set_target_sparsity_loss(1.0)

            with tf.GradientTape() as tape:
                loss = algo.loss()
            grads = tape.gradient(loss, model.trainable_weights)
            if frozen:
                assert all(x is None for x in grads)
            else:
                for grad, weight in zip(grads, model.trainable_weights):
                    if "mask" in weight.name:
                        assert tf.reduce_all(grad > 0)

    @pytest.mark.parametrize(
        ("target", "expected_rate"),
        ((None, 0), (0, 1), (0.5, 0.5), (1, 0), (1.5, None), (-0.5, None)),
        ids=("default", "min", "middle", "max", "more_than_max", "less_then_min"),
    )
    def test_get_target_sparsity_rate(self, model_name, local_mode, target, expected_rate):
        config = get_empty_config()
        config["compression"] = {"algorithm": "rb_sparsity", "params": {"sparsity_init": 0}}
        _, algo, _ = get_basic_rb_sparse_model(model_name, local_mode, config=config)
        loss = algo.loss
        if target is not None:
            loss.target = target
        actual_rate = None
        try:
            actual_rate = loss.target_sparsity_rate
            if expected_rate is None:
                pytest.fail("Exception should be raised")
        except ValueError:
            if expected_rate is not None:
                pytest.fail("Exception is not expected")
        if expected_rate is not None:
            assert actual_rate == expected_rate


def test_loss_state():
    loss = SparseLoss([])

    # Test default state
    assert loss.get_state() == {"target": 1.0, "disabled": False, "p": 0.05}

    # Test get state
    loss.disable()
    loss.p = 0.5
    loss.set_target_sparsity_loss(0.1)
    state = loss.get_state()
    assert state["target"] == pytest.approx(0.9)
    assert state["disabled"]
    assert state["p"] == 0.5

    # Test load state
    new_state = {"target": 1.0, "disabled": False, "p": 0.05}
    loss.load_state(new_state)
    assert tf.equal(loss.target, tf.constant(new_state["target"]))
    assert loss.disabled == new_state["disabled"]
    assert loss.p == pytest.approx(new_state["p"])

    new_real_state = loss.get_state()
    assert new_real_state["target"] == pytest.approx(new_state["target"])
    assert new_real_state["disabled"] == new_state["disabled"]
    assert new_real_state["p"] == pytest.approx(new_state["p"])
