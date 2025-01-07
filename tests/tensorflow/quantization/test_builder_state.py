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
import os
from functools import partial

import tensorflow as tf

from examples.tensorflow.classification.main import load_checkpoint
from examples.tensorflow.classification.main import load_compression_state
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.tensorflow import create_compression_callbacks
from nncf.tensorflow import register_default_init_args
from nncf.tensorflow.callbacks.checkpoint_callback import CheckpointManagerCallback
from nncf.tensorflow.graph.transformations.commands import TFAfterLayer
from nncf.tensorflow.graph.transformations.commands import TFBeforeLayer
from nncf.tensorflow.graph.transformations.commands import TFLayer
from nncf.tensorflow.graph.transformations.commands import TFLayerPoint
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from nncf.tensorflow.quantization.algorithm import QuantizationBuilder
from nncf.tensorflow.quantization.algorithm import QuantizationController
from nncf.tensorflow.quantization.algorithm import TFQuantizationPoint
from nncf.tensorflow.quantization.algorithm import TFQuantizationSetup
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.utils.state import TFCompressionState
from tests.cross_fw.shared.serialization import check_serialization
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.quantization.test_algorithm_quantization import check_default_qspecs
from tests.tensorflow.quantization.test_algorithm_quantization import check_specs_for_disabled_overflow_fix
from tests.tensorflow.quantization.utils import get_basic_quantization_config
from tests.tensorflow.test_bn_adaptation import get_dataset_for_test
from tests.tensorflow.test_callbacks import REF_CKPT_DIR


def test_quantization_configs__on_resume_with_compression_state(tmp_path, mocker):
    model = get_basic_conv_test_model()
    config = get_basic_quantization_config()
    init_spy = mocker.spy(QuantizationBuilder, "initialize")
    gen_setup_spy = mocker.spy(QuantizationBuilder, "_get_quantizer_setup")
    dataset = get_dataset_for_test(shape=[4, 4, 1])
    config = register_default_init_args(config, dataset, 10)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    assert isinstance(compression_ctrl, QuantizationController)
    init_spy.assert_called()
    gen_setup_spy.assert_called()
    saved_quantizer_setup = gen_setup_spy.spy_return
    check_serialization(saved_quantizer_setup, _quantization_setup_cmp)

    compression_state_to_load = _save_and_load_compression_state(compression_ctrl, tmp_path)

    init_spy.reset_mock()
    gen_setup_spy.reset_mock()

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, config, compression_state_to_load
    )
    assert isinstance(compression_ctrl, QuantizationController)

    init_spy.assert_not_called()
    gen_setup_spy.assert_not_called()
    check_default_qspecs(compression_model)

    builder = QuantizationBuilder(config)
    builder.load_state(compression_state_to_load["builder_state"])

    loaded_quantizer_setup = builder._quantizer_setup
    assert _quantization_setup_cmp(loaded_quantizer_setup, saved_quantizer_setup)


def _save_and_load_compression_state(compression_ctrl, tmp_path):
    checkpoint_path = tmp_path / "compression_state"
    checkpoint_to_save = tf.train.Checkpoint(compression_state=TFCompressionState(compression_ctrl))
    checkpoint_to_save.save(checkpoint_path)

    compression_state = load_compression_state(str(checkpoint_path.parent))

    return compression_state


def test_quantization_configs__disable_overflow_fix_and_resume_from_compression_state(tmp_path):
    model = get_basic_conv_test_model()

    config = get_basic_quantization_config()
    config["compression"].update({"overflow_fix": "disable"})
    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config, force_no_init=True)

    compression_state_to_load = _save_and_load_compression_state(compression_ctrl, tmp_path)

    compression_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        model, config, compression_state_to_load
    )
    assert isinstance(compression_ctrl, QuantizationController)
    check_specs_for_disabled_overflow_fix(compression_model)


def test_checkpoint_callback_make_checkpoints(mocker, tmp_path):
    save_freq = 2
    config = get_basic_quantization_config()
    config["compression"]["initializer"] = {
        "range": {"num_init_samples": 0},
        "batchnorm_adaptation": {"num_bn_adaptation_samples": 0},
    }
    gen_setup_spy = mocker.spy(QuantizationBuilder, "_get_quantizer_setup")

    model, compression_ctrl = create_compressed_model_and_algo_for_test(
        get_basic_conv_test_model(), config, force_no_init=True
    )
    assert isinstance(compression_ctrl, QuantizationController)

    quantizer_setup = gen_setup_spy.spy_return
    compression_callbacks = create_compression_callbacks(compression_ctrl, log_tensorboard=False)
    dataset_len = 8

    dummy_x = tf.random.normal((dataset_len,) + model.input_shape[1:])
    dummy_y = tf.random.normal((dataset_len,) + model.output_shape[1:])

    model.compile(loss=tf.losses.CategoricalCrossentropy())

    ckpt_path = tmp_path / "checkpoint"
    ckpt = tf.train.Checkpoint(model=model, compression_state=TFCompressionState(compression_ctrl))
    model.fit(
        dummy_x,
        dummy_y,
        epochs=5,
        batch_size=2,
        callbacks=[CheckpointManagerCallback(ckpt, str(ckpt_path), save_freq), *compression_callbacks],
    )

    assert sorted(os.listdir(ckpt_path)) == REF_CKPT_DIR[save_freq]

    new_compression_state = load_compression_state(ckpt_path)

    new_model, new_compression_ctrl = create_compressed_model_and_algo_for_test(
        get_basic_conv_test_model(), config, new_compression_state
    )
    new_model.compile(loss=tf.losses.CategoricalCrossentropy())
    new_ckpt = tf.train.Checkpoint(model=new_model, compression_state=TFCompressionState(new_compression_ctrl))
    load_checkpoint(new_ckpt, ckpt_path)

    builder = QuantizationBuilder(config)
    builder.load_state(new_compression_state["builder_state"])

    new_quantizer_setup = builder._quantizer_setup

    assert _quantization_setup_cmp(quantizer_setup, new_quantizer_setup)
    assert new_compression_ctrl.get_state() == compression_ctrl.get_state()
    assert tf.reduce_all([tf.reduce_all(w_new == w) for w_new, w in zip(new_model.weights, model.weights)])


DUMMY_STR = "dummy_str"


def _quantization_setup_cmp(qs1: TFQuantizationSetup, qs2: TFQuantizationSetup):
    if qs1.__class__ is qs2.__class__:
        return all(_quantization_point_cmp(qp1, qp2) for qp1, qp2 in zip(iter(qs1), iter(qs2)))
    return False


def _quantization_point_cmp(qp1: TFQuantizationPoint, qp2: TFQuantizationPoint):
    if qp1.__class__ is qp2.__class__:
        return (
            qp1.target_point == qp2.target_point
            and qp1.quantizer_spec == qp2.quantizer_spec
            and qp1.op_name == qp2.op_name
        )
    return False


def _check_and_add_insertion_point(quantization_point_factory, setup, target_point):
    point = quantization_point_factory(target_point=target_point)
    setup.add_quantization_point(point)
    check_serialization(target_point)
    check_serialization(point, _quantization_point_cmp)


GROUND_TRUTH_STATE = {
    "quantization_points": [
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"target_type": {"name": "OPERATOR_POST_HOOK"}},
            "target_point_class_name": "TargetPoint",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"layer_name": "dummy_str", "target_type": {"name": "OPERATOR_POST_HOOK"}},
            "target_point_class_name": "TFLayerPoint",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"layer_name": "dummy_str"},
            "target_point_class_name": "TFLayer",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"input_port_id": 0, "instance_idx": 0, "layer_name": "dummy_str"},
            "target_point_class_name": "TFBeforeLayer",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"instance_idx": 0, "layer_name": "dummy_str", "output_port_id": 0},
            "target_point_class_name": "TFAfterLayer",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {"layer_name": "dummy_str", "weights_attr_name": "dummy_str"},
            "target_point_class_name": "TFLayerWeight",
        },
        {
            "op_name": "dummy_str",
            "quantizer_spec": {
                "half_range": True,
                "mode": "symmetric",
                "narrow_range": True,
                "num_bits": 8,
                "per_channel": True,
                "signedness_to_force": None,
            },
            "target_point": {
                "layer_name": "dummy_str",
                "operation_name": "dummy_str",
                "weights_attr_name": "dummy_str",
            },
            "target_point_class_name": "TFOperationWithWeights",
        },
    ],
    "unified_scale_groups": [],
}


def test_quantizer_setup_serialization():
    setup = TFQuantizationSetup()

    quantizer_spec = TFQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=None,
        narrow_range=True,
        half_range=True,
        per_channel=True,
    )
    check_serialization(quantizer_spec)

    target_type = TargetType.OPERATOR_POST_HOOK
    check_serialization(target_type)

    quantization_point_factory = partial(TFQuantizationPoint, op_name=DUMMY_STR, quantizer_spec=quantizer_spec)
    check_insertion_point = partial(_check_and_add_insertion_point, quantization_point_factory, setup)

    check_insertion_point(TargetPoint(target_type))
    check_insertion_point(TFLayerPoint(target_type, layer_name=DUMMY_STR))
    check_insertion_point(TFLayer(layer_name=DUMMY_STR))
    check_insertion_point(TFBeforeLayer(layer_name=DUMMY_STR))
    check_insertion_point(TFAfterLayer(layer_name=DUMMY_STR))
    check_insertion_point(TFLayerWeight(layer_name=DUMMY_STR, weights_attr_name=DUMMY_STR))
    check_insertion_point(
        TFOperationWithWeights(layer_name=DUMMY_STR, weights_attr_name=DUMMY_STR, operation_name=DUMMY_STR)
    )

    check_serialization(setup, _quantization_setup_cmp)
    assert setup.get_state() == GROUND_TRUTH_STATE
