# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as opset

import nncf
from nncf.common.graph import NNCFGraph
from nncf.quantization.algorithms.weight_compression.openvino_backend import (
    remove_variadic_splits_after_compressed_weights,
)
from tests.openvino.native.models import ModelForRepack


def get_constant_element_types(model: ov.Model) -> list[ov.Type]:
    """Collects element types of all Constant nodes in the model."""
    types = {}
    for op in model.get_ops():
        if op.get_type_name() == "Constant":
            etype = op.get_element_type()
            types[etype] = types.get(etype, 0) + 1
    return types


def test_repack_weights_produces_u3_and_u2():
    model = ModelForRepack().ov_model

    element_types = get_constant_element_types(model)
    assert ov.Type.u3 not in element_types, "Did not expect u3 constant before repacking"
    assert ov.Type.u2 not in element_types, "Did not expect u2 constant before repacking"

    repacked_model = nncf.repack_weights(model)

    element_types = get_constant_element_types(repacked_model)
    assert ov.Type.u3 in element_types, "Expected u3 constant after repacking"
    assert ov.Type.u2 in element_types, "Expected u2 constant after repacking"

    assert element_types[ov.Type.u3] == 3, "Expected exactly three u3 constants after repacking"
    assert element_types[ov.Type.u2] == 3, "Expected exactly three u2 constants after repacking"


@pytest.mark.parametrize("dtype", [ov.Type.i4, ov.Type.u4, ov.Type.u3, ov.Type.u2, ov.Type.i8, ov.Type.u8])
@pytest.mark.parametrize("decompression", ["symmetric", "asymmetric", "grouped"])
@pytest.mark.parametrize("axis", [0, -2])
def test_remove_variadic_splits_after_compressed_weights(dtype: ov.Type, decompression: str, axis: int) -> None:
    shape = (6, 2, 2) if decompression == "grouped" else (6, 4)
    values = np.arange(24, dtype=np.uint8).reshape(shape) % 4
    weights = opset.constant(values, dtype=dtype, name="weights")
    decompressed = opset.convert(weights, ov.Type.f32)
    if decompression == "asymmetric":
        zero_point = opset.convert(opset.constant(np.ones((6, 1), dtype=np.uint8), dtype=dtype), ov.Type.f32)
        decompressed = opset.subtract(decompressed, zero_point)
    scale_shape = (6, 2, 1) if decompression == "grouped" else (6, 1)
    scale = opset.constant(np.arange(1, np.prod(scale_shape) + 1, dtype=np.float32).reshape(scale_shape))
    decompressed = opset.multiply(decompressed, scale)
    if decompression == "grouped":
        decompressed = opset.reshape(decompressed, [6, 4], False)
    split = opset.variadic_split(decompressed, np.int64(axis), np.array([1, 2, -1], dtype=np.int64))
    split.output(1).set_names({"middle"})
    model = ov.Model([*split.outputs(), decompressed.output(0)], [])
    expected = [ov.Tensor(output.get_element_type(), output.get_shape()) for output in model.outputs]
    assert model.evaluate(expected, [])

    assert remove_variadic_splits_after_compressed_weights(model, NNCFGraph()) is model

    assert all(node.get_type_name() != "VariadicSplit" for node in model.get_ops())
    assert "middle" in model.output(1).get_names()
    actual = [ov.Tensor(output.get_element_type(), output.get_shape()) for output in model.outputs]
    assert model.evaluate(actual, [])
    for actual_tensor, expected_tensor in zip(actual, expected):
        np.testing.assert_array_equal(actual_tensor.data, expected_tensor.data)
    compressed_constants = [
        node for node in model.get_ops() if node.get_type_name() == "Constant" and node.get_element_type() == dtype
    ]
    assert sorted(node.get_shape()[0] for node in compressed_constants) == (
        [1, 1, 2, 2, 3, 3, 6, 6] if decompression == "asymmetric" else [1, 2, 3, 6]
    )


@pytest.mark.parametrize("dtype", [ov.Type.i4, ov.Type.u4, ov.Type.u3, ov.Type.u2, ov.Type.i8, ov.Type.u8])
def test_remove_variadic_splits_before_summed_matmuls(dtype: ov.Type) -> None:
    activations = opset.parameter([2, 4], ov.Type.f32)
    weights = opset.constant(np.arange(24, dtype=np.uint8).reshape(6, 4) % 4, dtype=dtype)
    zero_point = opset.convert(
        opset.constant(np.array([[0], [1], [2], [3], [1], [2]], dtype=np.uint8), dtype=dtype), ov.Type.f32
    )
    scale = opset.constant(np.arange(1, 7, dtype=np.float32).reshape(6, 1) / 8)
    decompressed = opset.multiply(opset.subtract(opset.convert(weights, ov.Type.f32), zero_point), scale)
    split = opset.variadic_split(decompressed, np.int64(0), np.array([3, 3], dtype=np.int64))
    first_matmul = opset.matmul(activations, split.output(0), transpose_a=False, transpose_b=True)
    second_matmul = opset.matmul(activations, split.output(1), transpose_a=False, transpose_b=True)
    model = ov.Model([opset.add(first_matmul, second_matmul)], [activations])
    inputs = [ov.Tensor(np.arange(-4, 4, dtype=np.float32).reshape(2, 4) / 4)]
    expected = [ov.Tensor(ov.Type.f32, [2, 3])]
    assert model.evaluate(expected, inputs)

    remove_variadic_splits_after_compressed_weights(model, NNCFGraph())

    model.validate_nodes_and_infer_types()
    assert all(node.get_type_name() != "VariadicSplit" for node in model.get_ops())
    assert sum(node.get_type_name() == "MatMul" for node in model.get_ops()) == 2
    actual = [ov.Tensor(ov.Type.f32, [2, 3])]
    assert model.evaluate(actual, inputs)
    np.testing.assert_array_equal(actual[0].data, expected[0].data)


@pytest.mark.parametrize("axis", [1, -2])
@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("scale_shape", [(2, 6, 2, 1), (6, 2, 1), (2, 1, 2, 1)])
def test_remove_variadic_splits_moe_grouped_weights(axis: int, asymmetric: bool, scale_shape: tuple[int, ...]) -> None:
    activations = opset.parameter([2, -1, 8], ov.Type.f32)
    values = (np.arange(96, dtype=np.int8).reshape(2, 6, 2, 4) % 16) - 8
    weights = opset.constant(values, dtype=ov.Type.i4)
    decompressed = opset.convert(weights, ov.Type.f16)
    if asymmetric:
        zero_point = opset.convert(
            opset.constant(np.arange(12, dtype=np.int8).reshape(6, 2, 1) % 4, dtype=ov.Type.i4), ov.Type.f16
        )
        decompressed = opset.subtract(decompressed, zero_point)
    scale = opset.constant(np.arange(1, np.prod(scale_shape) + 1, dtype=np.float16).reshape(scale_shape) / 8)
    decompressed = opset.multiply(decompressed, scale)
    decompressed = opset.convert(opset.reshape(decompressed, [2, 6, 8], False), ov.Type.f32)
    channel_count = opset.gather(opset.shape_of(decompressed, "i32"), np.int64(axis), np.int32(0))
    half_channels = opset.divide(channel_count, opset.constant(np.int32(2)))
    remainder = opset.mod(channel_count, opset.constant(np.int32(2)))
    first_length = opset.add(half_channels, opset.convert(opset.greater(remainder, np.int32(0)), ov.Type.i32))
    lengths = opset.concat([opset.reshape(first_length, [1], False), opset.constant(np.array([-1], dtype=np.int32))], 0)
    split = opset.variadic_split(decompressed, np.int64(axis), lengths)
    gate = opset.matmul(activations, split.output(0), transpose_a=False, transpose_b=True)
    up = opset.matmul(activations, split.output(1), transpose_a=False, transpose_b=True)
    model = ov.Model([opset.multiply(opset.swish(gate), up)], [activations])
    inputs = [ov.Tensor(np.arange(-8, 8, dtype=np.float32).reshape(2, 1, 8) / 16)]
    expected = [ov.Tensor(ov.Type.f32, [2, 1, 3])]
    assert model.evaluate(expected, inputs)

    remove_variadic_splits_after_compressed_weights(model, NNCFGraph())

    model.validate_nodes_and_infer_types()
    assert all(node.get_type_name() != "VariadicSplit" for node in model.get_ops())
    actual = [ov.Tensor(ov.Type.f32, [2, 1, 3])]
    assert model.evaluate(actual, inputs)
    np.testing.assert_array_equal(actual[0].data, expected[0].data)
    compressed_weights = [
        node
        for node in model.get_ops()
        if node.get_type_name() == "Constant" and node.get_element_type() == ov.Type.i4 and len(node.get_shape()) == 4
    ]
    assert len(compressed_weights) == 2
    assert all(list(node.get_shape()) == [2, 3, 2, 4] for node in compressed_weights)


@pytest.mark.parametrize("scale_shape", [(), (4,), (1, 4)])
def test_remove_variadic_splits_with_broadcast_parameters(scale_shape: tuple[int, ...]) -> None:
    weights = opset.constant(np.arange(24, dtype=np.uint8).reshape(6, 4) % 4, dtype=ov.Type.u4)
    zero_point = opset.convert(opset.constant(1, dtype=ov.Type.u4), ov.Type.f16)
    decompressed = opset.subtract(opset.convert(weights, ov.Type.f16), zero_point)
    scale = opset.multiply(opset.constant(np.full(scale_shape, 0.5, dtype=np.float16)), opset.constant(np.float16(2)))
    decompressed = opset.convert(opset.multiply(decompressed, scale), ov.Type.f32)
    split = opset.variadic_split(decompressed, np.int64(0), np.array([1, 2, 3], dtype=np.int64))
    model = ov.Model(split.outputs(), [])
    expected = [ov.Tensor(output.get_element_type(), output.get_shape()) for output in model.outputs]
    assert model.evaluate(expected, [])

    remove_variadic_splits_after_compressed_weights(model, NNCFGraph())

    model.validate_nodes_and_infer_types()
    assert all(node.get_type_name() != "VariadicSplit" for node in model.get_ops())
    actual = [ov.Tensor(output.get_element_type(), output.get_shape()) for output in model.outputs]
    assert model.evaluate(actual, [])
    for actual_tensor, expected_tensor in zip(actual, expected):
        np.testing.assert_array_equal(actual_tensor.data, expected_tensor.data)


@pytest.mark.parametrize(
    "case", ["activation", "uncompressed", "reshape_prefix", "unsupported_op", "reshape_axis", "dynamic_lengths"]
)
def test_remove_variadic_splits_leaves_unsupported_graphs(case: str) -> None:
    parameters = []
    if case == "activation":
        source = opset.parameter([6, 4], ov.Type.f32)
        parameters.append(source)
    else:
        dtype = ov.Type.f32 if case == "uncompressed" else ov.Type.u4
        source = opset.convert(opset.constant(np.ones((6, 4), dtype=np.uint8), dtype=dtype), ov.Type.f32)
    source = opset.multiply(source, opset.constant(np.float32(0.5)))
    if case == "unsupported_op":
        source = opset.transpose(source, [0, 1])
    elif case == "reshape_axis":
        source = opset.reshape(source, [3, 8], False)
    elif case == "reshape_prefix":
        source = opset.reshape(source, [3, 4, 2], False)
    axis = 1 if case == "reshape_prefix" else 0
    lengths = opset.constant(np.array([1, -1], dtype=np.int64))
    if case == "dynamic_lengths":
        lengths = opset.parameter([2], ov.Type.i64)
        parameters.append(lengths)
    split = opset.variadic_split(source, np.int64(axis), lengths)
    model = ov.Model(split.outputs(), parameters)
    original_ops = set(model.get_ops())

    assert remove_variadic_splits_after_compressed_weights(model, NNCFGraph()) is model

    assert set(model.get_ops()) == original_ops
