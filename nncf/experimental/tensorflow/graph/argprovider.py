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

from typing import Any
from typing import Tuple

import tensorflow as tf

from nncf.common.utils.registry import Registry


def replace_value_by_index(xs: Tuple[Any, ...], pos: int, value: Any) -> Tuple[Any, ...]:
    """
    Return a new instance of the tuple replacing the specified
    position with the new value.

    :param xs: A tuple.
    :param pos: Zero-based index of the item which should be replaced.
    :param value: New value.
    """
    return tuple(value if idx == pos else elem for idx, elem in enumerate(xs))


def check_port_id(port_id: int, min_port_id: int, max_port_id: int):
    if min_port_id <= port_id <= max_port_id:
        return
    raise ValueError(f'Unexpected `port_id`: {port_id}')


TF_ARG_PROVIDERS = Registry('TF_ARG_PROVIDERS')


class ArgProvider:
    """
    Base class for all argument providers. An `ArgProvider` instance
    describes how to extract input or output argument with specified
    `port_id` from the `args` and `kwargs`.
    """

    def get_input(self, input_port_id: int, args, kwargs) -> tf.Tensor:
        """
        Returns input Tensor with specified `input_port_id`.
        :param input_port_id: Zero-based index an input Tensor
            for TensorFlow operation.
        :return: A Tensor with specified `input_port_id`.
        """

    def get_output(self, output_port_id: int, args, kwargs) -> tf.Tensor:
        """
        Returns output Tensor with specified `output_port_id`.
        :param output_port_id: Zero-based index an output Tensor
            of TensorFlow operation.
        :return: A Tensor with specified `output_port_id`.
        """

    def set_input(self, input_port_id: int, value: tf.Tensor, args, kwargs):
        """
        Updates input Tensor with specified `input_port_id`.
        :param input_port_id: Zero-based index an input Tensor
            for TensorFlow operation which should be updated.
        :return: A tuple (args, kwargs) with updated Tensor.
        """

    def set_output(self, output_port_id: int, value: tf.Tensor, args, kwargs):
        """
        Updates output Tensor with specified `output_port_id`.
        :param output_port_id: Zero-based index an output Tensor
            for TensorFlow operation which should be updated.
        :return: A tuple (args, kwargs) with updated Tensor.
        """


@TF_ARG_PROVIDERS.register('Relu6')
@TF_ARG_PROVIDERS.register('Relu')
@TF_ARG_PROVIDERS.register('Mean')
@TF_ARG_PROVIDERS.register('AddV2')
@TF_ARG_PROVIDERS.register('Placeholder')
@TF_ARG_PROVIDERS.register('BiasAdd')
class SimpleOutputArgProvider(ArgProvider):
    def get_output(self, output_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return args[output_port_id]

    def set_output(self, output_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return replace_value_by_index(args, output_port_id, value), kwargs


@TF_ARG_PROVIDERS.register('ResizeNearestNeighbor')
class ResizeNearestNeighborArgProvider(ArgProvider):
    """
    Argument provider for the `ResizeNearestNeighbor` operation.
    """

    def get_output(self, output_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return args[output_port_id]

    def set_output(self, output_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return replace_value_by_index(args, output_port_id, value), kwargs

    def get_input(self, input_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(input_port_id, min_port_id=0, max_port_id=0)
        return args[0]

    def set_input(self, input_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(input_port_id, min_port_id=0, max_port_id=0)
        return replace_value_by_index(args, input_port_id, value), kwargs


@TF_ARG_PROVIDERS.register('Conv2D')
class Conv2DArgProvider(ArgProvider):
    """
    Argument provider of the `Conv2D` operation.
    Inputs:
        port_id: 0 - input tensor.
        port_id: 1 - filter tensor.
    Outputs:
        port_id: 0 - output tensor.
    """

    def get_input(self, input_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)

        if input_port_id == 0:
            return args[0]

        return kwargs['filter']  # input_port_id == 1

    def set_input(self, input_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)

        if input_port_id == 0:
            return replace_value_by_index(args, input_port_id, value), kwargs

        kwargs['filter'] = value
        return args, kwargs

    def get_output(self, output_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return args[output_port_id]

    def set_output(self, output_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)

        if len(args) > 1:
            raise ValueError(f'Unexpected `args`: {args}')

        return replace_value_by_index(args, output_port_id, value), kwargs


@TF_ARG_PROVIDERS.register('FusedBatchNormV3')
class FusedBatchNormV3ArgProvider(ArgProvider):
    """
    Argument provider of the `FusedBatchNormV3` operation.
    Outputs:
        port_id: 0 - output tensor (`y` key).
    """

    def get_output(self, output_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)
        return args[0].y

    def set_output(self, output_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(output_port_id, min_port_id=0, max_port_id=0)
        x = args[0]._replace(y=value)
        return replace_value_by_index(args, 0, x), kwargs


@TF_ARG_PROVIDERS.register('DepthwiseConv2dNative')
class DepthwiseConv2dNativeArgProvider(ArgProvider):
    """
    Argument provider of the `DepthwiseConv2dNative` operation.
    Inputs:
        port_id: 0 - input tensor (args[0]).
        port_id: 1 - filter tensor (args[1]).
    """

    def get_input(self, input_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)
        return args[input_port_id]

    def set_input(self, input_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)
        return replace_value_by_index(args, input_port_id, value), kwargs


@TF_ARG_PROVIDERS.register('MatMul')
class MatMulArgProvider(ArgProvider):
    """
    Argument provider of the `MatMul` operation.
    Inputs:
        port_id: 0 - input tensor (args[0]).
        port_id: 1 - filter (always?) tensor (args[1]).
    """

    def get_input(self, input_port_id: int, args, kwargs) -> tf.Tensor:
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)

        if len(args) == 0:
            return kwargs['a' if input_port_id == 0 else 'b']
        return args[input_port_id]

    def set_input(self, input_port_id: int, value: tf.Tensor, args, kwargs):
        check_port_id(input_port_id, min_port_id=0, max_port_id=1)

        if len(args) == 0:
            kwargs['a' if input_port_id == 0 else 'b'] = value
            return args, kwargs
        return replace_value_by_index(args, input_port_id, value), kwargs
