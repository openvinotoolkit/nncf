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

from typing import Optional
import threading

import tensorflow as tf


_CURRENT_CONTEXT = threading.local()


def get_current_context():
    """
    Returns current active `TFTracingContext`.

    :return: Tracing context.
    """
    tracing_context = getattr(_CURRENT_CONTEXT, 'tracing_context', None)
    if tracing_context is None:
        tracing_context = TFTracingContext()
        setattr(_CURRENT_CONTEXT, 'tracing_context', tracing_context)
    return tracing_context


class TFTracingContextState:
    """
    Contains values that describe a state of the `TFTracingContext`.
    """

    def __init__(self,
                 in_call: bool = False,
                 wrap_ops: bool = False,
                 model: Optional[tf.keras.Model] = None):
        """
        Initializes the `TFTracingContextState` instance.

        :param in_call: Whether currently inside the `call()` method of a `tf.keras.Model`.
        :param wrap_ops: Whether currently adding the compression pre-hooks and post-hooks
            to TensorFlow operations.
        :param model: The Keras model whose `call()` method is currently active.
            The `None` value is specified that model is undefined at this moment. This is only
            possible when `in_call` is equal to `False`.
        """
        self._in_call = in_call
        self._wrap_ops = wrap_ops

        if model is None and in_call:
            raise ValueError(
                f'Inconsisten values `{in_call}` and `{model}` for `in_call` and `model` parameters. '
                'The `None` value is specified that model is undefined at this moment. This is only '
                'possible when `in_call` is equal to `False`.'
            )

        self._model = model

    @property
    def in_call(self) -> bool:
        return self._in_call

    @property
    def wrap_ops(self) -> bool:
        return self._wrap_ops

    @property
    def model(self) -> tf.keras.Model:
        return self._model


class TFTracingContext:
    """
    Contains information about should we wrap the TensorFlow
    operation or not.
    """

    def __init__(self):
        """
        Initializes the `TFTracingContext` instance.
        """
        self._state = TFTracingContextState()
        # Maps a name used in the graph to the next id to use for that name.
        self.names_in_use = {}

    @property
    def model(self) -> Optional[tf.keras.Model]:
        return self.state.model

    @property
    def in_call(self) -> bool:
        return self.state.in_call

    @property
    def wrap_ops(self) -> bool:
        return self.state.wrap_ops

    def enter(self,
              in_call: bool,
              wrap_ops: bool,
              model: Optional[tf.keras.Model] = None):
        """
        Pushes parameters onto the tracing context.

        :param in_call: Whether currently inside the `call()` method of a model.
        :param wrap_ops: Whether currently adding the compression pre-hooks and post-hooks
            to TensorFlow operations.
        :param model: The Keras model whose `call()` method is currently active.
        """
        model = self.state.model if model is None else model
        next_state = TFTracingContextState(in_call, wrap_ops, model)
        return TFTracingContextManager(self, next_state)

    def unique_name(self, name: str) -> str:
        """
        Returns a unique operation name for `name`.

        For more details, please, see implementation of
        the `tf.Graph.unique_name()` method.

        :param name: The name for an operation.
        :return: Unique name.
        """
        name_key = name.lower()

        i = self.names_in_use.get(name_key, 0)
        self.names_in_use[name_key] = i + 1

        if i > 0:
            base_name_key = name_key
            # Make sure the composed name key is not already used.
            while name_key in self.names_in_use:
                name_key = f'{base_name_key}_{i}'
                i += 1

            # Mark the composed name_key as used in case someone wants
            # to call unique_name('name_1').
            self.names_in_use[name_key] = 1

            # Return the new name with the original capitalization of the given name.
            name = f'{name}_{i - 1}'
        return name

    @property
    def state(self) -> TFTracingContextState:
        return self._state

    def load_state(self, state: TFTracingContextState) -> None:
        self._state = state


class TFTracingContextManager:
    """
    Context manager for the tracing context.
    """

    def __init__(self,
                 tracing_context: TFTracingContext,
                 next_state: TFTracingContextState):
        """
        Initializes the tracing context manager.

        :param tracing_context: Tracing context.
        :param next_state: Next state of the tracing context which
            should be applied.
        """
        self._tracing_context = tracing_context
        self._next_state = next_state
        self._prev_state = None

    def __enter__(self):
        self._prev_state = self._tracing_context.state
        self._tracing_context.load_state(self._next_state)

    def __exit__(self, *exc):
        self._tracing_context.load_state(self._prev_state)

        if not self._tracing_context.in_call:
            self._tracing_context.names_in_use = {}
