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

from typing import Union, Dict, Tuple, List, Any
import itertools

import tensorflow as tf

from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.experimental.tensorflow.context import get_current_context
from nncf.experimental.tensorflow.graph.transformations.commands import TFTargetPoint
from nncf.experimental.tensorflow.patch_tf import Hook


InputSignature = Union[tf.TensorSpec, Dict[str, tf.TensorSpec], Tuple[tf.TensorSpec, ...], List[tf.TensorSpec]]


def _add_names_to_input_signature(input_signature: InputSignature):
    xs = tf.nest.flatten(input_signature)
    ys = []
    for i, spec in enumerate(xs):
        ys.append(
            tf.TensorSpec.from_spec(
                spec,
                name=spec.name if spec.name else f'input_{i}'
            )
        )

    return tf.nest.pack_sequence_as(input_signature, ys)


class NNCFNetwork(tf.keras.Model):
    """
    Wraps the Keras model.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 input_signature: InputSignature,
                 **kwargs):
        """
        Initializes the NNCF network.

        :param model: Keras model.
        :param input_signature: Input signature of the moodel.
        """
        super().__init__(**kwargs)
        self._model = model
        self._input_signature = _add_names_to_input_signature(input_signature)

        # The `__setattr__` was was overridden inside superclasses.
        # This workaround allows not add dependencies from hooks to the model.
        # See `tensorflow.python.training.tracking.autotrackable.AutoTrackable`
        # class for more details.
        self.__dict__['_pre_hooks'] = {}  # type: Dict[str, List[Hook]]
        self.__dict__['_post_hooks'] = {}  # type: Dict[str, List[Hook]]

    @property
    def nncf_operations(self) -> List[NNCFOperation]:
        """
        Returns list of the NNCF operations which were added to the NNCF network.

        :return: List of the NNCF operations.
        """
        return [op for hook in getattr(self, '_hooks') for op in hook.operations]

    @property
    def input_signature(self) -> InputSignature:
        """
        Returns input signature of the model.

        :return: Input signature of the model.
        """
        return self._input_signature

    def get_nncf_operations_with_params(self) -> List[Tuple[NNCFOperation, Any]]:
        return [
            (op, hook.get_operation_weights(op.name)) \
                 for hook in getattr(self, '_hooks') for op in hook.operations
        ]

    def get_config(self):
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        """
        Calls the model on new inputs and returns the outputs as tensors.
        We call the model inside the tracing context to add the NNCF
        operations to the graph of the model.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :return: A tensor if there is a single output, or a list of tensors
            if there are more than one outputs.
        """
        xs = self._apply_post_hooks_for_inputs(inputs)
        with get_current_context().enter(in_call=True, wrap_ops=True, model=self):
            outputs = self._model(xs, **kwargs)
        return outputs

    def insert_at_point(self, point: TFTargetPoint, ops: List[NNCFOperation]) -> None:
        """
        Inserts the list of the NNCF operations according to the target point.

        :param point: The location where operations should be inserted.
        :param ops: List of the NNCF operarions.
        """
        ops_weights = {op.name: op.create_variables(self) for op in ops}
        hook = Hook(ops, point, ops_weights)
        hooks = getattr(self, '_pre_hooks') if hook.is_pre_hook else getattr(self, '_post_hooks')
        # TODO(andrey-churkin): What we should do if the hook with the same `target_point`
        # already exists inside `hooks`? Is it a valid case?
        hooks.setdefault(hook.target_point.op_name, []).append(hook)

    @property
    def _hooks(self):
        pre_hooks = getattr(self, '_pre_hooks')
        post_hooks = getattr(self, '_post_hooks')
        return itertools.chain(*pre_hooks.values(), *post_hooks.values())

    def _apply_post_hooks_for_inputs(self, inputs):
        """
        Applies post-hooks to inputs.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :return: Modified input tensor, or dict/list/tuple of input tensors.
        """
        input_name_to_post_hook_map = {
            hook.target_point.op_name: hook for hook in getattr(self, '_hooks') \
            if hook.target_point.op_type_name == 'Placeholder'
        }

        if not input_name_to_post_hook_map:
            return inputs

        xs = tf.nest.flatten(inputs)
        ys = tf.nest.flatten(self.input_signature)
        flat_sequence = []
        for input_tensor, input_spec in zip(xs, ys):
            post_hook = input_name_to_post_hook_map[input_spec.name]
            (ans,), _ = post_hook(input_tensor)
            flat_sequence.append(ans)
        return tf.nest.pack_sequence_as(inputs, flat_sequence)
