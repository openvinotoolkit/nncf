"""
 Copyright (c) 2021 Intel Corporation
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

from typing import Union, Dict, Tuple, List

import tensorflow as tf

from nncf.experimental.tensorflow.nncf_context import get_nncf_context


InputSignature = Union[tf.TensorSpec, Dict[str, tf.TensorSpec], Tuple[tf.TensorSpec, ...], List[tf.TensorSpec]]


def _add_names_to_input_signature(input_signature: InputSignature):
    if isinstance(input_signature, tf.TensorSpec):
        name = input_signature.name if input_signature.name else 'input_0'
        return tf.TensorSpec.from_spec(input_signature, name=name)

    if isinstance(input_signature, dict):
        _input_signature = {}
        for i, (k, spec) in enumerate(input_signature.items()):
            _input_signature[k] = tf.TensorSpec.from_spec(
                spec,
                name=spec.name if spec.name else f'input_{i}'
            )
        return _input_signature

    specs = []
    for i, spec in input_signature:
        specs.append(
            tf.TensorSpec.from_spec(
                spec,
                name=spec.name if spec.name else f'input_{i}'
            )
        )
    return input_signature.__class__(specs)


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
        self.__dict__['_hooks'] = []

    @property
    def nncf_operations(self):
        return [op for hook in getattr(self, '_hooks') for op in hook.operations]

    @property
    def input_signature(self):
        return self._input_signature

    def get_config(self):
        raise NotImplementedError

    def call(self, inputs, **kwargs):
        with get_nncf_context().enter(wrap_ops=True):
            x = self._apply_post_hooks_for_inputs(inputs)
            outputs = self._model(x, **kwargs)
        return outputs

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

        if isinstance(self.input_signature, tf.TensorSpec):
            post_hook = input_name_to_post_hook_map[self.input_signature.name]
            return post_hook(inputs)

        if isinstance(self.input_signature, dict) and isinstance(inputs, dict):
            _inputs = {}
            for k, x in inputs.items():
                input_name = self.input_signature[k].name
                post_hook = input_name_to_post_hook_map[input_name]
                _inputs[k] = post_hook(x)
            return _inputs

        _inputs = []
        for x, spec in zip(inputs, self.input_signature):
            post_hook = input_name_to_post_hook_map[spec.name]
            _inputs.append(post_hook(x))
        return inputs.__class__(_inputs)
