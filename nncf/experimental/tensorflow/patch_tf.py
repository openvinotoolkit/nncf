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

import functools
import inspect
from typing import Optional
from typing import List
from typing import Dict
from typing import Any

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn

from nncf.common.graph.transformations.commands import TargetType
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.experimental.tensorflow.context import get_current_context
from nncf.experimental.tensorflow.scope import get_op_name
from nncf.experimental.tensorflow.graph.argprovider import replace_value_by_index
from nncf.experimental.tensorflow.graph.transformations.commands import TFTargetPoint
from nncf.experimental.tensorflow.graph.argprovider import TF_ARG_PROVIDERS


class Hook:
    """
    Contains the NNCF operations and target point where
    these operations should be applied.
    """

    def __init__(self,
                 operations: List[NNCFOperation],
                 target_point: TFTargetPoint,
                 ops_weights: Dict[str, Any]):
        """
        Initializes the hook.

        :param operations: List of the NNCF operations in the correct order.
            The operation at index 0 is applied first, with index -1 last.
        :param target_point: A target point. Contains information on where the
            operations should be applied.
        :param ops_weights: Weights of the operations.
        """
        self._operations = operations
        self._target_point = target_point
        self._ops_weights = ops_weights

        arg_provider_cls = TF_ARG_PROVIDERS.registry_dict.get(self._target_point.op_type_name)
        if arg_provider_cls is None:
            raise ValueError(f'Unexpected type of the TensorFlow operation: {self._target_point.op_type_name}. '
                             'Register an `ArgProvider` instance for this type in the '
                             '`TF_ARG_PROVIDERS` registry, please.')

        self._arg_provider = arg_provider_cls()

    @property
    def operations(self) -> List[NNCFOperation]:
        return self._operations

    @property
    def target_point(self) -> TFTargetPoint:
        return self._target_point

    @property
    def is_pre_hook(self) -> bool:
        """
        Returns the boolean flag that specified whether the hook should be
        applied before or after target operation.

        :return: `True` if hook should be applied before target operation,
            `False` otherwise.
        """
        return self._target_point.type == TargetType.OPERATOR_PRE_HOOK

    def get_operation_weights(self, op_name: str) -> Any:
        """
        Returns weights of the operation with `op_name`.

        :param op_name: Name of the operation.
        :return: Weihts of the operation.
        """
        return self._ops_weights[op_name]

    def __call__(self, *args, **kwargs):
        """
        Applies this hook.

        :return: A tuple (args, kwargs)
        """
        if self.is_pre_hook:
            get_fn = self._arg_provider.get_input
            set_fn = self._arg_provider.set_input
        else:
            get_fn = self._arg_provider.get_output
            set_fn = self._arg_provider.set_output

        x = get_fn(self.target_point.port_id, args, kwargs)
        for op in self.operations:
            w = self.get_operation_weights(op.name)
            x = op(*(x, w, None))
        args_, kwargs_ = set_fn(self.target_point.port_id, x, args, kwargs)

        return args_, kwargs_


class TensorFlowOpWrapper:
    """
    Describes a wrapper around the method from the TensorFlow API.
    """

    def __init__(self, op, op_type_name: str):
        """
        Initializes a wrapper.

        :param op: Original method.
        :param op_type_name: Operation type name (name of the function
            from the `tf.raw_ops` module).
        """
        self._op = op
        self._op_type_name = op_type_name

    def __call__(self, *args, **kwargs):
        """
        Applies TensorFlow operation with compression extensions.
        """
        tracing_context = get_current_context()

        # Should we wrap current operation?
        if not tracing_context.wrap_ops:
            return self._op(*args, **kwargs)

        op_name = get_op_name(self._op_type_name, kwargs.get('name'))

        _pre_hooks = getattr(get_current_context().model, '_pre_hooks')
        _post_hooks = getattr(get_current_context().model, '_post_hooks')

        with tracing_context.enter(in_call=True, wrap_ops=False):
            # Apply pre-hooks
            args, kwargs = TensorFlowOpWrapper._apply_hooks(
                _pre_hooks.get(op_name, []),
                args,
                kwargs
            )

            # Apply TensorFlow operation
            outputs = self._op(*args, **kwargs)

            # Apply post-hooks
            (outputs,), _ = TensorFlowOpWrapper._apply_hooks(
                _post_hooks.get(op_name, []),
                (outputs,),
                {}
            )

        return outputs

    @staticmethod
    def _apply_hooks(hooks: List[Hook], args, kwargs):
        """
        Applies hooks.

        :param hooks: A list of hooks.
        :return: A tuple (args, kwargs).
        """
        for hook in hooks:
            args, kwargs = hook(*args, **kwargs)
        return args, kwargs


class TFPatcher:
    """
    Performs modifications of the TensorFlow code.
    """

    @staticmethod
    def patch_tf_operations() -> None:
        """
        Applies patches to the TensorFlow code.
        """
        op_type_name_to_op_info_map = TFPatcher._get_ops_info()
        for op_type_name, (fn, fn_name, module) in op_type_name_to_op_info_map.items():
            tf_op_wrapper = TensorFlowOpWrapper(fn, op_type_name)
            setattr(module, fn_name, tf_op_wrapper)

            if hasattr(module, op_type_name):
                setattr(module, op_type_name, tf_op_wrapper)

            # Wraps `fn` from the public API
            if hasattr(fn, '_tf_api_names'):
                tf_api_names = getattr(fn, '_tf_api_names')
                for api_name in tf_api_names:
                    items = api_name.split('.')
                    module_names = items[:-1]
                    name = items[-1]

                    curr_module = tf
                    for curr_name in module_names:
                        curr_module = getattr(curr_module, curr_name)
                    setattr(curr_module, name, tf_op_wrapper)

            # TODO(andrey-churkin): Changes references from the `tensorflow.python.ops.nn`
            # module because the Keras uses it (Only for TF versions: 2.4.x, 2.5.x). Need
            # to remove this for new versions.
            if getattr(nn, fn_name, None) is fn:
                setattr(nn, fn_name, tf_op_wrapper)

        ops.name_scope = TFPatcher._wrap_name_scope_internal_fn(ops.name_scope)
        ops.name_scope_v2.__enter__ = TFPatcher._wrap_name_scope_v2_enter_fn(ops.name_scope_v2.__enter__)
        tf.name_scope.__enter__ = TFPatcher._wrap_name_scope_v2_enter_fn(tf.name_scope.__enter__)

    @staticmethod
    def _get_ops_info(op_type_names: Optional[List[str]] = None):
        raw_ops = inspect.getmembers(tf.raw_ops, predicate=inspect.isfunction)
        op_type_name_to_fn_map = dict(raw_ops)

        if op_type_names is None:
            op_type_names = list(op_type_name_to_fn_map)

        op_type_name_to_op_info_map = {}
        for op_type_name in op_type_names:
            original_fn = op_type_name_to_fn_map[op_type_name]

            module = inspect.getmodule(original_fn)
            fn_name = original_fn.__name__
            fn = getattr(module, fn_name)

            op_type_name_to_op_info_map[op_type_name] = (fn, fn_name, module)

        return op_type_name_to_op_info_map

    @staticmethod
    def _wrap_name_scope_internal_fn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 4:
                args = replace_value_by_index(args, 3, False)
            else:
                kwargs['skip_on_eager'] = False
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def _wrap_name_scope_v2_enter_fn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracing_context = get_current_context()

            if tf.executing_eagerly() and tracing_context.in_call:
                obj, = args  # self
                eager_context = context.context()
                old_name = eager_context.scope_name
                name = obj._name  # pylint: disable=protected-access

                if not name:
                    scope_name = ''
                elif name[-1] == '/':
                    scope_name = name
                elif old_name:
                    scope_name = tracing_context.unique_name(old_name + name) + '/'
                else:
                    scope_name = name + '/'
                eager_context.scope_name = scope_name

                def _restore_name_scope(*_):
                    eager_context.scope_name = old_name

                obj._exit_fns.append(_restore_name_scope)  # pylint: disable=protected-access
            else:
                scope_name = func(*args, **kwargs)
            return scope_name

        return wrapper


def patch_tf_operations() -> None:
    """
    Applies patches to the TensorFlow code.
    """
    TFPatcher.patch_tf_operations()
